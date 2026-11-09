# IMPORTS *****************************************************************************************
# External libraries
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial
from pathlib import Path
import inspect
import os
import warnings
import numpy as np
import pyvips
from PIL import Image
from tqdm.auto import tqdm
from torch.utils.data import Dataset

# Internal imports ********************************************************************************
from utils.clean_empty import TileFilter

# Helper functions ********************************************************************************
def _index_slide(slide_path, level, tile_size, stride, tile_filter):
    """
    Index one slide: return its TIFF level pages, the (x, y) of every tile that passes the filter,
    and the number of tiles that could not be read (damaged data in the file).

    Module-level (not a method) so it can run in a worker process. The coordinate list is
    empty when the slide does not have the requested level.

    * Arguments:
    slide_path: The path to the slide image
    level: The pyramid level to tile, counted from 1
    tile_size: The width and height of each square tile
    stride: The distance between neighboring tiles
    tile_filter: The optional filter used to skip background or invalid tiles
    """

    # STEP 1: Resolve the pyramid pages and skip slides without the requested level
    level_pages = WSITileDataset._get_level_pages(slide_path)

    if level > len(level_pages):
        return level_pages, [], 0

    image = pyvips.Image.tiffload(
        str(slide_path),
        page=level_pages[level - 1],
        access="random",
    )

    # STEP 2: Walk through every valid tile coordinate
    coordinates = []
    unreadable = 0

    for y in range(0, image.height - tile_size + 1, stride):
        for x in range(0, image.width - tile_size + 1, stride):
            # STEP 3: Skip tiles rejected by the optional filter
            if tile_filter is not None:
                # Damaged JPEG tiles make libvips raise; skip them instead of failing the whole dataset
                try:
                    tile = image.crop(x, y, tile_size, tile_size)
                    array = WSITileDataset._to_numpy(tile)[..., :3]
                except pyvips.Error:
                    unreadable += 1
                    continue

                if not tile_filter(array):
                    continue

            coordinates.append((x, y))

    return level_pages, coordinates, unreadable


# Main class **************************************************************************************
class WSITileDataset(Dataset):
    """
    Build a dataset of fixed-size tiles from whole-slide SVS images.

    The dataset indexes every SVS file in a directory and returns image tiles
    from the requested pyramid level, read from its TIFF pages. Slides that
    do not have the requested level are skipped (see skipped_slides).

    * Arguments:
    data_dir: The directory containing the SVS slide images
    level: The pyramid level to read from each slide, from 1 (full resolution) to MAX_LEVEL
    tile_size: The width and height of each square tile
    stride: The distance between neighboring tiles
    basic_transform: The transform applied to each tile before it is returned
    tile_filter: The optional filter used to skip background or invalid tiles
    index_workers: The number of processes used to index slides (None = all CPU cores, 1 = serial);
        the tile filter must be picklable when more than one is used
    show_progress: Whether to show a progress bar (slides indexed, elapsed time and estimated time left)
    """

    # Workable file extensions
    EXTENSIONS = {".svs"}

    # Highest pyramid level that may be requested
    MAX_LEVEL = 4

    # Class initialization
    def __init__(
        self,
        data_dir: str | Path,
        level: int = 1,
        tile_size: int = 224,
        stride: int | None = None,
        basic_transform=None,
        tile_filter: TileFilter | None = None,
        index_workers: int | None = None,
        show_progress: bool = True,
    ):
        """
        Initialize the tile dataset and index every valid tile.

        * Arguments:
        data_dir: The directory containing the SVS slide images
        level: The pyramid level to read from each slide, from 1 (full resolution) to MAX_LEVEL
        tile_size: The width and height of each square tile
        stride: The distance between neighboring tiles
        basic_transform: The transform applied to each tile before it is returned
        tile_filter: The optional filter used to skip background or invalid tiles
        index_workers: The number of processes used to index slides (None = all CPU cores, 1 = serial)
        show_progress: Whether to show a progress bar while the slides are indexed
        """

        # STEP 1: Validate dataset parameters
        if not 1 <= level <= self.MAX_LEVEL:
            raise ValueError(
                f"level must be between 1 and {self.MAX_LEVEL}, got {level}"
            )

        if tile_size <= 0:
            raise ValueError("tile_size must be > 0")

        if stride is not None and stride <= 0:
            raise ValueError("stride must be > 0")

        # STEP 2: Store dataset configuration
        self.data_dir = Path(data_dir)
        self.level = level
        self.tile_size = tile_size
        self.stride = stride if stride is not None else tile_size
        self.tile_filter = tile_filter
        self.index_workers = index_workers if index_workers is not None else os.cpu_count() or 1
        self.show_progress = show_progress
        # TODO: I should only transform if pass filter, right???
        self.transform = basic_transform

        # STEP 2b: Single-slot cache for the last opened slide image, valid
        # because create_dataloader forces shuffle=False/num_workers=0, so
        # __getitem__ calls are always slide-major and sequential.
        self._cached_slide_path: Path | None = None
        self._cached_image: pyvips.Image | None = None

        # STEP 3: Validate the input directory
        if not self.data_dir.is_dir():
            raise FileNotFoundError(
                f"Directory does not exist: {self.data_dir}"
            )

        # STEP 4: Collect supported slide paths
        self.slide_paths = sorted(
            path
            for path in self.data_dir.iterdir()
            if path.is_file()
            and path.suffix.lower() in self.EXTENSIONS
        )

        if not self.slide_paths:
            raise FileNotFoundError(
                f"No .svs images found in {self.data_dir}"
            )

        # STEP 5: Build the tile index
        self._level_pages: dict[Path, list[int]] = {}
        self.tiles: list[tuple[Path, int, int]] = []
        self.skipped_slides: list[tuple[Path, int]] = []
        self.unreadable_tiles: list[tuple[Path, int]] = []
        self._index_tiles()

    def __getstate__(self) -> dict:
        """
        Drop the cached pyvips image when the dataset is pickled (it cannot be pickled and
        each DataLoader worker reopens its own).
        """

        state = self.__dict__.copy()
        state["_cached_slide_path"] = None
        state["_cached_image"] = None

        return state

    def get_config(self) -> dict:
        """
        Return the tiling parameters as plain Python values (safe to save next to features).

        The tile filter is described by its function name and the effective value of every
        argument except the tile itself, so defaults are recorded as well as overrides.
        """

        # STEP 1: Describe the tile filter (None when no filter is used)
        tile_filter = None

        if self.tile_filter is not None:
            func = self.tile_filter.func if isinstance(self.tile_filter, partial) else self.tile_filter
            bound = inspect.signature(self.tile_filter).parameters
            defaults = {
                name: param.default
                for name, param in bound.items()
                if param.default is not inspect.Parameter.empty
            }
            tile_filter = {"name": func.__name__, "params": defaults}

        # STEP 2: Return the tiling configuration
        return {
            "level": self.level,
            "tile_size": self.tile_size,
            "stride": self.stride,
            "tile_filter": tile_filter,
        }

    def _index_tiles(self) -> None:
        """
        Index all tile coordinates for every slide in the dataset.

        The optional tile filter is applied during indexing so filtered tiles
        are never returned by __getitem__. A progress bar over slides (with elapsed time and
        estimated time left) is shown when show_progress is True.
        """

        # STEP 1: Index every slide, in parallel across processes; results are stored by slide position
        # so the sorted order is kept, while the progress bar advances as soon as any slide is done
        index_slide = partial(
            _index_slide,
            level=self.level,
            tile_size=self.tile_size,
            stride=self.stride,
            tile_filter=self.tile_filter,
        )
        workers = min(self.index_workers, len(self.slide_paths))
        bar = dict(total=len(self.slide_paths), desc="Indexing slides", unit="slide", disable=not self.show_progress)

        if workers > 1:
            results = [None] * len(self.slide_paths)

            with ProcessPoolExecutor(max_workers=workers) as pool:
                futures = {pool.submit(index_slide, path): position for position, path in enumerate(self.slide_paths)}

                for future in tqdm(as_completed(futures), **bar):
                    results[futures[future]] = future.result()
        else:
            results = [index_slide(path) for path in tqdm(self.slide_paths, **bar)]

        # STEP 2: Collect the results slide by slide
        for slide_path, (level_pages, coordinates, unreadable) in zip(self.slide_paths, results):
            self._level_pages[slide_path] = level_pages

            # STEP 3: Skip slides that do not have the requested pyramid level
            if self.level > len(level_pages):
                self.skipped_slides.append((slide_path, len(level_pages)))
                continue

            # STEP 4: Keep the tiles that passed the optional filter
            self.tiles.extend((slide_path, x, y) for x, y in coordinates)

            if unreadable:
                self.unreadable_tiles.append((slide_path, unreadable))

        # STEP 5: Fail when no slide has the requested level
        if len(self.skipped_slides) == len(self.slide_paths):
            raise ValueError(f"No slide has pyramid level {self.level}")

        # STEP 6: Report the slides left out of the dataset
        if self.skipped_slides:
            details = ", ".join(
                f"{path.name} ({count} levels)"
                for path, count in self.skipped_slides[:5]
            )
            warnings.warn(
                f"Skipped {len(self.skipped_slides)} slide(s) without "
                f"level {self.level}: {details}"
                + (" ..." if len(self.skipped_slides) > 5 else "")
            )

        # STEP 7: Report the slides that contain damaged (unreadable) tiles
        if self.unreadable_tiles:
            details = ", ".join(
                f"{path.name} ({count} tiles)"
                for path, count in self.unreadable_tiles[:5]
            )
            warnings.warn(
                f"Left out unreadable tiles in {len(self.unreadable_tiles)} slide(s): {details}"
                + (" ..." if len(self.unreadable_tiles) > 5 else "")
            )

    @staticmethod
    def _get_level_pages(slide_path: Path) -> list[int]:
        """
        Return the TIFF page index of every pyramid level in an SVS file.

        SVS stores each level as a TIFF page, interleaved with a thumbnail and
        label/macro pages. Only the pyramid levels are tiled, so tiled pages
        are the levels, ordered from the base level downwards.

        * Arguments:
        slide_path: The path to the slide image
        """

        # STEP 1: Read the page count from the first page
        first_page = pyvips.Image.tiffload(str(slide_path), page=0)
        page_count = int(first_page.get("n-pages"))

        # STEP 2: Keep the pages that are tiled
        level_pages = []

        for page in range(page_count):
            image = pyvips.Image.tiffload(str(slide_path), page=page)

            if image.get_typeof("tile-width") != 0:
                level_pages.append(page)

        # STEP 3: Fail clearly when no pyramid levels were found
        if not level_pages:
            raise ValueError(f"{slide_path.name} has no tiled pyramid levels")

        return level_pages

    def _open_level(
        self,
        slide_path: Path,
        level: int,
    ) -> pyvips.Image:
        """
        Open a requested pyramid level from an SVS image.

        * Arguments:
        slide_path: The path to the slide image
        level: The pyramid level to open, counted from 1
        """

        # STEP 1: Resolve the TIFF page of the requested level
        level_pages = self._level_pages[slide_path]

        # STEP 2: Open that page
        return pyvips.Image.tiffload(
            str(slide_path),
            page=level_pages[level - 1],
            access="random",
        )

    @staticmethod
    def _to_numpy(image: pyvips.Image) -> np.ndarray:
        """
        Convert an 8-bit pyvips image into a copied NumPy array.

        * Arguments:
        image: The pyvips image to be converted
        """

        return np.ndarray(
            buffer=image.write_to_memory(),
            dtype=np.uint8,
            shape=(image.height, image.width, image.bands),
        ).copy()

    def __len__(self) -> int:
        """
        Return the number of indexed tiles.
        """

        return len(self.tiles)

    def _get_cached_level_image(self, slide_path: Path) -> pyvips.Image:
        """
        Return the requested slide's pyramid level, reusing the last opened
        image when consecutive calls target the same slide.

        * Arguments:
        slide_path: The path to the slide image
        """

        if slide_path != self._cached_slide_path:
            self._cached_image = self._open_level(slide_path, self.level)
            self._cached_slide_path = slide_path

        return self._cached_image

    def __getitem__(self, idx: int) -> dict:
        """
        Return one transformed tile and its slide metadata.

        * Arguments:
        idx: The integer index of the tile to read
        """

        # STEP 1: Resolve the tile coordinates
        slide_path, x, y = self.tiles[idx]

        # STEP 2: Read and crop the tile from the requested pyramid level
        image = self._get_cached_level_image(slide_path)
        tile = image.crop(
            x,
            y,
            self.tile_size,
            self.tile_size,
        )

        # STEP 3: Convert the tile into an RGB PIL image
        array = self._to_numpy(tile)
        rgb_array = array[..., :3].astype(np.uint8, copy=False)
        output_image = Image.fromarray(rgb_array, mode="RGB")

        # STEP 4: Apply the optional transform
        if self.transform is not None:
            output_image = self.transform(output_image)

        # STEP 5: Return the image and metadata
        return {
            "image": output_image,
            "slide": slide_path.name,
            "level": self.level,
            "x": x,
            "y": y,
        }
