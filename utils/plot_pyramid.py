# IMPORTS *****************************************************************************************
# External libraries
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyvips
from matplotlib.patches import Rectangle

# Internal imports ********************************************************************************
from utils.clean_empty import background_filter
from utils.cls_overview import INK_2, SERIES, STYLE, SURFACE
from utils.dataset import WSITileDataset
from utils.figure_captions import split_with_notes
from utils.labels import BRACS_CLASSES
from utils.slide_stats import _parse_description


# Helper functions ********************************************************************************
def _page_image(slide_path: Path, page: int) -> pyvips.Image:
    """Open one TIFF page (pyramid level) of a slide lazily."""
    return pyvips.Image.tiffload(str(slide_path), page=page)


def _to_array(image: pyvips.Image) -> np.ndarray:
    """Realise a pyvips image as an RGB uint8 array [height, width, 3]."""
    array = np.ndarray(buffer=image.write_to_memory(), dtype=np.uint8, shape=(image.height, image.width, image.bands))
    return array[..., :3]


def _read_region(slide_path: Path, page: int, x: int, y: int, size: int, out_px: int | None = None, smooth: bool = True) -> np.ndarray:
    """
    Read a square region of one page, optionally resized to out_px pixels wide.

    The region is read lazily from the tiled file, so a region of a full-resolution level is streamed
    through the resize and never held in memory at full size.
    """
    region = _page_image(slide_path, page).crop(int(x), int(y), int(size), int(size))
    if out_px is not None and out_px != size:
        region = region.resize(out_px / size, kernel="lanczos3" if smooth and out_px < size else "nearest")
    return _to_array(region)


def _tissue_window(thumbnail: np.ndarray, window: int) -> tuple[int, int]:
    """Top-left corner (x, y) of the window of the given size (in thumbnail pixels) with the most tissue."""
    rgb = thumbnail.astype(int)
    tissue = (rgb.max(-1) - rgb.min(-1) > 25) & (rgb.max(-1) < 235)         # coloured and not white glass
    integral = np.pad(tissue.cumsum(0).cumsum(1), ((1, 0), (1, 0)))
    window = min(window, *tissue.shape)
    sums = integral[window:, window:] - integral[:-window, window:] - integral[window:, :-window] + integral[:-window, :-window]
    y, x = np.unravel_index(np.argmax(sums), sums.shape)

    return int(x), int(y)


def _slide_levels(slide_path: Path) -> tuple[list[int], list[tuple[int, int]]]:
    """Return the TIFF page and (width, height) of every pyramid level of a slide."""
    pages = WSITileDataset._get_level_pages(slide_path)
    sizes = [(_page_image(slide_path, p).width, _page_image(slide_path, p).height) for p in pages]

    return pages, sizes


# Main functions **********************************************************************************
def plot_pyramid_levels(
    slide_path: str | Path,
    tile_size: int = 224,
    levels: tuple[int, ...] = (1, 2, 3, 4),
    display_px: int = 480,
    show: bool = True,
) -> dict:
    """
    Show the same piece of tissue at every pyramid level, and the tile the model receives at each level.

    A window with plenty of tissue is chosen automatically. Top row: the window at each level (the
    square marks one tile). Bottom row: the tile of tile_size pixels around the window centre, at the
    native pixels of each level (nearest-neighbour enlargement, so the resolution is visible).

    * Arguments:
    slide_path: The .svs file
    tile_size: The tile edge in pixels
    levels: The pyramid levels to show, counted from 1 (full resolution)
    display_px: The width in pixels of the top-row panels
    show: Whether to call plt.show() (the figure is returned either way)

    Returns a dict with the figure, the downsample factor per level, the window position and size at level 1.
    """

    # STEP 1: Pyramid levels of the slide and the window size
    slide_path = Path(slide_path)
    pages, sizes = _slide_levels(slide_path)
    width1, height1 = sizes[0]
    downsample = {lvl: width1 / sizes[lvl - 1][0] for lvl in levels if lvl <= len(pages)}
    mpp = _parse_description(_page_image(slide_path, pages[0]).get("image-description"))["mpp"]
    window = int(min(tile_size * max(downsample.values()) * 1.5, 0.9 * min(width1, height1)))

    # STEP 2: Window with the most tissue, found on the smallest level
    last = max(downsample)
    thumb_full = _to_array(_page_image(slide_path, pages[last - 1]))
    tx, ty = _tissue_window(thumb_full, max(1, int(window / downsample[last])))
    x1, y1 = int(tx * downsample[last]), int(ty * downsample[last])
    cx, cy = x1 + window / 2, y1 + window / 2

    # STEP 3: Read the window and the centred tile at every level
    fields, tiles = {}, {}
    for lvl, d in downsample.items():
        side = max(int(window / d), 1)
        fields[lvl] = _read_region(slide_path, pages[lvl - 1], x1 / d, y1 / d, side, display_px)
        tx0 = int(np.clip(cx / d - tile_size / 2, 0, sizes[lvl - 1][0] - tile_size))
        ty0 = int(np.clip(cy / d - tile_size / 2, 0, sizes[lvl - 1][1] - tile_size))
        tiles[lvl] = _read_region(slide_path, pages[lvl - 1], tx0, ty0, tile_size)

    # STEP 4: Figure
    notes = [(0, len(levels),
              f"Top row: the same {window * mpp / 1000:.1f} × {window * mpp / 1000:.1f} mm field of view at each pyramid level of the slide {slide_path.stem}; the square marks the area of one "
              f"{tile_size} × {tile_size} px tile at that level. Bottom row: that tile at its native pixels, as the model receives it (enlarged without interpolation).\n"
              "Each step down the pyramid divides the resolution by the downsample factor and multiplies the tissue covered by a tile by the same factor squared in area. "
              "Level 1 shows individual nuclei and chromatin texture; coarser levels show tissue architecture (glands, ducts, stroma) and lose cellular detail. "
              "The level therefore decides which morphological features the embedding can encode, and a level has to be chosen with the diagnostic task in mind.")]
    with plt.rc_context(STYLE):
        fig = plt.figure(figsize=(3.6 * len(levels), 9.5), layout="constrained")
        fig.suptitle(f"{slide_path.name} · pyramid levels {', '.join(map(str, levels))}", fontsize=16, weight="bold", x=0.01, ha="left")
        top = split_with_notes(fig, notes, [1] * len(levels), 3.6 * len(levels), plot_height=7.0, note_height=1.8)
        axes = top.subplots(2, len(levels), height_ratios=[1, 1])
        for col, lvl in enumerate(levels):
            ax_field, ax_tile = axes[0, col], axes[1, col]
            for ax in (ax_field, ax_tile):
                ax.set_xticks([]); ax.set_yticks([]); ax.grid(False); ax.spines[:].set_visible(False)
            if lvl not in downsample:
                for ax in (ax_field, ax_tile):
                    ax.text(0.5, 0.5, f"level {lvl}\nnot available\n(slide has {len(pages)} levels)", ha="center", va="center", color=INK_2, transform=ax.transAxes)
                continue
            d = downsample[lvl]
            ax_field.imshow(fields[lvl], interpolation="nearest")
            side_display = tile_size / max(window / d, 1) * display_px
            ax_field.add_patch(Rectangle(((display_px - side_display) / 2,) * 2, side_display, side_display, fill=False, edgecolor=SERIES[1], lw=2))
            ax_field.set_title(f"level {lvl} · {d:.0f}× · {mpp * d:.2f} µm/px", fontsize=12)
            ax_tile.imshow(tiles[lvl], interpolation="nearest")
            ax_tile.set_title(f"{tile_size} px tile = {tile_size * d * mpp / 1000:.2f} mm", fontsize=11, color=INK_2)
        axes[0, 0].set_ylabel("same field of view", fontsize=12)
        axes[1, 0].set_ylabel("tile seen by the model", fontsize=12)

    if show:
        plt.show()

    return {"figure": fig, "downsample": downsample, "window": (x1, y1, window), "mpp": mpp}


def _representative_slides(table: pd.DataFrame) -> pd.DataFrame:
    """The slide of median full-resolution size of each class (deterministic; large, small and odd slides are not picked)."""
    rows = []
    for c in BRACS_CLASSES:
        group = table[table["label"] == c].sort_values(["gigapixels", "slide"])
        rows.append(group.iloc[len(group) // 2])

    return pd.DataFrame(rows)


def plot_class_gallery(table: pd.DataFrame, max_px: int = 700, show: bool = True) -> dict:
    """
    Show one overview image per class: the slide of median size of each class at its coarsest pyramid level.

    * Arguments:
    table: The output of dataset_overview.build_slide_table (with the scan columns)
    max_px: The longest side of every overview image in pixels
    show: Whether to call plt.show() (the figure is returned either way)

    Returns a dict with the figure and the table of the slides shown.
    """

    # STEP 1: Choose one slide per class and read its smallest pyramid level
    chosen = _representative_slides(table)
    images = []
    for _, row in chosen.iterrows():
        path = Path(row["path"])
        pages = WSITileDataset._get_level_pages(path)
        image = _page_image(path, pages[-1])
        images.append(_to_array(image.resize(max_px / max(image.width, image.height))))

    # STEP 2: Figure
    notes = [(0, 4, "Overview of one slide per class, chosen as the slide of median full-resolution size within the class (so that the choice is deterministic and not "
                    "picked for appearance), shown at the coarsest pyramid level of the slide. Titles give the class, the slide, the size at full resolution and the official set.\n"
                    "The images show the variation that the model has to cope with before any pathology is considered: the amount of tissue, the number of tissue pieces, staining "
                    "intensity, and artefacts such as pen marks, folds or air bubbles. Single slides are illustrations, not evidence of class differences.")]
    with plt.rc_context(STYLE):
        fig = plt.figure(figsize=(15, 9.5), layout="constrained")
        fig.suptitle("One slide per class (median size)", fontsize=16, weight="bold", x=0.01, ha="left")
        top = split_with_notes(fig, notes, [1, 1, 1, 1], 15, plot_height=7.0, note_height=1.6)
        axes = top.subplots(2, 4).ravel()
        for ax, (_, row), image in zip(axes, chosen.iterrows(), images):
            ax.imshow(image); ax.set_xticks([]); ax.set_yticks([]); ax.grid(False); ax.spines[:].set_visible(False)
            ax.set_title(f"{row['label']} · {row['slide']}\n{row['width_1'] / 1000:.0f}k × {row['height_1'] / 1000:.0f}k px · {row['gigapixels']:.1f} Gpx · {row['set']}", fontsize=10.5)
        for ax in axes[len(chosen):]:
            ax.axis("off")

    if show:
        plt.show()

    return {"figure": fig, "slides": chosen}


def plot_tile_examples(
    table: pd.DataFrame,
    level: int = 3,
    n_per_class: int = 8,
    tile_size: int = 224,
    seed: int = 0,
    max_tries: int = 80,
    show: bool = True,
) -> dict:
    """
    Show random tissue tiles of every class, each from a different slide, exactly as the model receives them.

    Tile positions are drawn at random and kept when they pass the tissue filter used in the pipeline
    (utils.clean_empty.background_filter).

    * Arguments:
    table: The output of dataset_overview.build_slide_table (with the scan columns)
    level: The pyramid level to read tiles from, counted from 1
    n_per_class: The number of tiles (and slides) per class
    tile_size: The tile edge in pixels
    seed: The random seed
    max_tries: The number of random positions tried per slide before the slide is skipped
    show: Whether to call plt.show() (the figure is returned either way)

    Returns a dict with the figure and a table of the tiles shown (class, slide, x, y).
    """

    # STEP 1: For every class, draw slides at random and one tissue tile from each
    rng = np.random.default_rng(seed)
    grid, records = [], []
    for c in BRACS_CLASSES:
        candidates = table[(table["label"] == c) & (table["n_levels"] >= level)]
        order = rng.permutation(len(candidates))
        row_tiles = []
        for i in order:
            if len(row_tiles) == n_per_class:
                break
            slide = candidates.iloc[i]
            path = Path(slide["path"])
            page = WSITileDataset._get_level_pages(path)[level - 1]
            width, height = int(slide[f"width_{level}"]), int(slide[f"height_{level}"])
            for _ in range(max_tries):
                x, y = int(rng.integers(0, width - tile_size)), int(rng.integers(0, height - tile_size))
                tile = _read_region(path, page, x, y, tile_size)
                if background_filter(tile):
                    row_tiles.append(tile)
                    records.append({"label": c, "slide": slide["slide"], "x": x, "y": y})
                    break
        grid.append(row_tiles)

    # STEP 2: Figure
    notes = [(0, n_per_class, f"Random {tile_size} × {tile_size} px tiles at pyramid level {level}, one row per slide-level class, each tile from a different slide and kept only if it "
                              "passes the tissue filter of the pipeline. These are the inputs of the model, before normalisation.\n"
                              "Tiles of the same slide label can look very different, because the slide label is the highest grade present and most slides also contain benign tissue. "
                              "Differences in staining and sharpness between slides are visible and are one source of the slide effect analysed after feature extraction. "
                              "The examples are random, not selected; they illustrate the variation and do not show diagnostic features of each class.")]
    with plt.rc_context(STYLE):
        fig = plt.figure(figsize=(2.0 * n_per_class, 1.9 * len(BRACS_CLASSES) + 2.2), layout="constrained")
        fig.suptitle(f"Random tissue tiles per class · level {level}", fontsize=16, weight="bold", x=0.01, ha="left")
        top = split_with_notes(fig, notes, [1] * n_per_class, 2.0 * n_per_class, plot_height=1.9 * len(BRACS_CLASSES), note_height=2.2)
        axes = top.subplots(len(BRACS_CLASSES), n_per_class)
        for r, c in enumerate(BRACS_CLASSES):
            for k in range(n_per_class):
                ax = axes[r, k]
                ax.set_xticks([]); ax.set_yticks([]); ax.grid(False); ax.spines[:].set_visible(False)
                if k < len(grid[r]):
                    ax.imshow(grid[r][k])
            axes[r, 0].set_ylabel(c, fontsize=13, rotation=0, labelpad=18, va="center")

    if show:
        plt.show()

    return {"figure": fig, "tiles": pd.DataFrame(records)}
