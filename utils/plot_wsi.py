# IMPORTS *****************************************************************************************
# External libraries
from math import ceil
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pyvips
import torch
from matplotlib.collections import PatchCollection
from matplotlib.patches import Rectangle

# Internal imports ********************************************************************************
from utils.dataset import WSITileDataset

# Helper functions ********************************************************************************
def _resolve_slide_path(dataset: WSITileDataset, slide) -> Path:
    """
    Resolve which slide to plot among the slides that contributed tiles to the dataset.

    * Arguments:
    dataset: The tile dataset
    slide: An integer position (0 = first slide with tiles), a slide file name or stem, or a Path
    """

    # STEP 1: List the slides that have at least one tile, in dataset order
    slides_with_tiles = list(dict.fromkeys(path for path, _, _ in dataset.tiles))

    if not slides_with_tiles:
        raise ValueError("The dataset has no tiles to plot.")

    # STEP 2: Resolve by position
    if isinstance(slide, int):
        if not -len(slides_with_tiles) <= slide < len(slides_with_tiles):
            raise IndexError(
                f"slide={slide} is out of range, the dataset has {len(slides_with_tiles)} slides with tiles"
            )

        return slides_with_tiles[slide]

    # STEP 3: Resolve by file name or stem
    name = Path(slide).name

    for path in slides_with_tiles:
        if path.name == name or path.stem == name:
            return path

    raise ValueError(f"Slide {slide!r} has no tiles in this dataset")


def _load_overview(dataset: WSITileDataset, slide_path: Path, max_overview_px: int):
    """
    Read the slide at the dataset's pyramid level, downscaled so its longest side is at most
    `max_overview_px`, as an RGB array. Returns (array, scale, level_width, level_height).

    The image is streamed, so a full-resolution level never has to fit in memory.
    """

    page = dataset._level_pages[slide_path][dataset.level - 1]
    image = pyvips.Image.tiffload(str(slide_path), page=page, access="sequential")
    width, height = image.width, image.height

    scale = min(1.0, max_overview_px / max(width, height))

    if scale < 1.0:
        image = image.resize(scale)

    array = WSITileDataset._to_numpy(image)[..., :3].astype(np.uint8, copy=False)

    return array, scale, width, height


def _tile_to_rgb(image) -> np.ndarray:
    """Convert a dataset tile (tensor [C, H, W] or array) to an image matplotlib can draw."""

    if torch.is_tensor(image):
        image = image.permute(1, 2, 0).cpu().numpy()

    if image.dtype.kind == "f":
        image = np.clip(image, 0.0, 1.0)

    return image


# Main function ***********************************************************************************
def plot_wsi(
    dataset: WSITileDataset,
    slide=0,
    n_tiles: int = 12,
    ncols: int = 4,
    max_overview_px: int = 2500,
    show: bool = True,
):
    """
    Plot one slide: the tissue, which tiles passed the background filter, and the first tile crops.

    The figure has three parts:
      1. The full slide at the dataset's pyramid level (downscaled for display only).
      2. The same slide with the tile grid; every tile that passed the filter is outlined in
         blue and the tiles shown below are outlined in orange and numbered.
      3. The first `n_tiles` tile crops of the slide, exactly as the dataset returns them.

    * Arguments:
    dataset: The WSITileDataset to take tiles from
    slide: Which slide to plot: an integer position among the slides with tiles (0 = first),
        a file name or stem, or a Path
    n_tiles: The number of tile crops to show at the bottom
    ncols: The number of columns in the tile crop grid
    max_overview_px: The longest side, in pixels, of the slide overview drawn on screen
    show: Whether to call plt.show() (the figure is returned either way)
    """

    # STEP 1: Resolve the slide and gather its tiles (the first n_tiles are shown as crops)
    slide_path = _resolve_slide_path(dataset, slide)
    tile_indices = [i for i, (path, _, _) in enumerate(dataset.tiles) if path == slide_path]
    slide_tiles = [dataset.tiles[i][1:] for i in tile_indices]
    shown = tile_indices[:n_tiles]
    samples = [dataset[i] for i in shown]

    # STEP 2: Read a display-sized overview of the slide at the tiled level
    overview, scale, level_width, level_height = _load_overview(dataset, slide_path, max_overview_px)

    grid_columns = len(range(0, level_width - dataset.tile_size + 1, dataset.stride))
    grid_rows = len(range(0, level_height - dataset.tile_size + 1, dataset.stride))
    grid_total = grid_columns * grid_rows

    # STEP 3: Lay out the figure: two overview panels, then the tile crops
    crop_rows = max(1, ceil(len(samples) / ncols))
    overview_height = float(np.clip(12 * level_height / level_width, 2.5, 6.0))
    fig = plt.figure(figsize=(12, 2 * overview_height + 2.8 * crop_rows + 0.8), layout="constrained")
    grid_spec = fig.add_gridspec(
        2 + crop_rows,
        ncols,
        height_ratios=[overview_height, overview_height] + [2.8] * crop_rows,
    )

    # STEP 4: Top panel, the full slide for reference
    ax_full = fig.add_subplot(grid_spec[0, :])
    ax_full.imshow(overview)
    ax_full.set_title(f"{slide_path.name} | level {dataset.level} | {level_width} x {level_height} px", fontsize=11)
    ax_full.axis("off")

    # STEP 5: Middle panel, the tile grid with the tiles that passed the filter
    ax_grid = fig.add_subplot(grid_spec[1, :])
    ax_grid.imshow(overview)
    ax_grid.set_title(
        f"Tile grid ({dataset.tile_size} px, stride {dataset.stride}) | "
        f"{len(slide_tiles)} of {grid_total} tiles passed the filter "
        f"({100 * len(slide_tiles) / max(grid_total, 1):.0f}%) | blue: passed, orange: shown below",
        fontsize=10,
    )

    # Grid lines are only drawn when they are far enough apart to be readable
    step = dataset.stride * scale
    if step >= 4:
        ax_grid.vlines(np.arange(0, overview.shape[1], step), 0, overview.shape[0], color="cyan", linewidth=0.3, alpha=0.5)
        ax_grid.hlines(np.arange(0, overview.shape[0], step), 0, overview.shape[1], color="cyan", linewidth=0.3, alpha=0.5)

    side = dataset.tile_size * scale
    ax_grid.add_collection(
        PatchCollection(
            [Rectangle((x * scale, y * scale), side, side) for x, y in slide_tiles],
            facecolor="none", edgecolor="#2a78d6", linewidth=0.8, alpha=0.7,
        )
    )
    ax_grid.add_collection(
        PatchCollection(
            [Rectangle((dataset.tiles[i][1] * scale, dataset.tiles[i][2] * scale), side, side) for i in shown],
            facecolor="none", edgecolor="#eb6834", linewidth=1.6,
        )
    )

    if len(shown) <= 24:
        for number, i in enumerate(shown):
            ax_grid.text(
                dataset.tiles[i][1] * scale, dataset.tiles[i][2] * scale, f"#{number}",
                color="#eb6834", fontsize=7, va="bottom", ha="left", weight="bold",
            )

    ax_grid.set_xlim(0, overview.shape[1])
    ax_grid.set_ylim(overview.shape[0], 0)
    ax_grid.axis("off")

    # STEP 6: Bottom panels, the tile crops as the model pipeline receives them
    for number, sample in enumerate(samples):
        ax = fig.add_subplot(grid_spec[2 + number // ncols, number % ncols])
        ax.imshow(_tile_to_rgb(sample["image"]))
        ax.set_title(f'#{number} | x={sample["x"]}, y={sample["y"]}', fontsize=9)
        ax.axis("off")

    if show:
        plt.show()

    return fig
