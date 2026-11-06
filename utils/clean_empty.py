# IMPORTS *****************************************************************************************
# External libraries
from collections.abc import Callable
import numpy as np

# Internal imports ********************************************************************************

# Create a type alias for the tile filter function
TileFilter = Callable[[np.ndarray], bool]

# Main function ***********************************************************************************
def background_filter(
    tile: np.ndarray,
    min_tissue_fraction: float = 0.70,
    white_threshold: int = 220,
    black_threshold: int = 15,
) -> bool:
    """
    Return True when a tile contains enough non-background pixels.

    A pixel is considered:
      - white when all RGB channels >= white_threshold
      - black when all RGB channels <= black_threshold

    * Arguments:
    tile: The tile to be filtered
    min_tissue_fraction: The minimum tissue fraction required for the tile to be kept
    white_threshold: The threshold for white pixels from RGB channels
    black_threshold: The threshold for black pixels from RGB channels
    """

    # STEP 1: Extract RGB channels
    rgb = tile[..., :3]

    # STEP 2: Count non-background pixels for each channel
    white_pixels = np.all(rgb >= white_threshold, axis=-1)
    black_pixels = np.all(rgb <= black_threshold, axis=-1)

    # STEP 3: Calculate tissue fraction
    background_pixels = white_pixels | black_pixels
    tissue_fraction = 1.0 - background_pixels.mean()

    # STEP 4: Return True if tissue fraction is sufficient
    return tissue_fraction >= min_tissue_fraction
