# IMPORTS *****************************************************************************************
# External libraries
from pathlib import Path
import re
import numpy as np
import pandas as pd
import pyvips
from tqdm.auto import tqdm

# Internal imports ********************************************************************************
from utils.dataset import WSITileDataset

# Constants ***************************************************************************************
MAX_LEVEL = WSITileDataset.MAX_LEVEL


# Helper functions ********************************************************************************
def _parse_description(text: str) -> dict:
    """Read the Aperio ImageDescription header: JPEG quality plus the 'key = value' fields after the first '|'."""
    fields = dict(part.split(" = ", 1) for part in text.split("|")[1:] if " = " in part)
    quality = re.search(r"Q=(\d+)", text)

    return {
        "scanner": fields.get("ScanScope ID"),
        "app_mag": float(fields["AppMag"]) if "AppMag" in fields else np.nan,
        "mpp": float(fields["MPP"]) if "MPP" in fields else np.nan,
        "scan_date": pd.to_datetime(fields.get("Date"), format="%m/%d/%y", errors="coerce"),
        "jpeg_quality": int(quality.group(1)) if quality else np.nan,
    }


# Main function ***********************************************************************************
def scan_slides(data_dir: str | Path, slides: list[str] | None = None, show_progress: bool = True) -> pd.DataFrame:
    """
    Read the technical properties of every .svs slide from its headers (pixels are not decoded).

    * Arguments:
    data_dir: The folder with the .svs files
    slides: The slide names (file stems) to scan; None = every .svs file in data_dir
    show_progress: Whether to show a progress bar

    Returns a DataFrame with one row per slide: slide, path, file_gb, n_levels (tiled pyramid levels, as
    used by WSITileDataset), width_L / height_L for L = 1..MAX_LEVEL (NaN when the level does not exist),
    downsample_L (width of level 1 divided by width of level L), gigapixels (level 1), scanner, app_mag,
    mpp (microns per pixel at level 1), scan_date and jpeg_quality.
    """

    # STEP 1: List the slides to scan
    folder = Path(data_dir)
    paths = sorted(folder.glob("*.svs")) if slides is None else [folder / f"{s}.svs" for s in slides]

    # STEP 2: Read the pyramid levels and the header description of every slide
    rows = []
    for path in tqdm(paths, desc="Scanning slides", disable=not show_progress):
        pages = WSITileDataset._get_level_pages(path)
        images = [pyvips.Image.tiffload(str(path), page=p) for p in pages]
        sizes = [(image.width, image.height) for image in images]
        description = pyvips.Image.tiffload(str(path), page=0).get("image-description")

        row = {"slide": path.stem, "path": str(path), "file_gb": path.stat().st_size / 1e9, "n_levels": len(pages)}
        for level in range(1, MAX_LEVEL + 1):
            width, height = sizes[level - 1] if level <= len(sizes) else (np.nan, np.nan)
            row[f"width_{level}"], row[f"height_{level}"] = width, height
            row[f"downsample_{level}"] = sizes[0][0] / width if width == width else np.nan
        row["gigapixels"] = sizes[0][0] * sizes[0][1] / 1e9
        rows.append({**row, **_parse_description(description)})

    return pd.DataFrame(rows)
