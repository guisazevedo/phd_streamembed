# IMPORTS *****************************************************************************************
# External libraries
from collections.abc import Iterator
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm

# Internal imports ********************************************************************************
from utils.labels import BRACS_CLASSES


# Main functions **********************************************************************************
def get_classes(table: pd.DataFrame) -> list[str]:
    """Return the classes present in the label table (read from BRACS.xlsx), in clinical order."""
    present = set(table["label"].astype(str))
    return [c for c in BRACS_CLASSES if c in present]


def load_bags(features_dir: str | Path, model_name: str, table: pd.DataFrame) -> dict:
    """
    Load the CLS embeddings of every labelled slide as one bag of tile vectors per slide.

    Slides without a feature file, or without any tile (for example because they lack the pyramid
    level that was tiled), are dropped and reported. Vectors are kept in float16 to save memory.

    * Arguments:
    features_dir: The folder that holds one subfolder per model
    model_name: The model subfolder with the per-slide .pt files
    table: The output of dataset_overview.build_slide_table (columns slide, patient, label, set)

    Returns a dict with bags ({slide: float16 tensor [tiles, dim]}), coords ({slide: [tiles, 2]}),
    table (the rows of `table` that have features, reindexed), classes, missing (slide names), dim and
    tile_params (the tiling parameters stored in the files; a list when the files disagree).
    """

    # STEP 1: Find the feature files
    folder = Path(features_dir) / model_name
    files = {p.stem: p for p in folder.glob("*.pt")}
    if not files:
        raise FileNotFoundError(f"No .pt files found in {folder.resolve()}")

    # STEP 2: Load the bag of every labelled slide that has a file
    bags, coords, params, missing, empty = {}, {}, [], [], []
    for slide in tqdm(table["slide"], desc="Loading bags"):
        if slide not in files:
            missing.append(slide)
            continue
        data = torch.load(files[slide], map_location="cpu", weights_only=True, mmap=True)
        x = data["cls_features"]
        if len(x) == 0:
            empty.append(slide)
            continue
        bags[slide] = x.to(torch.float16).clone()
        coords[slide] = data["coordinates"].clone()
        params.append(data.get("tile_params"))

    kept = table[table["slide"].isin(bags)].reset_index(drop=True)
    if kept.empty:
        raise ValueError("None of the labelled slides has a feature file; check FEATURES_DIR, MODEL_NAME and the slide names.")

    # STEP 3: The files must describe the same tiling
    distinct = {repr({k: v for k, v in p.items()}) if p else "none" for p in params}
    tile_params = params[0] if len(distinct) == 1 else params
    if len(distinct) > 1:
        print(f"Warning: the feature files do not share one tiling ({len(distinct)} different tile_params); the results mix them.")

    # STEP 4: Report
    sizes = np.array([len(b) for b in bags.values()])
    print(f"Slides with features: {len(kept)} of {len(table)}  |  tiles {sizes.sum():,}  |  tiles per slide: median {int(np.median(sizes))}, min {sizes.min()}, max {sizes.max()}  |  dim {next(iter(bags.values())).shape[1]}")
    print("Per set (with features / labelled): " + ", ".join(
        f"{s} {int((kept['set'] == s).sum())}/{int((table['set'] == s).sum())}" for s in ("Training", "Validation", "Testing")))
    if missing or empty:
        print(f"Dropped: {len(missing)} without a feature file, {len(empty)} without tiles " + (f"(e.g. {(missing + empty)[:4]})" if missing + empty else ""))
    if isinstance(tile_params, dict):
        print(f"Tiling in the files: level {tile_params['level']}, tile {tile_params['tile_size']}, stride {tile_params['stride']}")

    return {
        "bags": bags, "coords": coords, "table": kept, "classes": get_classes(kept), "missing": missing + empty,
        "dim": next(iter(bags.values())).shape[1], "tile_params": tile_params,
    }


def fit_scaler(bags: dict[str, torch.Tensor], slides: list[str]) -> dict[str, torch.Tensor]:
    """
    Mean and standard deviation of every embedding dimension over all tiles of the given slides.

    Fit it on the Training slides only. Sums are accumulated slide by slide in float64, so the tiles
    never have to be held in memory at once.
    """
    n, total, total_sq = 0, 0.0, 0.0
    for slide in slides:
        x = bags[slide].double()
        n += len(x)
        total = total + x.sum(0)
        total_sq = total_sq + (x ** 2).sum(0)
    mean = total / n
    std = (total_sq / n - mean ** 2).clamp(min=1e-12).sqrt()

    return {"mean": mean.float(), "std": std.float().clamp(min=1e-6)}


def batch_iterator(
    bags: dict[str, torch.Tensor],
    slides: list[str],
    labels: dict[str, int],
    batch_size: int,
    device: torch.device,
    scaler: dict[str, torch.Tensor] | None = None,
    max_tiles: int | None = None,
    shuffle: bool = False,
    generator: torch.Generator | None = None,
    tile_dropout: float = 0.0,
) -> Iterator[tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[str]]]:
    """
    Yield padded batches of bags: (x [B, N, dim] float32, mask [B, N] bool, y [B] long, slide names).

    With tile_dropout > 0 a random fraction of the tiles of every bag is removed first (at least one
    tile is kept); then bags longer than max_tiles are randomly subsampled (a different subset every
    call, which also acts as tile dropout); None keeps every tile. Shorter bags are padded with zeros and
    masked. Use tile_dropout only for training batches; evaluation should keep every tile.

    * Arguments:
    bags: The bags from load_bags
    slides: The slides to iterate over
    labels: {slide: class index}
    batch_size: The number of bags per batch
    device: The device the batches are moved to
    scaler: The output of fit_scaler (None = no standardisation)
    max_tiles: The maximum number of tiles per bag (None = all)
    shuffle: Whether to shuffle the slides each call
    generator: The seeded generator for shuffling and subsampling (use one for reproducibility)
    tile_dropout: The fraction of the tiles of each bag removed at random before max_tiles is applied (0 = none)
    """

    if not 0.0 <= tile_dropout < 1.0:
        raise ValueError(f"tile_dropout must be in [0, 1), got {tile_dropout}")

    order = torch.randperm(len(slides), generator=generator).tolist() if shuffle else list(range(len(slides)))

    for start in range(0, len(order), batch_size):
        names = [slides[i] for i in order[start:start + batch_size]]
        tensors = []
        for name in names:
            x = bags[name]
            if tile_dropout > 0:
                keep = max(1, round(len(x) * (1 - tile_dropout)))
                if keep < len(x):
                    x = x[torch.randperm(len(x), generator=generator)[:keep]]
            if max_tiles is not None and len(x) > max_tiles:
                x = x[torch.randperm(len(x), generator=generator)[:max_tiles]]
            tensors.append(x)

        length = max(len(x) for x in tensors)
        padded = torch.zeros(len(names), length, tensors[0].shape[1], dtype=torch.float16)
        mask = torch.zeros(len(names), length, dtype=torch.bool)
        for i, x in enumerate(tensors):
            padded[i, : len(x)] = x
            mask[i, : len(x)] = True

        x = padded.to(device).float()
        if scaler is not None:
            x = (x - scaler["mean"].to(device)) / scaler["std"].to(device)
        x = x * mask.to(device).unsqueeze(-1)
        y = torch.as_tensor([labels[n] for n in names], device=device)

        yield x, mask.to(device), y, names
