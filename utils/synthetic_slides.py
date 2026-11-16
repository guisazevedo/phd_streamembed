# IMPORTS *****************************************************************************************
# External libraries
from collections.abc import Mapping
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

# Internal imports ********************************************************************************
from utils.cls_overview import SERIES, STYLE, SURFACE
from utils.figure_captions import split_with_notes


# Helper functions ********************************************************************************
def _quotas(size: int, n_sources: int, cap: int, rng: np.random.Generator) -> np.ndarray:
    """
    Integer tile quotas of the sources of one synthetic slide: every quota is between 1 and cap and they sum to size.

    The shares are drawn from a flat Dirichlet distribution; shares above the cap are set to the cap and the surplus is
    spread over the other sources in proportion to their shares, until none is above the cap.
    """
    if n_sources * cap < size:
        raise ValueError(f"{n_sources} sources of at most {cap} tiles cannot make {size} tiles.")

    # STEP 1: One tile is reserved for every source; the rest is shared out with a cap of cap - 1
    free, room = size - n_sources, cap - 1
    q = rng.dirichlet(np.ones(n_sources)) * free
    capped = np.zeros(n_sources, dtype=bool)

    # STEP 2: Clip to the cap and redistribute the surplus over the other sources
    while True:
        over = (q > room + 1e-9) & ~capped
        if not over.any():
            break
        capped |= over
        q[capped] = room
        open_ = ~capped
        rest = free - room * capped.sum()
        if not open_.any() or rest < 0:
            raise ValueError("The quotas cannot respect the cap.")
        q[open_] = q[open_] / q[open_].sum() * rest

    # STEP 3: Round to integers (largest remainders first) without passing the cap
    base = np.floor(q + 1e-9).astype(int)
    left = free - int(base.sum())
    for i in np.argsort(-(q - base)):
        if left == 0:
            break
        if base[i] < room:
            base[i] += 1
            left -= 1
    while left > 0:
        i = int(np.argmax(np.where(base < room, room - base, -1)))
        base[i] += 1
        left -= 1

    return base + 1


# Main functions **********************************************************************************
def make_recipe(
    name: str,
    label: str,
    pool: list[str],
    n_tiles: dict[str, int],
    rng: np.random.Generator,
    sizes: tuple[int, ...] = (4096, 8192),
    k_slides: int = 12,
    top_percentage: float = 0.20,
    max_slides: int | None = None,
) -> dict:
    """
    The recipe of one synthetic slide: which tiles of which real slides it is made of, in its (random) order.

    The size is drawn from sizes; the number of source slides is drawn uniformly between k_slides and max_slides
    (default: every slide of the pool); the sources are distinct real slides of the pool (all of the same class);
    the tiles of a source are drawn with replacement; no source gives more than top_percentage of the tiles; and the
    tiles of all sources are put in one random order.

    * Arguments:
    name: The name of the synthetic slide
    label: Its class
    pool: The real slides of that class the tiles may come from
    n_tiles: {real slide: number of tiles}
    rng: The random generator
    sizes: The possible numbers of tiles of a synthetic slide
    k_slides: The minimum number of source slides
    top_percentage: The largest share of a synthetic slide that one source slide may give
    max_slides: The maximum number of source slides (None = the pool size)

    Returns a dict with name, label, size, sources (slide names), counts (tiles per source), indices (tile index in
    its source, in slide order) and source_ids (position of the source of every tile in sources).
    """

    # STEP 1: Check that the rules can be met
    if k_slides * top_percentage < 1:
        raise ValueError(f"k_slides={k_slides} sources cannot respect top_percentage={top_percentage} (k_slides * top_percentage must be at least 1).")
    if len(pool) < k_slides:
        raise ValueError(f"The pool of '{label}' has {len(pool)} slides, fewer than k_slides={k_slides}.")

    # STEP 2: Size, number of sources, sources and quotas
    size = int(rng.choice(sizes))
    upper = len(pool) if max_slides is None else min(max_slides, len(pool))
    m = int(rng.integers(k_slides, upper + 1))
    sources = [pool[i] for i in rng.choice(len(pool), m, replace=False)]
    counts = _quotas(size, m, int(np.floor(top_percentage * size)), rng)

    # STEP 3: Tiles drawn with replacement from every source, then one random order for the whole slide
    indices = np.concatenate([rng.integers(0, n_tiles[s], c) for s, c in zip(sources, counts)]).astype(np.int32)
    source_ids = np.repeat(np.arange(m), counts).astype(np.int16)
    order = rng.permutation(size)

    return {"name": name, "label": label, "size": size, "sources": sources, "counts": counts.tolist(), "indices": indices[order], "source_ids": source_ids[order]}


class SyntheticBags(Mapping):
    """
    The bags of synthetic slides, built on access from their recipes (nothing is stored but the recipes).

    bags[name] is a float16 tensor [size, dim] with the tiles of the recipe in its order; the real bags are only read.
    It can replace the bags dict of mil_data.load_bags in batch_iterator, fit_scaler and predict.
    """

    def __init__(self, recipes: list[dict], real_bags: Mapping):
        self.recipes = {r["name"]: r for r in recipes}
        self.real_bags = real_bags

    def __getitem__(self, name: str) -> torch.Tensor:
        recipe = self.recipes[name]
        first = self.real_bags[recipe["sources"][0]]
        out = torch.empty(recipe["size"], first.shape[1], dtype=first.dtype)
        for j, source in enumerate(recipe["sources"]):
            where = np.flatnonzero(recipe["source_ids"] == j)
            out[torch.from_numpy(where)] = self.real_bags[source][torch.from_numpy(recipe["indices"][where].astype(np.int64))]

        return out

    def __iter__(self):
        return iter(self.recipes)

    def __len__(self) -> int:
        return len(self.recipes)


def check_recipes(recipes: list[dict], allowed: set[str], sizes: tuple[int, ...], k_slides: int, top_percentage: float) -> None:
    """Raise ValueError if a recipe breaks a rule: size, number of sources, largest share, sources outside the allowed real slides."""
    for r in recipes:
        counts = np.bincount(r["source_ids"], minlength=len(r["sources"]))
        if len(r["indices"]) != r["size"] or r["size"] not in sizes:
            raise ValueError(f"{r['name']}: wrong size.")
        if len(set(r["sources"])) < k_slides or (counts == 0).any():
            raise ValueError(f"{r['name']}: fewer than {k_slides} source slides.")
        if counts.max() > np.floor(top_percentage * r["size"]):
            raise ValueError(f"{r['name']}: a source gives {counts.max() / r['size']:.1%} of the tiles.")
        if not set(r["sources"]) <= allowed:
            raise ValueError(f"{r['name']}: a source is not in the allowed real slides.")


def recipe_table(folds: list[dict]) -> pd.DataFrame:
    """One row per synthetic slide of the folds from make_synthetic_folds: fold, part (fit or stop), label, size, sources and the largest share of one source."""
    rows = []
    for k, fold in enumerate(folds):
        for part in ("fit", "stop"):
            for r in fold["recipes"][part]:
                rows.append({"fold": k, "part": part, "slide": r["name"], "label": r["label"], "size": r["size"], "n_sources": len(r["sources"]), "max_share": max(r["counts"]) / r["size"]})

    return pd.DataFrame(rows)


def make_synthetic_folds(
    data: dict,
    folds: list[dict],
    n_total: int = 1000,
    sizes: tuple[int, ...] = (4096, 8192),
    k_slides: int = 12,
    top_percentage: float = 0.20,
    stop_fraction: float = 0.2,
    max_slides: int | None = None,
    seed: int = 0,
    cache_dir: str | Path | None = None,
) -> list[dict]:
    """
    Add synthetic training and early-stopping slides to every cross-validation fold.

    For fold k, n_total synthetic slides are made, balanced over the classes: stop_fraction of them for early stopping,
    built from the real early-stopping slides of the fold, and the rest for fitting, built from the real fit slides of the
    fold. The held-out slides of the fold are never a source, so nothing of them reaches training. The real slide
    table of the fold (fold["table"]) and the held-out list (fold["held"]) are unchanged, so the held-out slides are
    scored as real slides; the added keys are train_table (the synthetic slides, set Training or Validation),
    bags (SyntheticBags over the real bags) and recipes ({"fit": [...], "stop": [...]}).

    With cache_dir, the recipes are written to <cache_dir>/synthetic_recipes.pt and read again when the file exists and was made
    with the same settings and the same real fit/early-stopping slides; otherwise they are made again and overwritten.

    * Arguments:
    data: The output of mil_data.load_bags (relabelled for the classes of the model)
    folds: The output of mil_cv.make_cv_folds
    n_total: The number of synthetic slides per fold
    sizes: The possible numbers of tiles of a synthetic slide
    k_slides: The minimum number of source slides of a synthetic slide
    top_percentage: The largest share of a synthetic slide that one source slide may give
    stop_fraction: The share of the synthetic slides used for early stopping
    max_slides: The maximum number of source slides (None = every slide of the class pool)
    seed: The seed (fold k uses seed + k)
    cache_dir: Folder of the recipe file (None = nothing is saved)

    Returns the folds with the added keys (a new list of new dicts).
    """

    # STEP 1: Settings that identify a recipe file, and the real slides of every part of every fold
    classes = list(data["classes"])
    config = {"n_total": n_total, "sizes": list(sizes), "k_slides": k_slides, "top_percentage": top_percentage, "stop_fraction": stop_fraction,
              "max_slides": max_slides, "seed": seed, "classes": classes, "parts": [[sorted(f["fit"]), sorted(f["stop"])] for f in folds]}
    path = Path(cache_dir) / "synthetic_recipes.pt" if cache_dir is not None else None
    saved = None
    if path is not None and path.exists():
        candidate = torch.load(path, weights_only=False)
        if candidate["config"] == config:
            saved = candidate["recipes"]
    n_tiles = {s: len(b) for s, b in data["bags"].items()}

    # STEP 2: The recipes of every fold (read from the file or made now)
    result, all_recipes = [], []
    for k, fold in enumerate(folds):
        if saved is not None:
            recipes = saved[k]
        else:
            rng = np.random.default_rng(seed + k)
            label_of = dict(zip(fold["table"]["slide"], fold["table"]["label"].astype(str)))
            recipes = {"fit": [], "stop": []}
            n_stop = round(n_total * stop_fraction)
            for part, n_part in (("fit", n_total - n_stop), ("stop", n_stop)):
                for c, label in enumerate(classes):
                    pool = [s for s in fold[part] if label_of[s] == label]
                    n_class = n_part // len(classes) + (1 if c < n_part % len(classes) else 0)
                    recipes[part] += [make_recipe(f"SYN_f{k}_{part}_c{c}_{i:04d}", label, pool, n_tiles, rng, sizes, k_slides, top_percentage, max_slides) for i in range(n_class)]
        for part in ("fit", "stop"):
            check_recipes(recipes[part], set(fold[part]), sizes, k_slides, top_percentage)
        all_recipes.append(recipes)

        # STEP 3: The table and the bags of the fold (the real table and the held-out slides stay as they are)
        rows = [{"slide": r["name"], "patient": r["name"], "label": r["label"], "set": "Training" if part == "fit" else "Validation"} for part in ("fit", "stop") for r in recipes[part]]
        result.append({**fold, "train_table": pd.DataFrame(rows), "bags": SyntheticBags(recipes["fit"] + recipes["stop"], data["bags"]), "recipes": recipes})

    # STEP 4: Save the recipes
    if path is not None and saved is None:
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"config": config, "recipes": all_recipes}, path)

    return result


class _DiskBags(Mapping):
    """The bags of saved synthetic slides, read from their .pt files (memory-mapped) when accessed."""

    def __init__(self, paths: dict[str, Path]):
        self.paths = paths

    def __getitem__(self, name: str) -> torch.Tensor:
        return torch.load(self.paths[name], weights_only=True, mmap=True)["cls_features"]

    def __iter__(self):
        return iter(self.paths)

    def __len__(self) -> int:
        return len(self.paths)


def save_synthetic_folds(
    folds: list[dict],
    data: dict,
    out_dir: str | Path,
    model_name: str,
    settings: dict | None = None,
    overwrite: bool = False,
) -> list[Path]:
    """
    Write every synthetic slide of the folds from make_synthetic_folds as a feature file, like those of notebook 01.

    The file <out_dir>/<model_name>/fold_<k>/<slide>.pt holds cls_features (float16 [size, dim]), coordinates (zeros,
    long [size, 2]: a synthetic tile has no position), model_encoder, tile_params (those of the real files plus
    synthetic=True and the settings), label, sources, counts, source_indices (tile index in its source, in slide order) and source_ids
    (position in sources of the source of every tile).

    * Arguments:
    folds: The output of make_synthetic_folds
    data: The output of mil_data.load_bags (for tile_params)
    out_dir: The root folder of the synthetic vectors (for example .../level_1_mod)
    model_name: The encoder name (the sub-folder, as in notebook 01)
    settings: Construction settings stored in tile_params (n_total, sizes, k_slides, top_percentage, seed)
    overwrite: Whether to rewrite files that exist (default: skip them)

    Returns the written (or already existing) paths.
    """

    # STEP 1: One folder per fold
    paths = []
    base_params = {**(data.get("tile_params") or {}), "synthetic": True, **(settings or {})}
    for k, fold in enumerate(folds):
        folder = Path(out_dir) / model_name / f"fold_{k}"
        folder.mkdir(parents=True, exist_ok=True)

        # STEP 2: One file per synthetic slide (the tensor is built from its recipe and then released)
        for part in ("fit", "stop"):
            for r in fold["recipes"][part]:
                path = folder / f"{r['name']}.pt"
                paths.append(path)
                if path.exists() and not overwrite:
                    continue
                tiles = fold["bags"][r["name"]]
                torch.save({"cls_features": tiles, "coordinates": torch.zeros(len(tiles), 2, dtype=torch.long), "model_encoder": model_name,
                            "tile_params": {**base_params, "fold": k, "part": part}, "label": r["label"], "sources": r["sources"], "counts": r["counts"],
                            "source_indices": torch.from_numpy(r["indices"]), "source_ids": torch.from_numpy(r["source_ids"].astype("int64"))}, path)

    return paths


def write_mod_labels(folds: list[dict], path: str | Path) -> pd.DataFrame:
    """
    Write the label file of the synthetic slides (the counterpart of labels/BRACS.xlsx, as CSV) and return it.

    Columns: WSI Filename, Patient Id (the slide name: a synthetic slide is its own group), WSI label, Set (Training = fit,
    Validation = early stopping), fold, part, size, n_sources, max_share, sources (real slides, separated by ';').
    """
    table = recipe_table(folds)
    sources = {r["name"]: ";".join(r["sources"]) for fold in folds for part in ("fit", "stop") for r in fold["recipes"][part]}
    out = pd.DataFrame({"WSI Filename": table["slide"], "Patient Id": table["slide"], "WSI label": table["label"],
                        "Set": table["part"].map({"fit": "Training", "stop": "Validation"}), "fold": table["fold"], "part": table["part"],
                        "size": table["size"], "n_sources": table["n_sources"], "max_share": table["max_share"],
                        "sources": table["slide"].map(sources)})
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(path, index=False)

    return out


def load_mod_folds(real_folds: list[dict], mod_dir: str | Path, labels_csv: str | Path, model_name: str) -> list[dict]:
    """
    Add the saved synthetic slides to the cross-validation folds (the counterpart of make_synthetic_folds, reading from disk).

    The keys train_table and bags are added as in make_synthetic_folds, so mil_cv.fold_data and cross_validate work unchanged.
    Every synthetic slide must have a file, and its sources must be real fit slides (part fit) or early-stopping slides
    (part stop) of its fold; otherwise ValueError (the folds differ from those the vectors were made with).

    * Arguments:
    real_folds: The output of mil_cv.make_cv_folds (same table, seed and settings as in the notebook that wrote the vectors)
    mod_dir: The root folder of the synthetic vectors (as out_dir of save_synthetic_folds)
    labels_csv: The file written by write_mod_labels
    model_name: The encoder name (the sub-folder)

    Returns the folds with the added keys (a new list of new dicts).
    """
    labels = pd.read_csv(labels_csv)
    result = []
    for k, fold in enumerate(real_folds):
        # STEP 1: The slides of this fold, checked against the real fit / early-stopping slides
        rows = labels[labels["fold"] == k]
        if rows.empty:
            raise ValueError(f"{labels_csv} has no synthetic slides for fold {k}.")
        paths = {}
        for _, row in rows.iterrows():
            path = Path(mod_dir) / model_name / f"fold_{k}" / f"{row['WSI Filename']}.pt"
            if not path.exists():
                raise ValueError(f"Missing file {path}.")
            if not set(row["sources"].split(";")) <= set(fold[row["part"]]):
                raise ValueError(f"{row['WSI Filename']}: a source is not a {row['part']} slide of fold {k}; the folds differ from those of the vectors.")
            paths[row["WSI Filename"]] = path

        # STEP 2: The table and the lazy bags
        train_table = pd.DataFrame({"slide": rows["WSI Filename"], "patient": rows["Patient Id"], "label": rows["WSI label"], "set": rows["Set"]}).reset_index(drop=True)
        result.append({**fold, "train_table": train_table, "bags": _DiskBags(paths)})

    return result


def plot_synthetic_summary(table: pd.DataFrame, top_percentage: float, k_slides: int, show: bool = True):
    """
    The synthetic slides of all folds: number of source slides, largest share of one source and size.

    * Arguments:
    table: The output of recipe_table
    top_percentage: The cap on the share of one source slide (drawn as a line)
    k_slides: The minimum number of source slides (drawn as a line)
    show: Whether to call plt.show() (the figure is returned either way)
    """
    notes = [(0, 1, "Number of real slides that contribute to each synthetic slide. The minimum is a constant of the construction; the number above it is drawn uniformly "
                    "up to the number of real slides of the class in the part of the fold, so it differs between classes."),
             (1, 2, "Largest share of a synthetic slide that one real slide gives. The cap is imposed by the construction; the quotas of the sources are drawn from a flat "
                    "Dirichlet distribution and clipped, so many slides reach it.\nThe tiles of a source are drawn with replacement, so a source with fewer tiles than its quota is repeated."),
             (2, 3, "Number of synthetic slides per class and size.\nSynthetic slides built from the same real slides are not independent samples of the class; the "
                    "held-out slides of each fold and the official Testing slides are never a source.")]
    with plt.rc_context(STYLE):
        fig = plt.figure(figsize=(16, 5.8), layout="constrained")
        top = split_with_notes(fig, notes, [1, 1, 1], 16, plot_height=3.6, note_height=2.0)
        a, b, c = top.subplots(1, 3)

        # STEP 1: Number of sources
        a.hist(table["n_sources"], bins=30, color=SERIES[0], edgecolor=SURFACE)
        a.axvline(k_slides, color=SERIES[1], lw=2)
        a.set_xlabel("source slides per synthetic slide"); a.set_ylabel("synthetic slides"); a.set_title("Source slides")

        # STEP 2: Largest share of one source
        b.hist(table["max_share"], bins=30, color=SERIES[0], edgecolor=SURFACE)
        b.axvline(top_percentage, color=SERIES[1], lw=2)
        b.set_xlabel("largest share of one source"); b.set_ylabel("synthetic slides"); b.set_title("Largest share")

        # STEP 3: Slides per class and size
        sizes = sorted(table["size"].unique())
        classes = list(dict.fromkeys(table["label"]))
        width = 0.8 / len(sizes)
        for j, size in enumerate(sizes):
            counts = [int(((table["label"] == cl) & (table["size"] == size)).sum()) for cl in classes]
            c.bar(np.arange(len(classes)) + (j - (len(sizes) - 1) / 2) * width, counts, width, color=SERIES[j % len(SERIES)], label=f"{size} tiles")
        c.set_xticks(range(len(classes)), classes, rotation=15, ha="right"); c.set_ylabel("synthetic slides"); c.set_title("Class and size"); c.legend(frameon=False)

    if show:
        plt.show()

    return fig
