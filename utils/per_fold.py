# IMPORTS *****************************************************************************************
# External libraries
import json
from pathlib import Path
import pandas as pd
import torch

# Internal imports ********************************************************************************
from utils.dataset_overview import build_slide_table
from utils.final_classifier import relabel_table
from utils.labels import BRACS_CLASSES
from utils.mil_cv import make_cv_folds

# Constants ***************************************************************************************
HELD_OUT_SET = "HeldOut"        # the `set` stored in the files of a fold for its held-out slides (the official "Testing" slides keep "Testing")
_FOLD_SETS = {"Training": "Training", "Validation": "Validation", "Testing": HELD_OUT_SET}


# Helpers *****************************************************************************************
def fold_root(root: str | Path, fold: int) -> Path:
    """The folder of one cross-validation fold: <root>/cv_fold_<fold>."""
    return Path(root) / f"cv_fold_{fold}"


def fold_pca_dir(root: str | Path, fold: int, dim: int) -> Path:
    """The folder with the PCA files of the fold: <root>/cv_fold_<fold>/pca_<dim> (the per-slide files are in its UNI2-h subfolder)."""
    return fold_root(root, fold) / f"pca_{dim}"


def fold_synthetic_root(root: str | Path, fold: int) -> Path:
    """The synthetic root of the fold: <root>/cv_fold_<fold>/synthetic (with <method>/pca_<dim>/k_<k>/ below it)."""
    return fold_root(root, fold) / "synthetic"


def fold_labels_file(labels_file: str | Path, fold: int) -> Path:
    """The label sheet of the synthetic slides of one fold: <stem>_cv_fold_<fold><suffix> next to the global sheet."""
    path = Path(labels_file)
    return path.with_name(f"{path.stem}_cv_fold_{fold}{path.suffix}")


def _has_tiles(features_dir: Path, model_name: str, slide: str) -> bool:
    """Whether the slide has a feature file with at least one tile (the rule of mil_data.load_bags)."""
    path = Path(features_dir) / model_name / f"{slide}.pt"
    return path.exists() and len(torch.load(path, map_location="cpu", weights_only=True, mmap=True)["cls_features"]) > 0


# Main functions **********************************************************************************
def cv_folds_for_features(features_dir: str | Path, model_name: str, labels_file: str | Path, mapping: dict[str, str],
                          n_folds: int, seed: int, early_stop_fraction: float) -> tuple[list[dict], pd.DataFrame]:
    """
    The cross-validation folds of notebooks 02_3 / 02_4, rebuilt without loading any bag.

    The slide table of BRACS.xlsx is restricted to the slides that have a usable file in `features_dir` (the rule of
    mil_data.load_bags), relabelled to the three classes of `mapping` and split with mil_cv.make_cv_folds, so the folds are
    those of 02_3 / 02_4 when the same n_folds, seed and early_stop_fraction are given.

    * Arguments:
    features_dir: The folder with the original embeddings (level_1)
    model_name: The model subfolder
    labels_file: labels/BRACS.xlsx
    mapping: {BRACS class: group}, the three-class mapping that stratifies the folds
    n_folds: The number of folds
    seed: The seed of the folds (the CV seed of 02_3 / 02_4, not the seed of 01_2)
    early_stop_fraction: The share of the non-held-out slides used for early stopping

    Returns the folds (make_cv_folds, each with the extra key `table7`: the slides of the fold and the official Testing slides with
    the 7-class label and the `set` that is stored in the fold's PCA files: Training, Validation, HeldOut, Testing) and the slide table.
    """

    # STEP 1: The slides with a usable file
    table = build_slide_table(labels_file)
    usable = [s for s in table["slide"] if _has_tiles(Path(features_dir), model_name, s)]
    table = table[table["slide"].isin(usable)].reset_index(drop=True)

    # STEP 2: The folds of the three-class table
    table3, _ = relabel_table(table, mapping)
    folds = make_cv_folds(table3, n_folds, seed, early_stop_fraction)

    # STEP 3: The 7-class table of every fold, with the official Testing slides
    official = table[table["set"].astype(str) == "Testing"]
    label7 = dict(zip(table["slide"], table["label"].astype(str)))
    for fold in folds:
        rows = fold["table"].assign(label=lambda t: t["slide"].map(label7), set=lambda t: t["set"].map(_FOLD_SETS))
        both = pd.concat([rows, official.assign(label=official["label"].astype(str), set="Testing")], ignore_index=True)[["slide", "patient", "label", "set"]]
        both["label"] = pd.Categorical(both["label"], categories=BRACS_CLASSES, ordered=True)      # the clinical class order, as build_slide_table
        fold["table7"] = both

    return folds, table


def fold_definition(fold: dict, index: int) -> dict:
    """The part of a fold that is written to the metadata of its PCA files and checked by the later notebooks."""
    return {"fold": index, "fit_slides": list(fold["fit"]), "stop_slides": list(fold["stop"]), "held_slides": list(fold["held"])}


def check_fold_metadata(folds: list[dict], root: str | Path, dim: int) -> None:
    """
    Stop when the folds of this notebook differ from those that 01_2 used to fit the PCA of every fold.

    * Arguments:
    folds: The folds of cv_folds_for_features
    root: The features root (with cv_fold_<k>/pca_<dim>/metadata.json below it)
    dim: A PCA dimension that was saved
    """
    for k, fold in enumerate(folds):
        path = fold_pca_dir(root, k, dim) / "metadata.json"
        if not path.exists():
            raise FileNotFoundError(f"{path} does not exist; run notebook 01_2 with PER_FOLD = True first.")
        saved = json.loads(path.read_text())
        for key, mine in (("fit_slides", fold["fit"]), ("stop_slides", fold["stop"]), ("held_slides", fold["held"])):
            if sorted(saved.get(key, [])) != sorted(mine):
                raise ValueError(f"Fold {k}: the {key} of {path} differ from the folds built here (check N_FOLDS, CV_SEED, EARLY_STOP_FRACTION and the slides with features).")


def fold_table_from_metadata(root: str | Path, fold: int, dim: int, table: pd.DataFrame) -> pd.DataFrame:
    """
    The slide table of one fold as 01_2 stored it (Training = fit slides), rebuilt from the metadata.json of the fold's PCA folder.

    Notebook 01_3 uses it, so that the generator of a fold is fitted on exactly the slides its PCA was fitted on, without
    rebuilding the folds.

    * Arguments:
    root: The features root (with cv_fold_<k>/pca_<dim>/metadata.json below it)
    fold: The fold index
    dim: A PCA dimension that was saved
    table: The slide table of build_slide_table (7-class labels, official split)

    Returns the table (columns slide, patient, label, set) with the sets Training, Validation, HeldOut and the official Testing slides.
    """
    path = fold_pca_dir(root, fold, dim) / "metadata.json"
    if not path.exists():
        raise FileNotFoundError(f"{path} does not exist; run notebook 01_2 with PER_FOLD = True first.")
    meta = json.loads(path.read_text())
    role = {**{s: "Training" for s in meta["fit_slides"]}, **{s: "Validation" for s in meta["stop_slides"]}, **{s: HELD_OUT_SET for s in meta["held_slides"]}}
    out = table[["slide", "patient", "label", "set"]].copy()
    official_testing = out["set"].astype(str) == "Testing"
    out["set"] = out["slide"].map(role).where(~official_testing, "Testing")
    out = out[out["set"].notna()].reset_index(drop=True)
    out["set"] = out["set"].astype(str)

    return out


def load_fold_data(root: str | Path, n_folds: int, dim: int, model_name: str, table: pd.DataFrame, device=None, lazy: bool = False) -> list:
    """
    The real data (mil_data.load_bags) of every fold's own PCA space, written by 01_2 with PER_FOLD.

    Each fold holds the pool slides and the official Testing slides, transformed by the PCA of that fold. The dict of a fold also carries
    `features_dir` (the folder it was read from), so a saved model records its space.

    * Arguments:
    root: The features root (with cv_fold_<k>/pca_<dim>/UNI2-h/ below it)
    n_folds: The number of folds
    dim: The PCA dimension
    model_name: The model subfolder
    table: The slide table of build_slide_table (7-class labels)
    device: Move the bags to this device (mil_data.bags_to_device); None = keep them on the host
    lazy: Return LazyFoldData objects that read a fold on first use and keep only one fold in memory (slower, for limited memory)

    Returns a list with one dict (or LazyFoldData) per fold; the dimension of every fold is checked against `dim`.
    """
    from utils.mil_data import bags_to_device, load_bags
    from utils.synthetic_training import LazyFoldData

    slot: dict = {}

    def loader_for(k: int):
        def loader() -> dict:
            folder = fold_pca_dir(root, k, dim)
            data = load_bags(folder, model_name, fold_table_from_metadata(root, k, dim, table))
            if data["dim"] != dim:
                raise ValueError(f"the files in {folder} have {data['dim']} components, not {dim}")
            if device is not None:
                data["bags"] = bags_to_device(data["bags"], device)
            data["features_dir"] = str(folder)
            return data
        return loader

    if lazy:
        return [LazyFoldData(loader_for(k), dim, slot, k) for k in range(n_folds)]

    return [loader_for(k)() for k in range(n_folds)]


def attach_fold_data(runs: list[dict], fold_data: list) -> list[dict]:
    """
    Give every run of a loaded ensemble (mil_tuning.load_ensemble) the real data of its own fold, so that predict_ensemble predicts in the model's PCA space.

    * Arguments:
    runs: The runs of load_ensemble (each with the key `fold`)
    fold_data: The output of load_fold_data

    Returns the runs (changed in place).
    """
    for run in runs:
        run["data"] = fold_data[run["fold"]]

    return runs
