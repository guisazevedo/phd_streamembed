# IMPORTS *****************************************************************************************
# External libraries
from collections.abc import Mapping
from pathlib import Path
from time import time
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score

# Internal imports ********************************************************************************
from utils.final_classifier import relabel_table
from utils.mil_cv import cross_validate
from utils.mil_tuning import sample_setting, score_cv, setting_from_row, setting_kwargs

# Constants ***************************************************************************************
SCORINGS = ("cv", "validation")
_TABLE_COLUMNS = ["slide", "patient", "label", "set"]


# Helper classes **********************************************************************************
class SyntheticBags(Mapping):
    """
    The bags of saved synthetic slides (01_3_pca_to_synthetic_vectors), read from their .pt files on access.

    A bag is a float16 tensor [tiles, dim], like those of mil_data.load_bags. With cache = True a bag is kept in
    memory after its first read, which avoids reading the file again in every epoch (about tiles x dim x 2 bytes
    per bag); without it the file is memory-mapped on every access.

    * Arguments:
    paths: {slide: file}
    cache: Whether to keep the bags that were read in memory
    """

    def __init__(self, paths: dict[str, Path], cache: bool = True):
        self.paths, self.cache, self._memo = paths, cache, {}

    def __getitem__(self, name: str) -> torch.Tensor:
        if name in self._memo:
            return self._memo[name]
        x = torch.load(self.paths[name], map_location="cpu", weights_only=True, mmap=True)["cls_features"].to(torch.float16)
        if self.cache:
            x = x.clone()
            self._memo[name] = x
        return x

    def __iter__(self):
        return iter(self.paths)

    def __len__(self) -> int:
        return len(self.paths)


class ChainBags(Mapping):
    """Two bag mappings with disjoint slide names as one (the first is searched first)."""

    def __init__(self, first: Mapping, second: Mapping):
        self.first, self.second = first, second

    def __getitem__(self, name: str) -> torch.Tensor:
        return self.first[name] if name in self.first else self.second[name]

    def __iter__(self):
        yield from self.first
        yield from self.second

    def __len__(self) -> int:
        return len(self.first) + len(self.second)


# Main functions: data ****************************************************************************
def load_synthetic_sets(labels_file: str | Path, root: str | Path, method: str, dim: int, ks: list[int],
                        mapping: dict[str, str], classes: list[str], cache: bool = True, n_finite_checks: int = 5) -> dict:
    """
    Load the synthetic bags of 01_3_pca_to_synthetic_vectors for several numbers of clusters.

    For each k the rows of the label sheet are selected (method, PCA dimension, k), relabelled with `mapping`
    and given the role Training. Every file must exist and have `dim` columns, and a few files are checked for
    NaN / infinite values. No synthetic slide may carry a real slide identifier.

    * Arguments:
    labels_file: The label sheet (labels/BRACS_synthetic.csv)
    root: The synthetic root folder, with <method>/pca_<dim>/k_<k>/ below it
    method: The generation method (a folder name, "gaussian")
    dim: The PCA dimension
    ks: The numbers of clusters to load
    mapping: {BRACS class: group name}, as relabel_table
    classes: The ordered group names of the model (checked against the mapping)
    cache: Keep the bags in memory after their first read
    n_finite_checks: Files per k whose values are checked for NaN / infinity

    Returns {k: {"table": DataFrame (slide, patient, label, set), "bags": SyntheticBags, "folder": Path}}.
    """

    # STEP 1: The sheet
    sheet = pd.read_csv(labels_file)
    sets = {}
    for k in ks:
        sub = sheet[(sheet["method"] == method) & (sheet["pca_dim"] == dim) & (sheet["k"] == k)]
        if sub.empty:
            raise FileNotFoundError(f"{labels_file} has no synthetic slides for method={method!r}, pca_dim={dim}, k={k}; run notebook 01_3 first.")
        if (sub["Set"] != "Synthetic").any():
            raise ValueError("Every slide of the synthetic sheet must have the set 'Synthetic'.")
        table = pd.DataFrame({"slide": sub["WSI Filename"], "patient": sub["Patient Id"], "label": sub["WSI label"], "set": "Training"}).reset_index(drop=True)
        if table["slide"].str.startswith("BRACS").any():
            raise ValueError("A synthetic slide carries a real slide identifier.")
        table, found = relabel_table(table, mapping)
        if found != classes:
            raise ValueError(f"The mapping gives the classes {found}, not {classes}")

        # STEP 2: The files
        folder = Path(root) / method / f"pca_{dim}" / f"k_{k}"
        paths = {s: folder / f"{s}.pt" for s in table["slide"]}
        missing = [str(p) for p in paths.values() if not p.exists()]
        if missing:
            raise FileNotFoundError(f"{len(missing)} synthetic files are missing in {folder}, for example {missing[0]}")
        for p in paths.values():
            shape = torch.load(p, map_location="cpu", weights_only=True, mmap=True)["cls_features"].shape
            if len(shape) != 2 or shape[1] != dim or shape[0] == 0:
                raise ValueError(f"{p}: expected [tiles, {dim}], found {tuple(shape)}")
        for p in list(paths.values())[:: max(1, len(paths) // n_finite_checks)][:n_finite_checks]:
            if not torch.isfinite(torch.load(p, map_location="cpu", weights_only=True, mmap=True)["cls_features"]).all():
                raise ValueError(f"{p}: NaN or infinite values")
        sets[k] = {"table": table, "bags": SyntheticBags(paths, cache), "folder": folder}

    return sets


def synthetic_tile_counts(synthetic_set: dict) -> pd.Series:
    """Tiles per synthetic slide of one value of load_synthetic_sets (only the file headers are read)."""
    return pd.Series({s: int(torch.load(p, map_location="cpu", weights_only=True, mmap=True)["cls_features"].shape[0]) for s, p in synthetic_set["bags"].paths.items()})


def training_folds(folds: list[dict], synthetic_set: dict, real_bags: Mapping) -> list[dict]:
    """
    Folds whose model is fitted on synthetic bags and stopped on the real early-stopping slides of the fold.

    mil_cv.fold_data uses the optional keys train_table and bags: the table holds the synthetic slides (role
    Training) and the real early-stopping slides of the fold (role Validation); the bags are the two sources
    chained. The held-out slides (fold["table"], fold["held"]) stay real and are never in train_table, nor are
    the Testing slides, so cross_validate predicts them from the real bags.

    * Arguments:
    folds: The output of mil_cv.make_cv_folds (relabelled to the model classes)
    synthetic_set: One value of load_synthetic_sets
    real_bags: The real bags (data["bags"])

    Returns the folds with train_table and bags added.
    """
    bags = ChainBags(synthetic_set["bags"], real_bags)
    result = []
    for fold in folds:
        stop = fold["table"][fold["table"]["set"] == "Validation"][_TABLE_COLUMNS]
        train_table = pd.concat([synthetic_set["table"][_TABLE_COLUMNS], stop], ignore_index=True)
        assert set(stop["slide"]) == set(fold["stop"]), "The early-stopping slides of the fold changed."
        result.append({**fold, "train_table": train_table, "bags": bags})

    return result


def validation_mask(table: pd.DataFrame, slides: list[str]) -> np.ndarray:
    """
    The slides that no synthetic generator has seen: official Validation slides of patients without Training slides.

    The generator of 01_3 uses all official Training slides, so only these slides give a clean out-of-fold score
    (the rule of pca_analysis.split_fit_eval).

    * Arguments:
    table: The slide table with the official split (columns slide, patient, set)
    slides: The slides of the pool, in the order of the out-of-fold arrays

    Returns a boolean array, one value per slide.
    """
    train_patients = set(table.loc[table["set"].astype(str) == "Training", "patient"])
    official = table.set_index("slide")
    return np.array([official.loc[s, "set"] == "Validation" and official.loc[s, "patient"] not in train_patients for s in slides])


# Main functions: scoring *************************************************************************
def subset_cv(cv: dict, classes: list[str], mask: np.ndarray) -> dict:
    """The out-of-fold result of cross_validate restricted to the slides of mask, with the per-fold macro-F1 recomputed."""
    fold_of, y, proba = cv["fold_of"][mask], cv["y"][mask], cv["proba"][mask]
    rows = [{"repeat": 0, "fold": int(f), "macro_f1": f1_score(y[fold_of == f], proba[fold_of == f].argmax(1), labels=range(len(classes)), average="macro", zero_division=0)}
            for f in np.unique(fold_of)]

    return {**cv, "slides": [s for s, m in zip(cv["slides"], mask) if m], "y": y, "proba": proba, "patients": cv["patients"][mask], "fold_of": fold_of,
            "per_fold": pd.DataFrame(rows)}


def score_oof(cv: dict, classes: list[str], scoring: str = "cv", mask: np.ndarray | None = None) -> dict:
    """
    Out-of-fold measures of a cross-validation (mil_tuning.score_cv): all pool slides ("cv") or only those of mask ("validation").

    * Arguments:
    cv: The output of cross_validate
    classes: The class names
    scoring: "cv" (every held-out slide) or "validation" (only the slides of mask)
    mask: The output of validation_mask (required for "validation")
    """
    if scoring not in SCORINGS:
        raise ValueError(f"scoring must be one of {SCORINGS}, got {scoring!r}")
    if scoring == "cv":
        return score_cv(cv, classes)
    if mask is None or not mask.any():
        raise ValueError("scoring='validation' needs a mask with at least one slide.")

    return score_cv(subset_cv(cv, classes, mask), classes)


# Main functions: cross-validation and search *****************************************************
def synthetic_cross_validate(data: dict, folds: list[dict], synthetic_set: dict, device: torch.device, weighting: str | None, repeats: int, seed: int, **train_kwargs) -> dict:
    """mil_cv.cross_validate of the CLAM-MB with the model fitted on synthetic bags (see training_folds); same return value."""
    return cross_validate("clam_mb", data, training_folds(folds, synthetic_set, data["bags"]), device, weighting, repeats, seed, **train_kwargs)


def setting_with_k(row: pd.Series) -> dict:
    """The setting of a row of trials_table (mil_tuning.setting_from_row) and the number of clusters syn_k of its synthetic bags."""
    return {**setting_from_row(row), "syn_k": int(row["syn_k"])}


def synthetic_objective_factory(data: dict, folds: list[dict], synthetic_sets: dict, device: torch.device, fixed: dict, weighting: str | None, seed: int,
                                space: dict, scoring: str = "cv", mask: np.ndarray | None = None):
    """
    The objective of the optimisation: one trial is one cross-validation of a sampled setting and a sampled number of clusters.

    Like mil_tuning.objective_factory, but the space holds syn_k (the k of the synthetic bags the model is fitted on) and the
    score is score_oof (all held-out slides, or only those of mask). The measures, the mean best epoch and the minutes are
    stored in trial.user_attrs.

    * Arguments:
    data: The relabelled real bags (the output of load_bags with the model classes)
    folds: The output of make_cv_folds (the same folds for every trial)
    synthetic_sets: The output of load_synthetic_sets
    device: The device to run on
    fixed: The train_model arguments that are not searched
    weighting: The loss weighting rule
    seed: The base seed of the cross-validation
    space: The search ranges, including syn_k (a list of the available k)
    scoring: "cv" or "validation"
    mask: The output of validation_mask (for "validation")
    """
    if "syn_k" not in space or not set(space["syn_k"]) <= set(synthetic_sets):
        raise ValueError("space['syn_k'] must list numbers of clusters that are in synthetic_sets.")

    def objective(trial) -> float:
        # STEP 1: Sample the setting and run the cross-validation on the synthetic bags of the sampled k
        setting = sample_setting(trial, space)
        start = time()
        cv = synthetic_cross_validate(data, folds, synthetic_sets[setting["syn_k"]], device, weighting, 1, seed, **setting_kwargs(setting, fixed))

        # STEP 2: Store the measures and return the pooled out-of-fold macro-F1
        scores = score_oof(cv, data["classes"], scoring, mask)
        for key, value in scores.items():
            trial.set_user_attr(key, value)
        trial.set_user_attr("mean_best_epoch", float(np.mean([r["best_epoch"] for r in cv["runs"]])))
        trial.set_user_attr("minutes", (time() - start) / 60)

        return scores["macro_f1"]

    return objective
