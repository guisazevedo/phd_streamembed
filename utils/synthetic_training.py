# IMPORTS *****************************************************************************************
# External libraries
from collections import OrderedDict
from collections.abc import Mapping
import gc
import json
from pathlib import Path
from time import time
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score

# Internal imports ********************************************************************************
from utils.final_classifier import relabel_table
from utils.mil_cv import cross_validate
from utils.mil_data import fit_scaler
from utils.mil_tuning import sample_setting, score_cv, setting_from_row, setting_kwargs
from utils.synthetic_gaussian import build_generator_from_parameters, composition_from_saved, make_synthetic_wsi

# Constants ***************************************************************************************
SCORINGS = ("cv", "validation")
_TABLE_COLUMNS = ["slide", "patient", "label", "set"]


# Helper classes **********************************************************************************
class ByteLRU:
    """
    A least-recently-used cache of tensors whose total size is capped in bytes.

    One cache can be shared by several bag mappings, so that the cap holds for all of them together (the memory of
    the host or of a device is one budget, not one per mapping). The newest entry is always kept.

    * Arguments:
    max_gb: The cap in GB
    """

    def __init__(self, max_gb: float):
        self.max_bytes, self.bytes, self._items = int(max_gb * 1e9), 0, OrderedDict()

    def get(self, key) -> torch.Tensor | None:
        if key not in self._items:
            return None
        self._items.move_to_end(key)
        return self._items[key]

    def put(self, key, x: torch.Tensor) -> None:
        self._items[key] = x
        self.bytes += x.numel() * x.element_size()
        while self.bytes > self.max_bytes and len(self._items) > 1:
            _, old = self._items.popitem(last=False)
            self.bytes -= old.numel() * old.element_size()


class SyntheticBags(Mapping):
    """
    The bags of saved synthetic slides (01_3_pca_to_synthetic_vectors), read from their .pt files on access.

    A bag is a float32 tensor [tiles, dim], like those of mil_data.load_bags. With cache = True a bag is kept in
    memory after its first read, which avoids reading the file again in every epoch (about tiles x dim x 4 bytes
    per bag). The cache is a least-recently-used cache (ByteLRU) capped at `max_cache_gb`, so it cannot grow with the
    number of bags read; without the cache the file is memory-mapped on every access. With a `device` (for example
    the GPU) the cached bags live on that device, so a batch needs no host-to-device copy of its bags. Several sets
    can share one `store`, which then caps them together.

    * Arguments:
    paths: {slide: file}
    cache: Whether to keep the bags that were read in memory
    max_cache_gb: The size of the cache in GB, on the device when there is one (used when no `store` is given)
    device: The device that holds the cached bags (None = host memory)
    store: A ByteLRU shared with other sets (None = a cache of this set alone)
    """

    def __init__(self, paths: dict[str, Path], cache: bool = True, max_cache_gb: float = 8.0, device: torch.device | None = None,
                 store: ByteLRU | None = None):
        self.paths, self.cache, self.device = paths, cache, device
        self.store = store if store is not None else ByteLRU(max_cache_gb)

    def __getitem__(self, name: str) -> torch.Tensor:
        key = (id(self), name)
        x = self.store.get(key)
        if x is not None:
            return x
        x = torch.load(self.paths[name], map_location="cpu", weights_only=True, mmap=True)["cls_features"]
        if self.cache:
            x = x.clone() if self.device is None else x.to(self.device)  # x points into the memory map, so copy it
            self.store.put(key, x)
        return x

    def __iter__(self):
        return iter(self.paths)

    def __len__(self) -> int:
        return len(self.paths)


class FreshSyntheticBags(Mapping):
    """
    Synthetic bags drawn anew on every access (on-the-fly version of SyntheticBags).

    The slide names and labels are fixed (those of the label sheet of 01_3), but `bags[name]` draws a new bag with
    make_synthetic_wsi from the fitted generator and the templates of the class; nothing is cached. The random
    generator of a draw is seeded by (seed, epoch, index of the name), so the same epoch and seed give the same
    bag and another epoch gives another one. `set_epoch` is called by mil_train.train_model at the start of every
    epoch. Epoch 0 is reserved for the fixed draw of the scaler: the feature scaler is fitted once on at most
    `scaler_bags` evenly spaced bags of that draw (`fit_scaler`, cached), not on every bag of every epoch, so it
    does not change between epochs and the cost does not grow with the number of bags. Bags are float32 [tiles, dim].

    * Arguments:
    names: The slide names, in the order of the label sheet
    classes: {slide name: BRACS class}, which selects the generator and templates of the bag
    generators: {BRACS class: fitted generator}
    compositions: {BRACS class: composition table}
    settings: max_tiles, n_templates and template_concentration of the configuration (metadata.json of 01_3)
    seed: The base seed
    scaler_bags: The number of bags of the fixed scaler draw
    """

    def __init__(self, names: list[str], classes: dict[str, str], generators: dict, compositions: dict, settings: dict, seed: int = 0,
                 scaler_bags: int = 100):
        self.names, self.classes, self.generators, self.compositions = list(names), classes, generators, compositions
        self.settings, self.seed, self.scaler_bags = settings, int(seed), int(scaler_bags)
        self._index, self.epoch, self._scalers = {n: i for i, n in enumerate(self.names)}, 0, {}

    def set_epoch(self, epoch: int) -> None:
        """Select the epoch whose draw the next accesses return (0 = the fixed draw of the scaler)."""
        self.epoch = int(epoch)

    def __getitem__(self, name: str) -> torch.Tensor:
        label = self.classes[name]
        rng = np.random.default_rng([self.seed, self.epoch, self._index[name]])
        wsi = make_synthetic_wsi(self.generators[label], self.compositions[label], rng, self.settings.get("max_tiles"),
                                 self.settings.get("n_templates", 1), self.settings.get("template_concentration", 1.0))
        return torch.from_numpy(wsi["X"]).to(torch.float32)  # the generators may sample in float64

    def fit_scaler(self, slides: list[str]) -> dict[str, torch.Tensor]:
        """The scaler (mil_data.fit_scaler) of the fixed epoch-0 draw of at most `scaler_bags` evenly spaced slides of `slides`; computed once per list."""
        key = tuple(slides)
        if key not in self._scalers:
            chosen = [slides[i] for i in np.unique(np.linspace(0, len(slides) - 1, min(self.scaler_bags, len(slides))).round().astype(int))]
            current, self.epoch = self.epoch, 0
            try:
                self._scalers[key] = fit_scaler(self, chosen)
            finally:
                self.epoch = current
        return self._scalers[key]

    def __iter__(self):
        return iter(self.names)

    def __len__(self) -> int:
        return len(self.names)


class ChainBags(Mapping):
    """Two bag mappings with disjoint slide names as one (the first is searched first). set_epoch and fit_scaler are those of the first mapping, when it has them."""

    def __init__(self, first: Mapping, second: Mapping):
        self.first, self.second = first, second

    def __getattr__(self, name: str):
        if name in ("set_epoch", "fit_scaler") and hasattr(self.__dict__.get("first"), name):
            return getattr(self.first, name)
        raise AttributeError(name)

    def __getitem__(self, name: str) -> torch.Tensor:
        return self.first[name] if name in self.first else self.second[name]

    def __iter__(self):
        yield from self.first
        yield from self.second

    def __len__(self) -> int:
        return len(self.first) + len(self.second)


class LazyFoldData(Mapping):
    """
    The real data (a mil_data.load_bags dict) of one per-fold PCA space, loaded on first access.

    With per-fold PCA spaces (02_4, PER_FOLD) every fold has its own real bags. To bound the memory, the folds that share one
    `slot` keep at most one loaded space: opening another fold drops the previous one. `dim` is known without loading.

    * Arguments:
    loader: A function with no argument that returns the load_bags dict of this fold (it may move the bags to a device)
    dim: The dimension of the vectors of this fold
    slot: A dict shared by the folds that take turns (None = this fold alone); it holds the loaded fold
    key: The identity of this fold in the slot
    """

    def __init__(self, loader, dim: int, slot: dict | None = None, key=None):
        self.loader, self.dim, self.slot, self.key = loader, int(dim), slot if slot is not None else {}, key if key is not None else id(self)

    def _data(self) -> dict:
        if self.slot.get("key") != self.key:
            self.slot.clear()
            gc.collect()
            self.slot.update({"key": self.key, "data": self.loader()})
        return self.slot["data"]

    def __getitem__(self, name: str):
        if name == "dim":
            return self.dim
        if name == "bags":
            return _LazyBags(self)
        return self._data()[name]

    def __iter__(self):
        return iter(self._data())

    def __len__(self) -> int:
        return len(self._data())


class _LazyBags(Mapping):
    """The bags of a LazyFoldData, loaded together with it."""

    def __init__(self, owner: LazyFoldData):
        self.owner = owner

    def __getitem__(self, name: str) -> torch.Tensor:
        return self.owner._data()["bags"][name]

    def __iter__(self):
        return iter(self.owner._data()["bags"])

    def __len__(self) -> int:
        return len(self.owner._data()["bags"])


# Main functions: data ****************************************************************************
def load_synthetic_sets(labels_file: str | Path, root: str | Path, method: str, dim: int, ks: list[int],
                        mapping: dict[str, str], classes: list[str], cache: bool = True, n_finite_checks: int = 5, fresh: bool = False,
                        seed: int = 0, scaler_bags: int = 100, max_cache_gb: float = 8.0, device: torch.device | None = None) -> dict:
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
    cache: Keep the bags in memory after their first read (one least-recently-used cache of at most `max_cache_gb` for all k together)
    n_finite_checks: Files per k whose values are checked for NaN / infinity
    fresh: Draw new bags on every access (FreshSyntheticBags) from the generators, templates and settings stored in the
        folder (cluster_models.pt, metadata.json) instead of reading the saved files; the slide names, labels and table
        stay those of the sheet and the per-file checks are skipped. Needs the `composition` key of cluster_models.pt
        (01_3 run again). `cache` is ignored.
    seed: The base seed of the fresh draws
    scaler_bags: The number of bags of the fixed draw on which a fresh set fits its scaler
    max_cache_gb: The size of the cache in GB, shared by all k (ignored without `cache` and for a fresh set)
    device: Keep the cached bags on this device (for example the GPU; None = host memory); needs `cache`, and max_cache_gb is then in device memory

    Returns {k: {"table": DataFrame (slide, patient, label, set), "bags": SyntheticBags or FreshSyntheticBags, "folder": Path,
    "fresh": bool, "tiles": Series (tiles per slide of the saved set)}}.
    """

    # STEP 1: The sheet
    sheet = pd.read_csv(labels_file)
    sets, store = {}, ByteLRU(max_cache_gb)
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

        # STEP 2: The files (a fresh set reads the generators instead)
        folder = Path(root) / method / f"pca_{dim}" / f"k_{k}"
        tiles = pd.Series(sub["n_tiles"].to_numpy(), index=table["slide"]) if "n_tiles" in sub else None
        if fresh:
            sets[k] = {"table": table, "bags": _fresh_bags(folder, sub, method, dim, seed, scaler_bags), "folder": folder, "fresh": True, "tiles": tiles}
            continue
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
        sets[k] = {"table": table, "bags": SyntheticBags(paths, cache, max_cache_gb, device, store), "folder": folder, "fresh": False, "tiles": tiles}

    return sets


def load_synthetic_sets_per_fold(labels_files: list[str | Path], roots: list[str | Path], method: str, dim: int, ks: list[int],
                                 mapping: dict[str, str], classes: list[str], **kwargs) -> dict:
    """
    load_synthetic_sets for every cross-validation fold, each with its own label sheet and synthetic root (01_3 with PER_FOLD).

    * Arguments:
    labels_files: One label sheet per fold (labels/BRACS_synthetic_cv_fold_<k>.csv)
    roots: One synthetic root per fold (<ROOT>/cv_fold_<k>/synthetic)
    method, dim, ks, mapping, classes, kwargs: As load_synthetic_sets

    Returns {k: [value of load_synthetic_sets for fold 0, fold 1, ...]}; training_folds and the objective accept this form.
    """
    if len(labels_files) != len(roots):
        raise ValueError("One label sheet and one root are needed per fold.")
    per_fold = [load_synthetic_sets(labels, root, method, dim, ks, mapping, classes, **kwargs) for labels, root in zip(labels_files, roots)]

    return {k: [sets[k] for sets in per_fold] for k in ks}


def _fresh_bags(folder: Path, sheet: pd.DataFrame, method: str, dim: int, seed: int, scaler_bags: int) -> FreshSyntheticBags:
    """The FreshSyntheticBags of one configuration folder of 01_3 (generators, templates and settings of that run)."""
    if not (folder / "cluster_models.pt").exists() or not (folder / "metadata.json").exists():
        raise FileNotFoundError(f"{folder} needs cluster_models.pt and metadata.json for fresh bags; run notebook 01_3 first.")
    meta = json.loads((folder / "metadata.json").read_text())
    if meta.get("composition_mode", "blend") != "blend":
        raise NotImplementedError(f"{folder}: fresh bags are not available for composition_mode={meta['composition_mode']!r}; use the saved bags (FRESH_BAGS = False).")
    saved = torch.load(folder / "cluster_models.pt", map_location="cpu", weights_only=True)
    generators = {c: build_generator_from_parameters(meta.get("generator_method", method), e["generator"]) for c, e in saved.items()}
    compositions = {c: composition_from_saved(e) for c, e in saved.items()}
    if meta.get("dim", dim) != dim or meta.get("method", method) != method:
        raise ValueError(f"{folder}: metadata.json is for method={meta.get('method')!r}, dim={meta.get('dim')}")
    unknown = set(sheet["WSI label"]) - set(generators)
    if unknown:
        raise ValueError(f"{folder}: no generator for the classes {sorted(unknown)}")

    return FreshSyntheticBags(sheet["WSI Filename"].tolist(), dict(zip(sheet["WSI Filename"], sheet["WSI label"])), generators, compositions, meta, seed, scaler_bags)


def synthetic_tile_counts(synthetic_set: dict) -> pd.Series:
    """Tiles per synthetic slide of one value of load_synthetic_sets (only the file headers are read; for a fresh set, the sizes of the saved set in the sheet)."""
    if synthetic_set.get("fresh"):
        return synthetic_set["tiles"]
    return pd.Series({s: int(torch.load(p, map_location="cpu", weights_only=True, mmap=True)["cls_features"].shape[0]) for s, p in synthetic_set["bags"].paths.items()})


def training_folds(folds: list[dict], synthetic_set: dict | list[dict], real_bags: Mapping | list, fresh: bool = False) -> list[dict]:
    """
    Folds whose model is fitted on synthetic bags and stopped on the real early-stopping slides of the fold.

    mil_cv.fold_data uses the optional keys train_table and bags: the table holds the synthetic slides (role
    Training) and the real early-stopping slides of the fold (role Validation); the bags are the two sources
    chained. The held-out slides (fold["table"], fold["held"]) stay real and are never in train_table, nor are
    the Testing slides, so cross_validate predicts them from the real bags.

    With per-fold PCA spaces (PER_FOLD) `synthetic_set` is a list with one value per fold and `real_bags` a list with the
    real data dict of every fold (load_bags, or LazyFoldData). Each fold is then fitted, stopped and scored in its own
    space: its bags are the chain of the fold's synthetic set and the fold's real bags, `dim` is the fold's, and
    `eval_data` (the fold's real data) predicts the held-out slides and, through the run, the Testing slides.

    * Arguments:
    folds: The output of mil_cv.make_cv_folds (relabelled to the model classes)
    synthetic_set: One value of load_synthetic_sets, or a list of them (one per fold)
    real_bags: The real bags (data["bags"]), or a list of the real data dicts (one per fold)
    fresh: Whether the set holds fresh bags (load_synthetic_sets(fresh=True)); a set of the other kind raises an error

    Returns the folds with train_table and bags added (per-fold form: also dim and eval_data).
    """
    _check_fresh(synthetic_set, fresh)
    per_fold = isinstance(synthetic_set, list)
    if per_fold and (not isinstance(real_bags, list) or len(real_bags) != len(folds) or len(synthetic_set) != len(folds)):
        raise ValueError("Per-fold synthetic sets need one synthetic set and one real data dict per fold.")
    bags = None if per_fold else ChainBags(synthetic_set["bags"], real_bags)
    result = []
    for k, fold in enumerate(folds):
        stop = fold["table"][fold["table"]["set"] == "Validation"][_TABLE_COLUMNS]
        assert set(stop["slide"]) == set(fold["stop"]), "The early-stopping slides of the fold changed."
        if per_fold:
            train_table = pd.concat([synthetic_set[k]["table"][_TABLE_COLUMNS], stop], ignore_index=True)
            result.append({**fold, "train_table": train_table, "bags": ChainBags(synthetic_set[k]["bags"], real_bags[k]["bags"]),
                           "dim": real_bags[k]["dim"], "eval_data": real_bags[k]})
        else:
            train_table = pd.concat([synthetic_set["table"][_TABLE_COLUMNS], stop], ignore_index=True)
            result.append({**fold, "train_table": train_table, "bags": bags})

    return result


def _check_fresh(synthetic_set: dict | list[dict], fresh: bool) -> None:
    """Stop when the flag and the kind of the synthetic set (or of every set of a per-fold list) disagree (a wrong flag would silently train on the other kind)."""
    for one in (synthetic_set if isinstance(synthetic_set, list) else [synthetic_set]):
        if bool(one.get("fresh", False)) != bool(fresh):
            raise ValueError(f"fresh={fresh} does not match the synthetic set (fresh={one.get('fresh', False)}); load it with the same flag.")


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
def synthetic_cross_validate(data: dict, folds: list[dict], synthetic_set: dict | list[dict], device: torch.device, weighting: str | None, repeats: int, seed: int,
                             fresh: bool = False, fold_data: list | None = None, **train_kwargs) -> dict:
    """
    mil_cv.cross_validate of the CLAM-MB with the model fitted on synthetic bags (see training_folds, which checks `fresh`); same return value.

    With per-fold PCA spaces, `synthetic_set` is the list of the fold's sets and `fold_data` the list of the fold's real data dicts
    (the held-out slides are then predicted in the space of their own fold, and each run carries its space as run["data"]).
    """
    return cross_validate("clam_mb", data, training_folds(folds, synthetic_set, data["bags"] if fold_data is None else fold_data, fresh), device, weighting, repeats, seed, **train_kwargs)


def constants_from_02_3(directory: str | Path, trials_file: str | Path, fallback: dict, names: tuple[str, ...] = ("lr", "weight_decay", "coarse_weight")) -> tuple[dict, str]:
    """
    The values that 02_4 keeps fixed, taken from the tuning of notebook 02_3.

    The order of preference is: the final_setting stored in the config of the latest ensemble file in `directory`
    (the setting that 02_3 selected), then the best row of its trials table `trials_file` (highest macro-F1; only
    the parameters that were searched are in it), then `fallback`. A name missing in a source is taken from the next one.

    * Arguments:
    directory: The folder of the 02_3 ensemble (ensemble_<date-time>.pt files)
    trials_file: The trials table written by 02_3 (CSV)
    fallback: The values used when neither file exists (every name must be in it)
    names: The parameters to read

    Returns the values {name: float} and a text that says where each came from.
    """
    values, source = {}, {}
    files = sorted(Path(directory).glob("ensemble_*.pt")) if Path(directory).exists() else []
    if files:
        chosen = torch.load(files[-1], map_location="cpu", weights_only=True)["config"].get("final_setting", {})
        for name in names:
            if name in chosen:
                values[name], source[name] = float(chosen[name]), f"ensemble {files[-1].name}"
    if Path(trials_file).exists():
        best = pd.read_csv(trials_file).sort_values("macro_f1", ascending=False).iloc[0]
        for name in names:
            if name not in values and name in best.index and pd.notna(best[name]):
                values[name], source[name] = float(best[name]), f"best trial of {Path(trials_file).name}"
    for name in names:
        if name not in values:
            values[name], source[name] = float(fallback[name]), "BASELINE fallback"

    return values, "; ".join(f"{name} = {values[name]:g} ({source[name]})" for name in names)


def add_constant_columns(trials: pd.DataFrame, constants: dict) -> pd.DataFrame:
    """
    The trials table with the parameters that are not searched added as columns (with their constant value).

    trials_table lists the searched parameters only, while select_setting and setting_with_k read lr, weight_decay,
    coarse_weight and k_sample of every row; a column that already exists is kept.
    """
    return trials.assign(**{name: value for name, value in constants.items() if name not in trials.columns})


def setting_with_k(row: pd.Series) -> dict:
    """The setting of a row of trials_table (mil_tuning.setting_from_row) and the number of clusters syn_k of its synthetic bags."""
    return {**setting_from_row(row), "syn_k": int(row["syn_k"])}


def synthetic_objective_factory(data: dict, folds: list[dict], synthetic_sets: dict, device: torch.device, fixed: dict, weighting: str | None, seed: int,
                                space: dict, scoring: str = "cv", mask: np.ndarray | None = None, fresh: bool = False, fold_data: list | None = None):
    """
    The objective of the optimisation: one trial is one cross-validation of a sampled setting and a sampled number of clusters.

    Like mil_tuning.objective_factory, but the space holds syn_k (the k of the synthetic bags the model is fitted on) and the
    score is score_oof (all held-out slides, or only those of mask). The measures, the mean best epoch and the minutes are
    stored in trial.user_attrs.

    * Arguments:
    data: The relabelled real bags (the output of load_bags with the model classes)
    folds: The output of make_cv_folds (the same folds for every trial)
    synthetic_sets: The output of load_synthetic_sets (or load_synthetic_sets_per_fold)
    device: The device to run on
    fixed: The train_model arguments that are not searched
    weighting: The loss weighting rule
    seed: The base seed of the cross-validation
    space: The search ranges, including syn_k (a list of the available k)
    scoring: "cv" or "validation"
    mask: The output of validation_mask (for "validation")
    fresh: Whether the synthetic sets hold fresh bags (checked for every set)
    fold_data: The real data dict of every fold when each fold has its own PCA space (synthetic_sets is then {k: [set per fold]})
    """
    for synthetic_set in synthetic_sets.values():
        _check_fresh(synthetic_set, fresh)
    if "syn_k" not in space or not set(space["syn_k"]) <= set(synthetic_sets):
        raise ValueError("space['syn_k'] must list numbers of clusters that are in synthetic_sets.")

    def objective(trial) -> float:
        # STEP 0: Return the memory of the previous trial (models, activations) before the next one
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

        # STEP 1: Sample the setting and run the cross-validation on the synthetic bags of the sampled k
        setting = sample_setting(trial, space)
        start = time()
        cv = synthetic_cross_validate(data, folds, synthetic_sets[setting["syn_k"]], device, weighting, 1, seed, fresh, fold_data, **setting_kwargs(setting, fixed))

        # STEP 2: Store the measures and return the pooled out-of-fold macro-F1
        scores = score_oof(cv, data["classes"], scoring, mask)
        for key, value in scores.items():
            trial.set_user_attr(key, value)
        trial.set_user_attr("mean_best_epoch", float(np.mean([r["best_epoch"] for r in cv["runs"]])))
        trial.set_user_attr("minutes", (time() - start) / 60)

        return scores["macro_f1"]

    return objective
