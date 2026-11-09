# IMPORTS *****************************************************************************************
# External libraries
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import balanced_accuracy_score, f1_score, log_loss
from sklearn.model_selection import StratifiedGroupKFold
from tqdm.auto import tqdm

# Internal imports ********************************************************************************
from utils.classification_metrics import TIERS, collapse_labels, collapse_probabilities
from utils.mil_train import predict, train_model, training_class_weights


# Constants ***************************************************************************************
POOL_SETS = ("Training", "Validation")


# Main functions **********************************************************************************
def make_cv_folds(table: pd.DataFrame, n_folds: int = 5, seed: int = 0, early_stop_fraction: float = 0.2) -> list[dict]:
    """
    Patient-grouped, class-stratified cross-validation folds over the Training and Validation slides.

    The official Testing slides are not part of any fold. For every fold the slides are split three ways:
    the held-out fold (scored only after training), the early-stopping slides (about early_stop_fraction of the
    rest, chosen patient-grouped and stratified, used to pick the epoch) and the fit slides (the remainder).
    The fold table is a copy of the pool table whose `set` column is "Training" (fit), "Validation" (early
    stopping) and "Testing" (held-out), so train_model can be used unchanged: it never reads the labels of
    slides marked "Testing". All slides of a patient are in the same part, which also removes the patient
    that has slides in both the official Training and Validation sets.

    * Arguments:
    table: The slide table of mil_data.load_bags (columns slide, patient, label, set)
    n_folds: The number of outer folds
    seed: The seed of the splits
    early_stop_fraction: The approximate share of the non-held-out slides used for early stopping

    Returns a list with one dict per fold: table (the fold table), fit, stop, held (lists of slide names).
    """

    # STEP 1: The pool of slides (never the official Testing slides)
    pool = table[table["set"].isin(POOL_SETS)].reset_index(drop=True)
    if pool.empty:
        raise ValueError("No Training or Validation slides with features.")
    labels, patients = pool["label"].astype(str).to_numpy(), pool["patient"].to_numpy()
    inner_splits = max(2, round(1 / early_stop_fraction))

    # STEP 2: Outer folds, then an inner split of the non-held-out slides for early stopping
    folds = []
    outer = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    for k, (rest_index, held_index) in enumerate(outer.split(pool, labels, patients)):
        rest = pool.iloc[rest_index].reset_index(drop=True)
        inner = StratifiedGroupKFold(n_splits=inner_splits, shuffle=True, random_state=seed + 1 + k)
        fit_index, stop_index = next(iter(inner.split(rest, rest["label"].astype(str), rest["patient"])))
        fold_table = pool.copy()
        fold_table["set"] = "Training"
        fold_table.loc[held_index, "set"] = "Testing"
        fold_table.loc[fold_table["slide"].isin(rest.loc[stop_index, "slide"]), "set"] = "Validation"
        folds.append({
            "table": fold_table, "fit": rest.loc[fit_index, "slide"].tolist(), "stop": rest.loc[stop_index, "slide"].tolist(),
            "held": pool.loc[held_index, "slide"].tolist(),
        })

    # STEP 3: Checks: every pool slide is held out exactly once, and no patient is shared between the parts of a fold
    held_all = [s for f in folds for s in f["held"]]
    if sorted(held_all) != sorted(pool["slide"]):
        raise ValueError("The held-out folds do not cover every pool slide exactly once.")
    patient_of = dict(zip(pool["slide"], pool["patient"]))
    for k, f in enumerate(folds):
        parts = {name: {patient_of[s] for s in f[name]} for name in ("fit", "stop", "held")}
        if parts["fit"] & parts["held"] or parts["stop"] & parts["held"] or parts["fit"] & parts["stop"]:
            raise ValueError(f"A patient is shared between the parts of fold {k}.")

    return folds


def fold_summary(folds: list[dict], classes: list[str]) -> pd.DataFrame:
    """Slides, patients and slides per class of the held-out part of every fold (one row per fold)."""
    rows = []
    for k, f in enumerate(folds):
        held = f["table"][f["table"]["set"] == "Testing"]
        row = {"fold": k, "fit": len(f["fit"]), "early stopping": len(f["stop"]), "held out": len(f["held"]), "held-out patients": held["patient"].nunique()}
        row.update({c: int((held["label"].astype(str) == c).sum()) for c in classes})
        rows.append(row)

    return pd.DataFrame(rows)


def fold_data(data: dict, fold: dict) -> dict:
    """A copy of the mil_data.load_bags dict whose table is the fold table (the bags are shared, not copied)."""
    return {**data, "table": fold["table"]}


def cross_validate(
    model_name: str,
    data: dict,
    folds: list[dict],
    device: torch.device,
    weighting: str | None = None,
    repeats: int = 1,
    seed: int = 0,
    **train_kwargs,
) -> dict:
    """
    Train one model per fold (and per repeat) and predict the held-out slides of every fold.

    The class weights of a fold come from its fit slides only. The held-out slides are predicted after
    training and never used for fitting or early stopping. With several repeats (different seeds, same
    folds) the held-out probabilities of a slide are averaged.

    * Arguments:
    model_name: "linear", "abmil" or "clam_mb"
    data: The output of mil_data.load_bags
    folds: The output of make_cv_folds
    device: The device to run on
    weighting: The loss weighting rule ("none", "sqrt", "inverse", "effective"; None = no weights)
    repeats: The number of times every fold is trained (a different seed each time)
    seed: The base seed; the run of repeat r on fold k uses seed + 1000 * r + k
    train_kwargs: Passed to train_model (epochs, lr, weight_decay, tile_dropout, coarse_weight, dropout, k_sample ...)

    Returns a dict with runs (the train_model results, fold-major within each repeat), slides, y (class index),
    patients, proba (mean held-out probabilities over repeats, one row per pool slide), fold_of (fold index of
    every slide) and per_fold (DataFrame with the held-out macro-F1 and balanced accuracy of every fold and repeat).
    """

    classes = data["classes"]
    index = {c: i for i, c in enumerate(classes)}
    pool = folds[0]["table"]
    slides = pool["slide"].tolist()
    position = {s: i for i, s in enumerate(slides)}
    y = np.array([index[str(l)] for l in pool["label"]])
    proba, runs, rows = np.zeros((len(slides), len(classes))), [], []
    fold_of = np.zeros(len(slides), dtype=int)

    for repeat in range(repeats):
        for k, fold in enumerate(folds):
            fold_dataset = fold_data(data, fold)
            run = train_model(model_name, fold_dataset, training_class_weights(fold_dataset, weighting), device, seed + 1000 * repeat + k, verbose=False, **train_kwargs)
            held = fold["held"]
            p = predict(run["model"], data, held, device, run["scaler"])["proba"]
            rows_held = [position[s] for s in held]
            proba[rows_held] += p / repeats
            fold_of[rows_held] = k
            truth = y[rows_held]
            rows.append({"repeat": repeat, "fold": k, "macro_f1": f1_score(truth, p.argmax(1), labels=range(len(classes)), average="macro", zero_division=0),
                         "balanced_accuracy": balanced_accuracy_score(truth, p.argmax(1))})
            runs.append(run)

    proba /= proba.sum(1, keepdims=True)      # float32 outputs averaged over repeats: make every row sum to exactly 1

    return {"runs": runs, "slides": slides, "y": y, "patients": pool["patient"].to_numpy(), "proba": proba, "fold_of": fold_of, "per_fold": pd.DataFrame(rows)}


def cv_compare(
    model_name: str,
    data: dict,
    folds: list[dict],
    device: torch.device,
    settings: list[dict],
    weighting: str | None = None,
    repeats: int = 1,
    seed: int = 0,
    progress: bool = True,
    **common_kwargs,
) -> pd.DataFrame:
    """
    Compare settings by patient-grouped cross-validation (see cross_validate); the Testing set is never used.

    Each setting is a dict that may hold "weighting" and any train_model argument (dropout, weight_decay,
    tile_dropout, coarse_weight ...); it overrides common_kwargs. The table has one row per setting with
    the out-of-fold measures pooled over all pool slides (macro-F1, balanced accuracy, log-loss and the
    accuracy and macro-F1 of the three clinical tiers), the mean and standard deviation over folds of the
    held-out macro-F1, the mean best epoch and number of epochs run, and the training and early-stopping
    loss at the best epoch with their difference. The training loss is class-weighted and has dropout
    active, whereas the early-stopping loss is unweighted, so the difference is informative between rows,
    not as an absolute value. Rows are sorted by the pooled macro-F1 (ties: lower log-loss); the first row
    is the choice. Because the same folds select and score, the pooled scores of the chosen row are
    slightly optimistic; the Testing set remains the unbiased estimate.

    * Arguments:
    model_name: "linear", "abmil" or "clam_mb"
    data: The output of mil_data.load_bags
    folds: The output of make_cv_folds
    device: The device to run on
    settings: A list of dicts, one per row of the table
    weighting: The default loss weighting (a setting may override it with "weighting")
    repeats: The number of repeats per fold
    seed: The base seed (the same for every setting)
    progress: Whether to show a progress bar over the settings
    common_kwargs: Passed to train_model for every setting
    """

    classes = data["classes"]
    rows = []
    for setting in tqdm(settings, desc=f"CV {model_name}", disable=not progress):
        kwargs = {**common_kwargs, **{k: v for k, v in setting.items() if k != "weighting"}}
        cv = cross_validate(model_name, data, folds, device, setting.get("weighting", weighting), repeats, seed, **kwargs)
        truth, proba = cv["y"], cv["proba"]
        tier_truth = collapse_labels(truth, classes, TIERS)
        tier_proba, tier_names = collapse_probabilities(proba, classes, TIERS)
        at_best = [r["history"].set_index("epoch").loc[r["best_epoch"]] for r in cv["runs"]]
        train_loss, stop_loss = float(np.mean([h["train_loss"] for h in at_best])), float(np.mean([h["val_loss"] for h in at_best]))
        per_fold = cv["per_fold"].groupby("fold")["macro_f1"].mean()
        rows.append({
            "model": model_name, **setting,
            "oof_macro_f1": f1_score(truth, proba.argmax(1), labels=range(len(classes)), average="macro", zero_division=0),
            "oof_balanced_accuracy": balanced_accuracy_score(truth, proba.argmax(1)),
            "oof_log_loss": log_loss(truth, np.clip(proba / proba.sum(1, keepdims=True), 1e-12, 1), labels=range(len(classes))),
            "tier_accuracy": float((tier_proba.argmax(1) == tier_truth).mean()),
            "tier_macro_f1": f1_score(tier_truth, tier_proba.argmax(1), labels=range(len(tier_names)), average="macro", zero_division=0),
            "fold_macro_f1_mean": float(per_fold.mean()), "fold_macro_f1_sd": float(per_fold.std(ddof=1)) if len(per_fold) > 1 else 0.0,
            "best_epoch_mean": float(np.mean([r["best_epoch"] for r in cv["runs"]])), "epochs_run_mean": float(np.mean([r["epochs_run"] for r in cv["runs"]])),
            "train_loss": train_loss, "stop_loss": stop_loss, "loss_gap": stop_loss - train_loss,
        })

    return pd.DataFrame(rows).sort_values(["oof_macro_f1", "oof_log_loss"], ascending=[False, True]).reset_index(drop=True)
