# IMPORTS *****************************************************************************************
# External libraries
import json
from datetime import datetime
from pathlib import Path
from time import time
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score, log_loss, recall_score

# Internal imports ********************************************************************************
from utils.cls_overview import INK_2, SERIES, STYLE, SURFACE
from utils.figure_captions import split_with_notes
from utils.mil_cv import cross_validate
from utils.mil_models import build_model
from utils.mil_train import predict

# Constants ***************************************************************************************
# Search ranges of the Bayesian optimisation: (low, high) for the numeric ones, the list of values for the categorical ones
DEFAULT_SPACE = {
    "lr": (5e-5, 1e-3),
    "weight_decay": (1e-3, 3e-1),
    "dropout": (0.1, 0.6),
    "coarse_weight": [0.0, 0.1, 0.3, 0.5],
    "k_sample": [4, 8, 16, 32],
    "max_tiles": [2048, 4096, 8192, None],
}
PARAMETERS = list(DEFAULT_SPACE)


# Helper functions ********************************************************************************
def batch_size_for(max_tiles: int | None, base_batch: int = 16, base_tiles: int = 2048, no_cap_batch: int = 2) -> int:
    """
    The number of slides per batch that keeps the tiles per batch about constant (the batch is padded to its longest bag).

    16 slides at 2048 tiles, 8 at 4096, 4 at 8192; without a cap (bags of up to about 35,000 tiles) no_cap_batch.
    """
    if max_tiles is None:
        return no_cap_batch

    return max(1, base_batch * base_tiles // max_tiles)


def sample_setting(trial, space: dict | None = None) -> dict:
    """Sample one setting from an optuna trial: tuples are (low, high) ranges (log scale for lr and weight_decay), lists are categorical."""
    space = space or DEFAULT_SPACE
    setting = {}
    for name, domain in space.items():
        if isinstance(domain, tuple):
            setting[name] = trial.suggest_float(name, *domain, log=name in ("lr", "weight_decay"))
        else:
            setting[name] = trial.suggest_categorical(name, domain)

    return setting


def setting_from_row(row: pd.Series) -> dict:
    """The setting of a row of trials_table with its types restored (a row of mixed values turns the integers into floats and None into NaN)."""
    setting = {"lr": float(row["lr"]), "weight_decay": float(row["weight_decay"]), "dropout": float(row["dropout"]), "coarse_weight": float(row["coarse_weight"]),
               "k_sample": int(row["k_sample"])}
    if "max_tiles" in row.index:      # absent when the tile cap is not searched (notebook 02_3)
        setting["max_tiles"] = None if pd.isna(row["max_tiles"]) else int(row["max_tiles"])

    return setting


def setting_kwargs(setting: dict, fixed: dict) -> dict:
    """The train_model arguments of a setting: the fixed ones, the sampled ones and the batch size (the one in fixed, if any, else the one that goes with the tile cap)."""
    return {**fixed, **{k: setting[k] for k in PARAMETERS if k in setting}, "batch_size": fixed.get("batch_size") or batch_size_for(setting.get("max_tiles", fixed.get("max_tiles")))}


def score_cv(cv: dict, classes: list[str], n_folds: int | None = None) -> dict:
    """
    Out-of-fold measures of a cross-validation (see mil_cv.cross_validate), pooled over all slides.

    Returns macro_f1, log_loss, recall_<class>, f1_<class>, fold_std (standard deviation over the fold and repeat
    macro-F1 values) and fold_se (fold_std / sqrt(number of folds)).
    """
    labels = list(range(len(classes)))
    truth, proba = cv["y"], cv["proba"]
    pred = proba.argmax(1)
    recalls = recall_score(truth, pred, labels=labels, average=None, zero_division=0)
    f1s = f1_score(truth, pred, labels=labels, average=None, zero_division=0)
    per_fold = cv["per_fold"]["macro_f1"].to_numpy()
    n_folds = n_folds or int(cv["per_fold"]["fold"].nunique())
    std = float(np.std(per_fold, ddof=1)) if len(per_fold) > 1 else float("nan")

    return {"macro_f1": float(f1_score(truth, pred, labels=labels, average="macro", zero_division=0)),
            "log_loss": float(log_loss(truth, np.clip(proba, 1e-7, 1), labels=labels)),
            **{f"recall_{c}": float(r) for c, r in zip(classes, recalls)}, **{f"f1_{c}": float(f) for c, f in zip(classes, f1s)},
            "fold_std": std, "fold_se": std / np.sqrt(n_folds)}


def objective_factory(data: dict, folds: list[dict], device: torch.device, fixed: dict, weighting: str | None, seed: int, space: dict | None = None):
    """
    The objective of the optimisation: one trial is one cross-validation of a sampled setting, and its value is the pooled out-of-fold macro-F1.

    The measures of the trial (see score_cv), the mean best epoch and the minutes are stored in trial.user_attrs.

    * Arguments:
    data: The output of mil_data.load_bags (relabelled, with the class list of the model)
    folds: The output of mil_cv.make_cv_folds (the same folds for every trial)
    device: The device to run on
    fixed: The train_model arguments that are not searched (epochs, patience, tile_dropout, bag_weight, weighted_val_loss, tiers ...)
    weighting: The loss weighting rule
    seed: The base seed of the cross-validation (the same for every trial)
    space: The search ranges (default DEFAULT_SPACE)
    """

    def objective(trial) -> float:
        # STEP 1: Sample the setting and run the cross-validation
        setting = sample_setting(trial, space)
        start = time()
        cv = cross_validate("clam_mb", data, folds, device, weighting, 1, seed, **setting_kwargs(setting, fixed))

        # STEP 2: Store the measures and return the pooled out-of-fold macro-F1
        scores = score_cv(cv, data["classes"])
        for key, value in scores.items():
            trial.set_user_attr(key, value)
        trial.set_user_attr("mean_best_epoch", float(np.mean([r["best_epoch"] for r in cv["runs"]])))
        trial.set_user_attr("minutes", (time() - start) / 60)

        return scores["macro_f1"]

    return objective


def _study_exists(name: str, storage: str) -> bool:
    import optuna
    return name in [s.study_name for s in optuna.get_all_study_summaries(storage)]


def run_study(name: str, storage: str, objective, n_trials: int, seed: int, baseline: dict | None = None):
    """
    Create (or resume) an optuna study and run trials until it holds n_trials finished ones.

    The sampler is TPE with joint modelling of the parameters. The baseline setting, if given, is the first trial,
    so it is scored on the same folds as the others. The study is stored in SQLite, so a restart continues it.
    """
    import optuna

    # STEP 1: Create or load the study (the sampler seed depends on the trials already done, so a resumed study does not repeat its random draws)
    existing = len(optuna.load_study(study_name=name, storage=storage).trials) if _study_exists(name, storage) else 0
    study = optuna.create_study(study_name=name, storage=storage, direction="maximize", load_if_exists=True,
                                sampler=optuna.samplers.TPESampler(seed=seed + existing, multivariate=True, n_startup_trials=10))
    if baseline is not None and not study.trials:
        study.enqueue_trial(baseline)

    # STEP 2: Run the missing trials
    done = len([t for t in study.trials if t.state.is_finished()])
    if n_trials > done:
        study.optimize(objective, n_trials=n_trials - done, show_progress_bar=True)

    return study


def trials_table(study) -> pd.DataFrame:
    """One row per finished trial: number, macro_f1 (the value), the six parameters and the stored measures; sorted by macro-F1 (ties: lower log-loss)."""
    rows = [{"trial": t.number, "macro_f1": t.value, **t.params, **t.user_attrs} for t in study.trials if t.state.name == "COMPLETE"]
    table = pd.DataFrame(rows)

    return table.sort_values(["macro_f1", "log_loss"], ascending=[False, True]).reset_index(drop=True)


def eligible_trials(table: pd.DataFrame, min_class_recall: float | dict | None, classes: list[str]) -> pd.DataFrame:
    """
    The trials in which every class reaches its out-of-fold recall floor, in the order of the table.

    * Arguments:
    table: The output of trials_table (or a table with the same recall_<class> columns)
    min_class_recall: One floor for every class, a dict {class: floor} (a class that is missing has no floor) or None (no floor)
    classes: The class names

    Raises ValueError, with the best minimum recall that any trial reaches, if no trial passes.
    """
    if min_class_recall is None:
        return table
    floors = {c: min_class_recall for c in classes} if not isinstance(min_class_recall, dict) else {c: f for c, f in min_class_recall.items() if c in classes}
    eligible = table[np.logical_and.reduce([table[f"recall_{c}"] >= f for c, f in floors.items()])] if floors else table
    if eligible.empty:
        lowest = table[[f"recall_{c}" for c in classes]].min(axis=1).max()
        best_by_class = {c: round(float(table[f"recall_{c}"].max()), 3) for c in classes}
        raise ValueError(f"No trial reaches the recall floor {min_class_recall} for every class. The best minimum recall of any trial is {lowest:.3f}; "
                         f"the best recall per class is {best_by_class}. Lower the floor (or give one floor per class) or run more trials.")

    return eligible


def select_setting(table: pd.DataFrame, min_class_recall: float | dict | None, classes: list[str]) -> tuple[pd.Series, pd.DataFrame]:
    """
    Choose the final setting from the trials table, without the Testing set.

    STEP 1 keeps the trials in which every class has an out-of-fold recall of at least its floor (see eligible_trials; None = no floor).
    STEP 2 takes the best pooled macro-F1 among them. STEP 3 applies the one-standard-error rule: among the trials whose macro-F1 is within
    one standard error (fold_se) of the best, the cheapest tile cap (None counts as the most expensive), then the higher dropout, then the higher weight decay.

    Returns the chosen row and the table of the candidates within one standard error (best first).
    """
    eligible = eligible_trials(table, min_class_recall, classes)
    best = eligible.iloc[0]
    candidates = eligible[eligible["macro_f1"] >= best["macro_f1"] - (best["fold_se"] if np.isfinite(best["fold_se"]) else 0.0)].copy()
    candidates["_cost"] = candidates["max_tiles"].astype(float).fillna(np.inf) if "max_tiles" in candidates else 0.0
    chosen = candidates.sort_values(["_cost", "dropout", "weight_decay"], ascending=[True, False, False]).iloc[0]

    return chosen.drop("_cost"), candidates.drop(columns="_cost")


def oof_by_repeat(cv: dict, data: dict, folds: list[dict], device: torch.device) -> list[np.ndarray]:
    """
    The out-of-fold probabilities of every repeat of a cross-validation, one array [n_slides, K] per repeat.

    cross_validate averages the repeats; this predicts the held-out slides of every fold again with the run of
    each repeat, to compare a single model with the repeat-averaged ensemble.
    """
    position = {s: i for i, s in enumerate(cv["slides"])}
    n_folds = len(folds)
    repeats = len(cv["runs"]) // n_folds
    result = [np.zeros_like(cv["proba"]) for _ in range(repeats)]
    for i, run in enumerate(cv["runs"]):
        repeat, fold = divmod(i, n_folds)
        held = folds[fold]["held"]
        result[repeat][[position[s] for s in held]] = predict(run["model"], data, held, device, run["scaler"])["proba"]

    return result


def single_vs_ensemble(cv: dict, per_repeat: list[np.ndarray], classes: list[str]) -> pd.DataFrame:
    """Out-of-fold macro-F1 and log-loss of each single repeat and of the average of the repeats (the ensemble)."""
    labels, truth = list(range(len(classes))), cv["y"]
    rows = {f"single model (repeat {r})": p for r, p in enumerate(per_repeat)}
    rows["ensemble (mean of repeats)"] = np.mean(per_repeat, axis=0)

    return pd.DataFrame({name: {"macro_f1": f1_score(truth, p.argmax(1), labels=labels, average="macro", zero_division=0),
                                "log_loss": log_loss(truth, np.clip(p / p.sum(1, keepdims=True), 1e-7, 1), labels=labels),
                                **{f"recall_{c}": r for c, r in zip(classes, recall_score(truth, p.argmax(1), labels=labels, average=None, zero_division=0))}}
                         for name, p in rows.items()}).T


def save_ensemble(runs: list[dict], directory, config: dict | None = None) -> Path:
    """
    Save every model of an ensemble (weights, scaler, classes, settings and the early-stopping result of each) to one file,
    <directory>/ensemble_<date>_<time>.pt; load_ensemble gives the models back in the form predict_ensemble takes.
    """
    folder = Path(directory)
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"ensemble_{datetime.now():%Y-%m-%d_%H-%M-%S}.pt"
    models = [{"model_name": r["model_name"], "model_kwargs": r["model_kwargs"], "in_dim": r["in_dim"], "classes": r["classes"], "state_dict": r["model"].state_dict(),
               "scaler": r["scaler"], "seed": r["seed"], "best_epoch": r["best_epoch"], "validation": r["best"]} for r in runs]
    torch.save({"models": models, "config": json.loads(json.dumps(config or {}, default=str))}, path)

    return path


def load_ensemble(path, device: torch.device | None = None) -> dict:
    """
    Load a file written by save_ensemble.

    Returns {"runs": list of dicts with model (eval mode), scaler, classes, seed, best_epoch, ready for mil_train.predict_ensemble, "classes", "config"}.
    """
    from utils.device import get_device
    device = device or get_device()
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    runs = []
    for entry in checkpoint["models"]:
        model = build_model(entry["model_name"], entry["in_dim"], len(entry["classes"]), **entry["model_kwargs"])
        model.load_state_dict(entry["state_dict"])
        runs.append({**{k: entry[k] for k in ("scaler", "classes", "seed", "best_epoch", "model_name")}, "best": entry["validation"], "model": model.to(device).eval()})

    return {"runs": runs, "classes": checkpoint["models"][0]["classes"], "config": checkpoint["config"]}


def plot_tuning(study, show: bool = True):
    """
    The Bayesian optimisation: the macro-F1 of every trial, the importance of each parameter and the macro-F1 by tile cap.

    * Arguments:
    study: The optuna study
    show: Whether to call plt.show() (the figure is returned either way)
    """
    table = trials_table(study).sort_values("trial")
    with_cap = "max_tiles" in table
    notes = [(0, 1, "Pooled out-of-fold macro-averaged F1 of every trial (points) and the best value so far (line); the first trial is the setting of notebook 02_1. "
                    "The first trials are drawn at random and the later ones are proposed from the earlier results (tree-structured Parzen estimator, Bergstra et al. 2011).\n"
                    "The same folds score every trial, so the best value is optimistic; the choice is confirmed with new seeds before it is used."),
             (1, 2, "Importance of each parameter for the macro-F1, estimated by optuna from the trials (fANOVA, Hutter et al. 2014). "
                    "The importance describes the explored ranges only and, with few trials, is itself uncertain.\n"
                    "A parameter with a low importance can be fixed at its simplest value."),
             (2, 3, "Macro-F1 of the trials by tile cap per slide in training (None: every tile); bars, the median. "
                    "A larger cap costs time and memory, so it is worth choosing only if its macro-F1 is clearly higher.\n"
                    "Trials differ in the other parameters as well, so the figure shows association, not the effect of the cap alone.")]
    notes = notes if with_cap else notes[:2]
    with plt.rc_context(STYLE):
        fig = plt.figure(figsize=(16 if with_cap else 11, 5.8), layout="constrained")
        top = split_with_notes(fig, notes, [1] * len(notes), 16 if with_cap else 11, plot_height=3.6, note_height=1.9)
        a, b, *c = top.subplots(1, len(notes))

        # STEP 1: Optimisation history
        a.scatter(table["trial"], table["macro_f1"], color=SERIES[0], s=26, edgecolor=SURFACE, zorder=3)
        a.step(table["trial"], table["macro_f1"].cummax(), where="post", color=SERIES[1], lw=2)
        a.set_xlabel("trial"); a.set_ylabel("out-of-fold macro F1"); a.set_title("Optimisation history")

        # STEP 2: Parameter importance
        try:
            import optuna
            importance = optuna.importance.get_param_importances(study)
            names = list(importance)[::-1]
            b.barh(names, [importance[n] for n in names], color=SERIES[0])
        except Exception as error:
            b.text(0.5, 0.5, f"not available\n({type(error).__name__})", ha="center", va="center", color=INK_2, transform=b.transAxes)
        b.set_xlabel("importance"); b.set_title("Importance of the parameters")

        # STEP 3: Macro-F1 by tile cap (only when the cap is searched)
        if not with_cap:
            if show:
                plt.show()
            return fig
        c = c[0]
        caps = [2048, 4096, 8192, None]
        labels = [("none" if cap is None else str(cap)) for cap in caps]
        rng = np.random.default_rng(0)
        for i, cap in enumerate(caps):
            values = table.loc[table["max_tiles"].isna() if cap is None else table["max_tiles"] == cap, "macro_f1"].to_numpy()
            if len(values):
                c.scatter(i + rng.uniform(-0.12, 0.12, len(values)), values, color=SERIES[0], s=26, edgecolor=SURFACE, zorder=3)
                c.hlines(np.median(values), i - 0.25, i + 0.25, color=SERIES[1], lw=2.5)
        c.set_xticks(range(len(caps)), labels); c.set_xlabel("tiles per slide in training"); c.set_ylabel("out-of-fold macro F1"); c.set_title("Tile cap")

    if show:
        plt.show()

    return fig
