# IMPORTS *****************************************************************************************
# External libraries
import copy
import os
import random
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import balanced_accuracy_score, f1_score

# Internal imports ********************************************************************************
from utils.class_weights import class_counts, compute_class_weights, weighted_cross_entropy
from utils.classification_metrics import TIERS
from utils.mil_data import batch_iterator, fit_scaler
from utils.mil_models import build_model


# Helper functions ********************************************************************************
def set_seed(seed: int) -> None:
    """
    Seed every source of randomness: Python, NumPy and PyTorch (CPU and GPU), and ask PyTorch for
    deterministic algorithms (a warning, not an error, when an operation has none).

    The same seed gives the same result on the same device; results on a CPU and a GPU can differ
    in the last digits because floating-point operations are ordered differently.
    """
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)


def _autocast(device: torch.device):
    """bfloat16 mixed precision on CUDA; plain float32 on every other device."""
    return torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda")


def _tier_matrix(classes: list[str]) -> torch.Tensor:
    """[n_classes, n_tiers] one-hot matrix that maps a class to its clinical tier (see classification_metrics.TIERS)."""
    names = list(TIERS)
    matrix = torch.zeros(len(classes), len(names))
    for i, c in enumerate(classes):
        owner = [t for t, members in TIERS.items() if c in members]
        if not owner:
            raise ValueError(f"Class {c!r} belongs to no tier of classification_metrics.TIERS; the coarse loss cannot be used.")
        matrix[i, names.index(owner[0])] = 1.0

    return matrix


def _coarse_loss(logits: torch.Tensor, y: torch.Tensor, tier_matrix: torch.Tensor, tier_weights: torch.Tensor) -> torch.Tensor:
    """Tier-weighted negative log-likelihood of the true tier under the summed class probabilities of each tier."""
    tier_probability = torch.softmax(logits.float(), dim=1) @ tier_matrix
    tier_of_class = tier_matrix.argmax(1)

    return torch.nn.functional.nll_loss(torch.log(tier_probability.clamp(min=1e-8)), tier_of_class[y], weight=tier_weights)


def _label_map(table: pd.DataFrame, classes: list[str]) -> dict[str, int]:
    """{slide: class index}, with the class order read from the label sheet."""
    index = {c: i for i, c in enumerate(classes)}
    return {s: index[str(l)] for s, l in zip(table["slide"], table["label"])}


# Main functions **********************************************************************************
@torch.no_grad()
def predict(
    model: torch.nn.Module,
    data: dict,
    slides: list[str],
    device: torch.device,
    scaler: dict | None = None,
    attention: bool = False,
) -> dict:
    """
    Class probabilities of the given slides (every tile of every slide is used).

    * Arguments:
    model: A model from mil_models.build_model
    data: The output of mil_data.load_bags
    slides: The slides to predict
    device: The device to run on
    scaler: The output of mil_data.fit_scaler (None = no standardisation)
    attention: Whether to also return the attention of every tile

    Returns {"proba": array [n_slides, n_classes], "attention": list of arrays (or None)}.
    """
    model.eval()
    labels = {s: 0 for s in slides}
    probabilities, weights = [], []
    for x, mask, _, _ in batch_iterator(data["bags"], slides, labels, 1, device, scaler):
        with _autocast(device):
            logits, attn = model(x, mask)
        probabilities.append(torch.softmax(logits.float(), dim=1).cpu().numpy())
        if attention:
            weights.append(attn[0].float().cpu().numpy())

    return {"proba": np.concatenate(probabilities), "attention": weights if attention else None}


def _scores(y: np.ndarray, proba: np.ndarray, n_classes: int) -> dict:
    """Validation measures used for model selection."""
    pred = proba.argmax(1)
    loss = float(-np.log(np.clip(proba[np.arange(len(y)), y], 1e-12, None)).mean())

    return {
        "val_loss": loss,
        "val_macro_f1": float(f1_score(y, pred, labels=range(n_classes), average="macro", zero_division=0)),
        "val_balanced_accuracy": float(balanced_accuracy_score(y, pred)),
    }


def train_model(
    model_name: str,
    data: dict,
    class_weights: torch.Tensor | None,
    device: torch.device,
    seed: int,
    epochs: int = 100,
    lr: float = 2e-4,
    weight_decay: float = 1e-2,
    batch_size: int = 16,
    max_tiles: int | None = 2048,
    patience: int = 20,
    warmup_epochs: int = 5,
    verbose: bool = True,
    tile_dropout: float = 0.0,
    coarse_weight: float = 0.0,
    **model_kwargs,
) -> dict:
    """
    Train one model on the Training slides, select the epoch on the Validation slides, return the best.

    Only slides of the Training and Validation sets are ever used here: the labels of the Testing set
    are not touched. The loss is the class-weighted cross-entropy (class_weights; None = unweighted);
    selection uses the unweighted Validation macro-F1 (ties broken by the Validation loss).

    * Arguments:
    model_name: "linear", "abmil" or "clam_mb"
    data: The output of mil_data.load_bags
    class_weights: The loss weights (compute them from the Training counts only), or None
    device: The device to run on (GPU when available, else CPU, see utils.device.get_device)
    seed: The random seed of this run (initialisation, batch order, tile subsampling)
    epochs: The maximum number of epochs
    lr: The peak learning rate (AdamW, linear warm-up then cosine decay)
    weight_decay: The AdamW weight decay
    batch_size: The number of slides per batch
    max_tiles: The maximum number of tiles per slide in training (random subset each epoch; None = all)
    patience: The number of epochs without improvement of the Validation macro-F1 before stopping; epochs of the
        warm-up are not counted, so a run cannot stop before the warm-up has ended
    warmup_epochs: The number of warm-up epochs
    verbose: Whether to print one line every 10 epochs
    tile_dropout: The fraction of the tiles of each training bag removed at random in every epoch (0 = none);
        validation always uses every tile
    coarse_weight: The weight of the coarse auxiliary loss (0 = none, the default); the loss is
        (1 - coarse_weight) * (loss of the model) + coarse_weight * (coarse loss), where the coarse loss is the
        negative log-likelihood of the true clinical tier (benign, atypical, malignant) under the summed
        probabilities of the classes of each tier. Its tier weights are the mean class weight of the
        Training slides of the tier, so the class weighting also acts on the tiers
    model_kwargs: Passed to the model (hidden, attention_dim, dropout)

    For a model with needs_labels (CLAM-MB) the batch labels are passed to the model and the loss is
    bag_weight * (class-weighted cross-entropy) + (1 - bag_weight) * (instance loss of the model); the
    history then also has a train_instance_loss column. train_loss is always the slide-level loss, so the
    models stay comparable. With coarse_weight > 0 the history has a train_coarse_loss column as well.

    Returns a dict with model (best weights, eval mode), scaler, history (DataFrame), best_epoch, epochs_run,
    best (the Validation measures at the best epoch), seed, model_name, weight_decay, tile_dropout and coarse_weight.
    """

    # STEP 1: Seed, select the slides, and make sure that no Testing slide is used
    set_seed(seed)
    table, classes = data["table"], data["classes"]
    train_table, val_table = table[table["set"] == "Training"], table[table["set"] == "Validation"]
    if train_table.empty or val_table.empty:
        raise ValueError("Both the Training and the Validation set need slides with features.")
    test_slides = set(table.loc[table["set"] == "Testing", "slide"])
    train_slides, val_slides = train_table["slide"].tolist(), val_table["slide"].tolist()
    assert not test_slides & (set(train_slides) | set(val_slides)), "A Testing slide would be used for training or selection."

    labels = _label_map(pd.concat([train_table, val_table]), classes)      # the labels of Testing slides are not even available here
    y_val = np.array([labels[s] for s in val_slides])
    scaler = fit_scaler(data["bags"], train_slides)
    generator = torch.Generator().manual_seed(seed)
    if not 0.0 <= coarse_weight <= 1.0:
        raise ValueError(f"coarse_weight must be in [0, 1], got {coarse_weight}")

    # STEP 2: Model, loss, optimiser and learning-rate schedule
    model = build_model(model_name, data["dim"], len(classes), **model_kwargs).to(device)
    loss_fn = weighted_cross_entropy(class_weights if class_weights is not None else torch.ones(len(classes))).to(device)
    optimiser = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    schedule = torch.optim.lr_scheduler.LambdaLR(
        optimiser, lambda e: (e + 1) / warmup_epochs if e < warmup_epochs else 0.5 * (1 + np.cos(np.pi * (e - warmup_epochs) / max(epochs - warmup_epochs, 1))))

    # STEP 2b: The coarse (tier) loss: class-to-tier matrix and tier weights from the Training slides only
    if coarse_weight > 0:
        tier_matrix = _tier_matrix(classes).to(device)
        counts = torch.as_tensor(class_counts([classes.index(str(l)) for l in train_table["label"]], len(classes)), dtype=torch.float32)
        class_w = class_weights.float() if class_weights is not None else torch.ones(len(classes))
        tier_weights = ((counts * class_w) @ tier_matrix.cpu()) / (counts @ tier_matrix.cpu()).clamp(min=1)
        tier_weights = tier_weights.to(device)

    # STEP 3: Train, validate every epoch, keep the best epoch
    needs_labels = getattr(model, "needs_labels", False)
    history, best, best_state, best_epoch, stale = [], None, None, 0, 0
    for epoch in range(1, epochs + 1):
        model.train()
        losses, instance_losses, coarse_losses = [], [], []
        for x, mask, y, _ in batch_iterator(data["bags"], train_slides, labels, batch_size, device, scaler, max_tiles, True, generator, tile_dropout):
            optimiser.zero_grad(set_to_none=True)
            with _autocast(device):
                logits, _ = model(x, mask, y) if needs_labels else model(x, mask)
                bag_loss = loss_fn(logits.float(), y)
                loss = model.bag_weight * bag_loss + (1 - model.bag_weight) * model.instance_loss.float() if needs_labels else bag_loss
                if coarse_weight > 0:
                    coarse_loss = _coarse_loss(logits, y, tier_matrix, tier_weights)
                    loss = (1 - coarse_weight) * loss + coarse_weight * coarse_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimiser.step()
            losses.append(bag_loss.item())
            if needs_labels:
                instance_losses.append(model.instance_loss.item())
            if coarse_weight > 0:
                coarse_losses.append(coarse_loss.item())
        schedule.step()

        scores = _scores(y_val, predict(model, data, val_slides, device, scaler)["proba"], len(classes))
        history.append({"epoch": epoch, "train_loss": float(np.mean(losses)), "lr": optimiser.param_groups[0]["lr"], **scores,
                        **({"train_instance_loss": float(np.mean(instance_losses))} if needs_labels else {}),
                        **({"train_coarse_loss": float(np.mean(coarse_losses))} if coarse_weight > 0 else {})})

        better = best is None or (scores["val_macro_f1"], -scores["val_loss"]) > (best["val_macro_f1"], -best["val_loss"])
        if better:
            best, best_epoch, stale = scores, epoch, 0
            best_state = copy.deepcopy({k: v.detach().cpu() for k, v in model.state_dict().items()})
        else:
            stale += 1
        if verbose and (epoch % 10 == 0 or epoch == 1):
            print(f"  [{model_name} seed {seed}] epoch {epoch:3d}  train loss {history[-1]['train_loss']:.3f}  val loss {scores['val_loss']:.3f}  val macro-F1 {scores['val_macro_f1']:.3f}")
        if stale >= patience and epoch > warmup_epochs:
            break

    # STEP 4: Restore the best epoch
    model.load_state_dict(best_state)
    model.eval()
    if verbose:
        print(f"  [{model_name} seed {seed}] best epoch {best_epoch}: val macro-F1 {best['val_macro_f1']:.3f}, balanced accuracy {best['val_balanced_accuracy']:.3f}")

    return {"model": model, "scaler": scaler, "history": pd.DataFrame(history), "best_epoch": best_epoch, "epochs_run": len(history), "best": best, "seed": seed,
            "model_name": model_name, "model_kwargs": model_kwargs, "weight_decay": weight_decay, "tile_dropout": tile_dropout, "coarse_weight": coarse_weight, "in_dim": data["dim"], "classes": classes}


def run_seeds(model_name: str, seeds: list[int], data: dict, class_weights: torch.Tensor | None, device: torch.device, **kwargs) -> list[dict]:
    """Train the same model once per seed (see train_model); returns the list of results."""
    return [train_model(model_name, data, class_weights, device, seed, **kwargs) for seed in seeds]


def predict_ensemble(results: list[dict], data: dict, slides: list[str], device: torch.device, attention: bool = False) -> dict:
    """
    Predict with every model of a multi-seed run and average the probabilities (and attention).

    Returns {"proba": mean probabilities [n, K], "per_seed": list of probabilities, "attention": mean
    attention per slide (or None)}.
    """
    outputs = [predict(r["model"], data, slides, device, r["scaler"], attention) for r in results]
    mean_attention = [np.mean([o["attention"][i] for o in outputs], axis=0) for i in range(len(slides))] if attention else None

    return {"proba": np.mean([o["proba"] for o in outputs], axis=0), "per_seed": [o["proba"] for o in outputs], "attention": mean_attention}


def training_class_weights(data: dict, method: str | None) -> torch.Tensor | None:
    """Loss weights from the Training slides only; method None (or "none") gives no weights."""
    if method in (None, "none"):
        return None
    table = data["table"][data["table"]["set"] == "Training"]
    index = {c: i for i, c in enumerate(data["classes"])}

    return compute_class_weights(class_counts([index[str(l)] for l in table["label"]], len(data["classes"])), method)


def compare_weightings(data: dict, device: torch.device, seeds: list[int], options: tuple[str, ...] = ("none", "sqrt", "inverse"), model_name: str = "abmil", **kwargs) -> pd.DataFrame:
    """
    Compare loss weightings on the Validation set (never on the Testing set).

    Each option is trained once per seed; the table gives the mean ± standard deviation over seeds of the
    best Validation macro-F1 and balanced accuracy, sorted by macro-F1. The first row is the choice.

    * Arguments:
    data: The output of mil_data.load_bags
    device: The device to run on
    seeds: The seeds (the same for every option)
    options: The weightings to compare ("none", "sqrt", "inverse", "effective")
    model_name: The model to train
    kwargs: Passed to train_model (epochs, lr, ...)
    """
    rows = []
    for option in options:
        runs = run_seeds(model_name, seeds, data, training_class_weights(data, option), device, verbose=False, **kwargs)
        f1 = np.array([r["best"]["val_macro_f1"] for r in runs])
        bal = np.array([r["best"]["val_balanced_accuracy"] for r in runs])
        rows.append({"weighting": option, "val_macro_f1_mean": f1.mean(), "val_macro_f1_sd": f1.std(ddof=1) if len(f1) > 1 else 0.0,
                     "val_balanced_accuracy_mean": bal.mean(), "val_balanced_accuracy_sd": bal.std(ddof=1) if len(bal) > 1 else 0.0})

    return pd.DataFrame(rows).sort_values("val_macro_f1_mean", ascending=False).reset_index(drop=True)


def regularisation_grid(dropouts: tuple[float, ...], weight_decays: tuple[float, ...], tile_dropouts: tuple[float, ...]) -> list[dict]:
    """Every combination of the three regularisers as a list of {"dropout", "weight_decay", "tile_dropout"} dicts."""
    return [{"dropout": d, "weight_decay": w, "tile_dropout": t} for t in tile_dropouts for d in dropouts for w in weight_decays]


def compare_regularisation(
    data: dict,
    device: torch.device,
    seeds: list[int],
    class_weights: torch.Tensor | None,
    grid: list[dict],
    model_name: str = "abmil",
    **kwargs,
) -> pd.DataFrame:
    """
    Compare combinations of dropout, weight decay and tile dropout on the Validation set (never on the Testing set).

    Each combination is trained once per seed (same seeds for every combination). The table gives, per
    combination, the mean and standard deviation over seeds of the best Validation macro-F1 and balanced
    accuracy, the mean best epoch and number of epochs run, and the training and Validation loss at the
    best epoch with their difference (a direct measure of overfitting). The training loss is the
    class-weighted loss with dropout active and the Validation loss is unweighted, so the gap is only
    comparable between rows of the same table. Rows are sorted by macro-F1 (ties: lower Validation loss);
    the first row is the choice.

    * Arguments:
    data: The output of mil_data.load_bags
    device: The device to run on
    seeds: The seeds (the same for every combination)
    class_weights: The loss weights (from the Training counts only), or None
    grid: A list of dicts with dropout, weight_decay and tile_dropout (see regularisation_grid)
    model_name: The model to train ("linear", "abmil" or "clam_mb")
    kwargs: Passed to train_model (epochs, lr, patience, model settings such as k_sample ...); the three
        regularisers of the grid override the same keys here
    """
    rows = []
    for setting in grid:
        runs = run_seeds(model_name, seeds, data, class_weights, device, verbose=False, **{**kwargs, **setting})
        f1 = np.array([r["best"]["val_macro_f1"] for r in runs])
        bal = np.array([r["best"]["val_balanced_accuracy"] for r in runs])
        at_best = [r["history"].set_index("epoch").loc[r["best_epoch"]] for r in runs]
        train_loss, val_loss = np.mean([h["train_loss"] for h in at_best]), np.mean([h["val_loss"] for h in at_best])
        rows.append({"model": model_name, **setting,
                     "val_macro_f1_mean": f1.mean(), "val_macro_f1_sd": f1.std(ddof=1) if len(f1) > 1 else 0.0,
                     "val_balanced_accuracy_mean": bal.mean(), "val_balanced_accuracy_sd": bal.std(ddof=1) if len(bal) > 1 else 0.0,
                     "best_epoch_mean": np.mean([r["best_epoch"] for r in runs]), "epochs_run_mean": np.mean([r["epochs_run"] for r in runs]),
                     "train_loss": train_loss, "val_loss": val_loss, "loss_gap": val_loss - train_loss})

    return pd.DataFrame(rows).sort_values(["val_macro_f1_mean", "val_loss"], ascending=[False, True]).reset_index(drop=True)


def device_description(device: torch.device) -> dict:
    """Name and numeric precision of the device that a run used (recorded with the results)."""
    import platform
    name = torch.cuda.get_device_name(device) if device.type == "cuda" else platform.processor() or platform.machine()

    return {"type": device.type, "name": name, "precision": "bfloat16 autocast" if device.type == "cuda" else "float32"}


def select_best_run(runs: dict[str, list[dict]]) -> dict:
    """
    Select the single best trained model: the run with the highest macro-F1 on its early-stopping slides
    (ties: lowest early-stopping loss) among all models, folds and repeats. The Testing set plays no part.

    * Arguments:
    runs: {model: list of train_model results}

    Returns the train_model result of the selected run.
    """
    candidates = [r for results in runs.values() for r in results]

    return max(candidates, key=lambda r: (r["best"]["val_macro_f1"], -r["best"]["val_loss"]))


def save_best_model(runs: dict[str, list[dict]], directory, config: dict | None = None) -> "Path":
    """
    Save the single best trained classifier to <directory>/classifier_model_<date>_<time>.pt.

    "Best" is chosen by select_best_run: the highest macro-F1 on the early-stopping slides of the fold (ties:
    lowest early-stopping loss) among all models, folds and repeats. The choice never uses the Testing set. The
    file holds everything needed to use the model later (see
    load_classifier): the weights, the class names of BRACS.xlsx, the feature scaler (fitted on the Training
    slides), the architecture settings, the Validation result and the configuration of the run.

    * Arguments:
    runs: {model: list of train_model results}
    directory: The folder to write to (created if missing)
    config: Optional dict with the parameters of the run (seeds, training arguments, tiling, versions, device)

    Returns the path of the file.
    """
    import json
    from datetime import datetime
    from pathlib import Path

    # STEP 1: The best run on the early-stopping slides
    best = select_best_run(runs)

    # STEP 2: Write one self-describing file
    folder = Path(directory)
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"classifier_model_{datetime.now():%Y-%m-%d_%H-%M-%S}.pt"
    torch.save({
        "model_name": best["model_name"], "model_kwargs": best["model_kwargs"], "in_dim": best["in_dim"], "classes": best["classes"],
        "state_dict": best["model"].state_dict(), "scaler": best["scaler"], "seed": best["seed"], "best_epoch": best["best_epoch"],
        "validation": best["best"], "config": json.loads(json.dumps(config or {}, default=str)),
    }, path)

    return path


def load_classifier(path, device: torch.device | None = None) -> dict:
    """
    Load a classifier written by save_best_model.

    * Arguments:
    path: The .pt file
    device: The device to put the model on (None = the device chosen by utils.device.get_device)

    Returns a dict with model (eval mode), scaler, classes (the output order of the model), model_name, seed,
    best_epoch, validation and config. Use mil_train.predict(model, data, slides, device, scaler) to classify slides.
    """
    from utils.device import get_device
    device = device or get_device()
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    model = build_model(checkpoint["model_name"], checkpoint["in_dim"], len(checkpoint["classes"]), **checkpoint["model_kwargs"])
    model.load_state_dict(checkpoint["state_dict"])

    return {**{k: checkpoint[k] for k in ("classes", "scaler", "model_name", "seed", "best_epoch", "validation", "config")}, "model": model.to(device).eval()}
