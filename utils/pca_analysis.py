# IMPORTS *****************************************************************************************
# External libraries
from contextlib import contextmanager
from pathlib import Path
import json
import time
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from sklearn.decomposition import IncrementalPCA, PCA

# Internal imports ********************************************************************************
from utils.cls_overview import GRID, INK_2, SERIES, STYLE
from utils.device import get_device
from utils.figure_captions import split_with_notes

# Constants ***************************************************************************************
VARIANCE_THRESHOLDS = (0.90, 0.95, 0.99, 0.995)


# Helpers *****************************************************************************************
@contextmanager
def timed(label: str):
    """Print the wall-clock time of the enclosed block."""
    start = time.perf_counter()
    yield
    print(f"[{label}] {time.perf_counter() - start:.1f} s")


def _allocate(quota: np.ndarray, budget: int, rng: np.random.Generator) -> np.ndarray:
    """Water-filling: equal share per item, items with a smaller quota give the rest to the others."""
    quota = np.asarray(quota, dtype=np.int64)
    if quota.sum() <= budget:
        return quota.copy()
    lo, hi = 0, int(quota.max())
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if np.minimum(quota, mid).sum() <= budget:
            lo = mid
        else:
            hi = mid - 1
    out = np.minimum(quota, lo)
    left = int(budget - out.sum())
    candidates = np.flatnonzero(quota > lo)
    rng.shuffle(candidates)
    out[candidates[:left]] += 1
    return out


def _load_cls(features_dir: Path, model_name: str, slide: str) -> dict | None:
    """Memory-map one slide file (None when it does not exist); patch tokens are never read."""
    path = Path(features_dir) / model_name / f"{slide}.pt"
    if not path.exists():
        return None
    return torch.load(path, map_location="cpu", weights_only=True, mmap=True)


def _scores(pca, X: np.ndarray) -> np.ndarray:
    """Projection on the principal axes (centred, not whitened), whatever the solver."""
    return (np.asarray(X, dtype=np.float32) - pca.mean_.astype(np.float32)) @ pca.components_.astype(np.float32).T


def _unit(X: torch.Tensor) -> torch.Tensor:
    return X / X.norm(dim=1, keepdim=True).clamp_min(1e-12)


# Main functions: data ****************************************************************************
def split_fit_eval(table: pd.DataFrame) -> dict:
    """
    Official Training slides fit the PCA, Validation slides evaluate it. Patients that have slides
    in both sets are excluded from evaluation, so no patient is seen in fitting and evaluation.
    Testing slides are never used for either.

    * Arguments:
    table: The slide table of build_slide_table (columns slide, patient, set)

    Returns a dict with fit and eval (lists of slide names) and the excluded slides.
    """

    # STEP 1: Fit slides and evaluation slides
    fit = table[table["set"] == "Training"]
    held = table[table["set"] == "Validation"]

    # STEP 2: Remove from evaluation the patients that are also in the fit set
    shared = set(fit["patient"]) & set(held["patient"])
    eval_ = held[~held["patient"].isin(shared)]

    # STEP 3: Checks
    assert not set(fit["patient"]) & set(eval_["patient"])
    assert not (set(fit["slide"]) | set(eval_["slide"])) & set(table.loc[table["set"] == "Testing", "slide"])

    return {"fit": fit["slide"].tolist(), "eval": eval_["slide"].tolist(),
            "excluded": held.loc[held["patient"].isin(shared), "slide"].tolist()}


def sample_tiles(features_dir: str | Path, model_name: str, slides: list[str], labels: dict[str, str],
                 max_per_slide: int, max_total: int, seed: int, stratify_by_label: bool = False) -> dict:
    """
    Draw a reproducible, slide-balanced sample of CLS vectors without loading every slide.

    Every slide contributes at most max_per_slide tiles. When the total exceeds max_total, the
    quotas are reduced by water-filling (equal share per slide, small slides keep all their tiles).
    With stratify_by_label the budget is first shared equally among the classes, then among the
    slides of each class.

    * Arguments:
    features_dir: The folder that holds one subfolder per model
    model_name: The model subfolder with the per-slide .pt files
    slides: The slide names to sample from
    labels: Slide name -> class label
    max_per_slide: The maximum number of tiles taken from one slide
    max_total: The maximum number of tiles in total
    seed: The random seed
    stratify_by_label: Balance the budget across classes first

    Returns a dict with X [N, D] float32, slide [N] (index into names), tile [N] (row in the slide
    file), names, available and taken (tiles per slide) and missing (slides without a file).
    """

    # STEP 1: Count the tiles of each slide (memory-mapped, cheap)
    rng = np.random.default_rng(seed)
    names, available, missing = [], [], []
    for s in slides:
        d = _load_cls(features_dir, model_name, s)
        if d is None:
            missing.append(s)
            continue
        names.append(s)
        available.append(len(d["cls_features"]))
    available = np.array(available, dtype=np.int64)
    if not names:
        raise FileNotFoundError(f"No feature file found in {Path(features_dir) / model_name}")

    # STEP 2: Quota per slide
    quota = np.minimum(available, max_per_slide)
    if stratify_by_label:
        lab = np.array([labels[s] for s in names])
        classes = sorted(set(lab))
        class_quota = _allocate(np.array([quota[lab == c].sum() for c in classes]), max_total, rng)
        taken = np.zeros_like(quota)
        for c, cq in zip(classes, class_quota):
            idx = np.flatnonzero(lab == c)
            taken[idx] = _allocate(quota[idx], int(cq), rng)
    else:
        taken = _allocate(quota, max_total, rng)

    # STEP 3: Draw the tiles of each slide
    xs, slide_idx, tile_idx = [], [], []
    for k, (s, n, t) in enumerate(zip(names, available, taken)):
        if t == 0:
            continue
        rows = np.sort(rng.choice(n, size=int(t), replace=False))
        d = _load_cls(features_dir, model_name, s)
        xs.append(d["cls_features"][torch.from_numpy(rows)].float().numpy())
        slide_idx.append(np.full(len(rows), k))
        tile_idx.append(rows)

    return {"X": np.concatenate(xs), "slide": np.concatenate(slide_idx), "tile": np.concatenate(tile_idx),
            "names": names, "available": available, "taken": taken, "missing": missing}


def sampling_stats(sample: dict, table: pd.DataFrame) -> dict:
    """
    Sampling statistics: one row per slide and one row per class.

    * Arguments:
    sample: The output of sample_tiles
    table: The slide table (columns slide, patient, label, set)

    Returns a dict with per_slide and per_class DataFrames.
    """

    # STEP 1: Per slide
    info = table.set_index("slide").loc[sample["names"], ["patient", "label", "set"]].reset_index()
    info["available"], info["sampled"] = sample["available"], sample["taken"]
    info["fraction"] = info["sampled"] / info["available"]

    # STEP 2: Per class
    per_class = info.groupby("label", observed=True).agg(
        slides=("slide", "size"), patients=("patient", "nunique"), available=("available", "sum"),
        sampled=("sampled", "sum")).reset_index()
    per_class["share"] = per_class["sampled"] / per_class["sampled"].sum()

    return {"per_slide": info, "per_class": per_class}


# Main functions: PCA *****************************************************************************
def fit_pca(X: np.ndarray, dims: list[int], solver: str = "randomized", batch_size: int = 20000,
            seed: int = 42) -> tuple:
    """
    Fit one PCA with enough components for every candidate dimension; the same fit serves all of them.

    The PCA is never whitened here: components_ are orthonormal axes and the scores are not rescaled.

    * Arguments:
    X: The [N, D] training sample
    dims: The candidate dimensions
    solver: "randomized" (sklearn PCA) or "incremental" (IncrementalPCA, fed in batches)
    batch_size: The batch size of the incremental solver
    seed: The random seed

    Returns (pca, usable_dims): the fitted model (with attribute total_variance_) and the candidate
    dimensions that the fit can support (dimensions above the fitted rank are dropped and reported).
    """

    # STEP 1: Number of components
    n_comp = int(min(max(dims), X.shape[0] - 1, X.shape[1]))
    usable = [d for d in dims if d <= n_comp]
    if len(usable) < len(dims):
        print(f"Dimensions {[d for d in dims if d > n_comp]} exceed the {n_comp} fitted components and are dropped.")

    # STEP 2: Fit
    if solver == "randomized":
        pca = PCA(n_components=n_comp, svd_solver="randomized", random_state=seed).fit(X)
    elif solver == "incremental":
        pca = IncrementalPCA(n_components=n_comp, batch_size=max(batch_size, n_comp))
        for i in range(0, len(X), batch_size):
            part = X[i:i + batch_size]
            if len(part) >= n_comp:
                pca.partial_fit(part)
    else:
        raise ValueError("solver must be 'randomized' or 'incremental'")

    # STEP 3: Total variance of the sample (the denominator of the variance ratios)
    pca.total_variance_ = float(pca.explained_variance_[0] / pca.explained_variance_ratio_[0])

    return pca, usable


def variance_table(pca, dims: list[int], thresholds: tuple = VARIANCE_THRESHOLDS) -> tuple:
    """
    Cumulative explained variance at each candidate dimension and the number of components needed
    for each variance threshold (NaN when the fitted components do not reach it).

    * Arguments:
    pca: The fitted PCA
    dims: The candidate dimensions
    thresholds: The cumulative variance levels

    Returns (by_dim, by_threshold) DataFrames.
    """

    # STEP 1: Cumulative variance
    cum = np.cumsum(pca.explained_variance_ratio_)
    by_dim = pd.DataFrame({"dimensions": dims, "explained_variance": [cum[d - 1] for d in dims]})
    by_dim["residual"] = 1 - by_dim["explained_variance"]

    # STEP 2: Components for each threshold
    rows = []
    for t in thresholds:
        k = int(np.searchsorted(cum, t) + 1) if cum[-1] >= t else np.nan
        rows.append({"threshold": t, "components": k})

    return by_dim, pd.DataFrame(rows)


def spectrum_stats(pca, dim_total: int | None = None) -> dict:
    """
    Eigenvalue spectrum, normalised distribution and effective rank (Roy and Vetterli 2007):
    r_eff = exp(-sum p_i log p_i), with p_i = lambda_i / sum lambda.

    With a truncated spectrum the kept eigenvalues are normalised by their own sum (r_trunc) and
    the variance left in the tail is used to bound the full-spectrum value: the lower bound
    puts it in a single component and the upper bound spreads it evenly over the missing ones.

    * Arguments:
    pca: The fitted PCA
    dim_total: The embedding dimension (default: from the fitted model)

    Returns a dict with eigenvalues, p, effective_rank (truncated), truncated (bool),
    effective_rank_bounds (low, high) and elbows (chord and curvature indices, 1-based).
    """

    # STEP 1: Spectrum and effective rank
    ev = pca.explained_variance_.astype(np.float64)
    D = dim_total or pca.components_.shape[1]
    p = ev / ev.sum()
    r_trunc = float(np.exp(-(p * np.log(p)).sum()))
    truncated = len(ev) < D

    # STEP 2: Bounds on the full spectrum
    residual = max(pca.total_variance_ - ev.sum(), 0.0)
    if truncated and residual > 0:
        total = ev.sum() + residual
        q = ev / total
        h_kept = -(q * np.log(q)).sum()
        r_low = float(np.exp(h_kept - (residual / total) * np.log(residual / total)))
        tail = residual / (D - len(ev)) / total
        r_high = float(np.exp(h_kept - (D - len(ev)) * tail * np.log(tail)))
    else:
        r_low = r_high = r_trunc

    # STEP 3: Elbows on the log spectrum: farthest point from the chord, and the largest curvature
    y = np.log10(np.clip(ev, 1e-30, None))
    x = np.arange(len(y), dtype=np.float64)
    chord = np.array([x[-1] - x[0], y[-1] - y[0]])
    chord /= np.linalg.norm(chord) + 1e-12
    vec = np.stack([x - x[0], y - y[0]], 1)
    dist = np.abs(vec[:, 0] * chord[1] - vec[:, 1] * chord[0])
    smooth = np.convolve(y, np.ones(5) / 5, mode="valid")
    curv = np.diff(smooth, 2)
    elbows = {"chord": int(dist.argmax() + 1), "curvature": int(curv.argmax() + 3) if len(curv) else None}

    return {"eigenvalues": ev, "p": p, "effective_rank": r_trunc, "truncated": truncated,
            "effective_rank_bounds": (r_low, r_high), "elbows": elbows}


# Main functions: reconstruction ******************************************************************
def reconstruction_metrics(pca, X_eval: np.ndarray, dims: list[int], batch_size: int = 8192) -> tuple:
    """
    Project held-out vectors, reconstruct the original 1536-d vectors from the first d scores and
    compare them with the originals.

    Per-tile error is the mean squared error over the coordinates of the tile. NMSE is the total
    squared error divided by the total squared deviation of the held-out vectors from the training
    mean (so NMSE = 1 - R^2 with respect to the mean vector; 1 is the error of predicting the mean).

    * Arguments:
    pca: The fitted PCA
    X_eval: The [N, D] held-out vectors
    dims: The candidate dimensions
    batch_size: The number of tiles per batch

    Returns (summary DataFrame, per_tile) where per_tile maps d to {"error", "cosine"} arrays.
    """

    # STEP 1: Accumulate per-tile errors batch by batch (one projection shared by all dimensions)
    mean = pca.mean_.astype(np.float32)
    comps = pca.components_.astype(np.float32)
    err = {d: [] for d in dims}
    cos = {d: [] for d in dims}
    sq_err = {d: 0.0 for d in dims}
    deviation = 0.0
    for i in range(0, len(X_eval), batch_size):
        x = np.asarray(X_eval[i:i + batch_size], dtype=np.float32)
        z = (x - mean) @ comps.T
        deviation += float(((x - mean) ** 2).sum())
        xn = np.linalg.norm(x, axis=1)
        for d in dims:
            rec = z[:, :d] @ comps[:d] + mean
            e = ((x - rec) ** 2).sum(1)
            sq_err[d] += float(e.sum())
            err[d].append(e / x.shape[1])
            cos[d].append((x * rec).sum(1) / (xn * np.linalg.norm(rec, axis=1) + 1e-12))

    # STEP 2: Summary
    per_tile, rows = {}, []
    for d in dims:
        e, c = np.concatenate(err[d]), np.concatenate(cos[d])
        per_tile[d] = {"error": e, "cosine": c}
        rows.append({"dimensions": d, "mse": e.mean(), "nmse": sq_err[d] / deviation, "cosine": c.mean(),
                     "error_mean": e.mean(), "error_median": np.median(e), "error_std": e.std(),
                     "error_p95": np.percentile(e, 95), "error_p99": np.percentile(e, 99)})

    return pd.DataFrame(rows), per_tile


# Main functions: neighbourhoods ******************************************************************
def _topk(ref: torch.Tensor, queries: np.ndarray, k: int, chunk: int) -> np.ndarray:
    """Top-k cosine neighbours (indices into ref) of ref[queries], the query itself excluded."""
    out = []
    unit = _unit(ref)
    for i in range(0, len(queries), chunk):
        q = torch.from_numpy(queries[i:i + chunk]).to(ref.device)
        sims = unit[q] @ unit.T
        sims[torch.arange(len(q)), q] = -float("inf")
        out.append(sims.topk(k, dim=1).indices.cpu().numpy())
    return np.concatenate(out)


def neighbour_preservation(X_ref: np.ndarray, pca, dims: list[int], ref_slide: np.ndarray,
                           ref_label: np.ndarray | None = None, n_queries: int = 5000,
                           ks: tuple = (10, 20), n_pairs: int = 200000, seed: int = 42,
                           space: str = "scores", chunk: int = 512, cache_path: str | Path | None = None,
                           device: torch.device | None = None) -> dict:
    """
    Does PCA keep the cosine neighbourhood structure? For random query tiles of a fixed reference
    population, the nearest neighbours (cosine, exact search, the query excluded) in the original
    space are compared with those in each reduced space.

    Recall@k is the mean fraction of the k original neighbours found among the k reduced-space
    neighbours. The overlap is the number of shared neighbours among the top 10. Spearman compares
    the cosine similarity of random tile pairs before and after reduction. Neighbours in the
    original space are cached (cache_path) and reused for every dimension.

    * Arguments:
    X_ref: The [N, D] reference vectors (the same for every dimension)
    pca: The fitted PCA
    dims: The candidate dimensions
    ref_slide: [N] slide index of each reference tile (for the per-slide results)
    ref_label: [N] class of each reference tile (optional, for per-class results)
    n_queries: The number of query tiles
    ks: The values of k (the largest is searched once)
    n_pairs: The number of random tile pairs for the Spearman correlation
    seed: The random seed
    space: "scores" (cosine on the centred PCA scores, what a downstream model receives) or
        "reconstructed" (cosine on the reconstruction in the original space)
    chunk: The number of queries per matrix product
    cache_path: A file for the original-space neighbours (None = no cache)
    device: The torch device (default: utils.device.get_device())

    Returns a dict with summary (per dimension), per_slide, per_label (None without labels) and
    per_query (dimension -> array of shared-neighbour fractions at the largest k).
    """

    # STEP 1: Queries and random pairs, fixed for every dimension
    device = device or get_device()
    rng = np.random.default_rng(seed)
    n = len(X_ref)
    queries = np.sort(rng.choice(n, size=min(n_queries, n), replace=False))
    pairs = rng.integers(0, n, size=(n_pairs, 2))
    pairs = pairs[pairs[:, 0] != pairs[:, 1]]
    kmax = max(ks)

    # STEP 2: Neighbours in the original space (cached)
    key = {"n": n, "queries": int(queries.sum()), "kmax": kmax, "dim": int(X_ref.shape[1]), "seed": seed}
    cache = Path(cache_path) if cache_path else None
    if cache is not None and cache.exists() and torch.load(cache, weights_only=False)["key"] == key:
        original = torch.load(cache, weights_only=False)["neighbours"]
    else:
        original = _topk(torch.from_numpy(np.ascontiguousarray(X_ref, dtype=np.float32)).to(device), queries, kmax, chunk)
        if cache is not None:
            cache.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"key": key, "neighbours": original}, cache)
    unit_full = _unit(torch.from_numpy(np.ascontiguousarray(X_ref, dtype=np.float32)))
    sim_full = (unit_full[pairs[:, 0]] * unit_full[pairs[:, 1]]).sum(1).numpy()

    # STEP 3: Reduced spaces
    z = _scores(pca, X_ref)
    mean = pca.mean_.astype(np.float32)
    rows, slide_rows, label_rows, per_query = [], [], [], {}
    for d in dims:
        if space == "scores":
            red = z[:, :d]
        elif space == "reconstructed":
            red = z[:, :d] @ pca.components_[:d].astype(np.float32) + mean
        else:
            raise ValueError("space must be 'scores' or 'reconstructed'")
        red_t = torch.from_numpy(np.ascontiguousarray(red)).to(device)
        reduced = _topk(red_t, queries, kmax, chunk)

        # STEP 4: Recall and overlap
        hit = {k: (original[:, :k, None] == reduced[:, None, :k]).any(-1).sum(1) for k in ks}
        frac = {k: hit[k] / k for k in ks}
        ten = hit[10] if 10 in hit else (original[:, :10, None] == reduced[:, None, :10]).any(-1).sum(1)
        unit_red = _unit(torch.from_numpy(np.ascontiguousarray(red)))
        sim_red = (unit_red[pairs[:, 0]] * unit_red[pairs[:, 1]]).sum(1).numpy()
        row = {"dimensions": d, **{f"recall@{k}": frac[k].mean() for k in ks},
               "overlap_mean": ten.mean(), "overlap_median": float(np.median(ten)),
               "spearman": float(spearmanr(sim_full, sim_red).statistic)}
        rows.append(row)
        per_query[d] = frac[kmax]

        # STEP 5: Per slide and per class
        base = {"dimensions": d}
        df = pd.DataFrame({"slide": ref_slide[queries], **{f"recall@{k}": frac[k] for k in ks}})
        slide_rows.append(df.groupby("slide").mean().reset_index().assign(**base))
        if ref_label is not None:
            df["label"] = ref_label[queries]
            label_rows.append(df.drop(columns="slide").groupby("label").mean().reset_index().assign(**base))

    return {"summary": pd.DataFrame(rows), "per_slide": pd.concat(slide_rows, ignore_index=True),
            "per_label": pd.concat(label_rows, ignore_index=True) if label_rows else None,
            "per_query": per_query, "queries": queries}


# Main functions: comparison **********************************************************************
def final_comparison(variance: pd.DataFrame, recon: pd.DataFrame, neigh: pd.DataFrame) -> pd.DataFrame:
    """One table with the metrics of every candidate dimension."""
    df = variance[["dimensions", "explained_variance"]].merge(
        recon[["dimensions", "nmse", "cosine"]], on="dimensions").merge(
        neigh[["dimensions", "recall@10", "recall@20", "spearman"]], on="dimensions")
    return df.rename(columns={"dimensions": "Dimensions", "explained_variance": "Explained variance",
                              "nmse": "NMSE", "cosine": "Cosine similarity", "recall@10": "Recall@10",
                              "recall@20": "Recall@20", "spearman": "Spearman"})


def recommend_dimensions(comparison: pd.DataFrame, recall_targets: tuple = (0.80, 0.90, 0.95),
                         max_nmse: float | None = None) -> pd.DataFrame:
    """
    Candidate dimensionalities from neighbourhood preservation, not from explained variance alone:
    for each Recall@10 target, the smallest dimension that reaches it (and, if given, an NMSE not
    above max_nmse). Targets that no dimension reaches are reported as such.

    * Arguments:
    comparison: The output of final_comparison
    recall_targets: The Recall@10 levels
    max_nmse: An optional ceiling on NMSE

    Returns a DataFrame with one row per target and the metrics of the chosen dimension.
    """
    rows = []
    ok = comparison if max_nmse is None else comparison[comparison["NMSE"] <= max_nmse]
    for t in recall_targets:
        hit = ok[ok["Recall@10"] >= t].sort_values("Dimensions")
        if hit.empty:
            rows.append({"Recall@10 target": t, "Dimensions": np.nan})
        else:
            rows.append({"Recall@10 target": t, **hit.iloc[0].to_dict()})
    return pd.DataFrame(rows)


# Main functions: figures *************************************************************************
def _axis_dims(ax, dims: list[int], ylabel: str) -> None:
    ax.set_xscale("log", base=2)
    ax.set_xticks(dims)
    ax.set_xticklabels([str(d) for d in dims], rotation=45)
    ax.minorticks_off()
    ax.set_xlabel("PCA dimensions")
    ax.set_ylabel(ylabel)
    ax.grid(color=GRID, linewidth=0.6)


def plot_variance(pca, dims: list[int], thresholds: tuple = VARIANCE_THRESHOLDS, show: bool = True):
    """
    Scree plot and cumulative explained variance.

    * Arguments:
    pca: The fitted PCA
    dims: The candidate dimensions (vertical guides)
    thresholds: The cumulative variance levels (horizontal guides)
    show: Display the figure
    """
    ratio, cum = pca.explained_variance_ratio_, np.cumsum(pca.explained_variance_ratio_)
    notes = [
        (0, 1, "Scree plot: the fraction of the total variance of the training sample explained by each "
               "principal component, on a logarithmic axis. A steep initial decay followed by a plateau "
               "indicates that most of the variance lies in a low-dimensional subspace."),
        (1, 2, "Cumulative explained variance. Dashed vertical lines mark the candidate dimensions and "
               "dotted horizontal lines the variance thresholds. Explained variance measures the retained "
               "energy of the centred embeddings; it does not by itself ensure that similarity structure "
               "is preserved."),
    ]
    with plt.rc_context(STYLE):
        fig = plt.figure(figsize=(14, 5.6))
        plots = split_with_notes(fig, notes, [1, 1], 14, plot_height=3.8, note_height=1.8)
        a, b = plots.subplots(1, 2)
        a.semilogy(np.arange(1, len(ratio) + 1), ratio, color=SERIES[0])
        a.set_xlabel("Component"); a.set_ylabel("Explained variance ratio"); a.grid(color=GRID, linewidth=0.6)
        b.plot(np.arange(1, len(cum) + 1), cum, color=SERIES[0])
        for d in dims:
            b.axvline(d, color=SERIES[1], linestyle="--", linewidth=0.8)
        for t in thresholds:
            b.axhline(t, color=INK_2, linestyle=":", linewidth=0.8)
        b.set_xlabel("Components"); b.set_ylabel("Cumulative explained variance"); b.grid(color=GRID, linewidth=0.6)
        if show:
            plt.show()
    return fig


def plot_spectrum(stats: dict, show: bool = True):
    """
    Eigenvalue spectrum on a linear and a logarithmic scale, with elbows and effective rank.

    * Arguments:
    stats: The output of spectrum_stats
    show: Display the figure
    """
    ev = stats["eigenvalues"]
    x = np.arange(1, len(ev) + 1)
    low, high = stats["effective_rank_bounds"]
    trunc = " (truncated spectrum)" if stats["truncated"] else ""
    notes = [
        (0, 1, "Eigenvalues of the sample covariance matrix in decreasing order (linear scale)."),
        (1, 2, "The same spectrum on logarithmic axes. Vertical lines mark two elbow candidates: the point "
               "farthest from the chord joining the first and last eigenvalue, and the point of largest "
               "curvature of the smoothed log spectrum. The effective rank is the exponential of the "
               "spectral entropy of the normalised eigenvalues (Roy and Vetterli 2007); when only the "
               "leading components are fitted it is computed on the kept spectrum and is therefore a "
               "truncated estimate, bounded using the residual variance."),
    ]
    with plt.rc_context(STYLE):
        fig = plt.figure(figsize=(14, 5.8))
        plots = split_with_notes(fig, notes, [1, 1], 14, plot_height=3.8, note_height=2.0)
        a, b = plots.subplots(1, 2)
        a.plot(x, ev, color=SERIES[0]); a.set_xlabel("Component"); a.set_ylabel("Eigenvalue")
        b.loglog(x, ev, color=SERIES[0])
        for name, k, c in (("chord", stats["elbows"]["chord"], SERIES[1]), ("curvature", stats["elbows"]["curvature"], SERIES[2])):
            if k:
                b.axvline(k, color=c, linestyle="--", linewidth=1, label=f"elbow ({name}): {k}")
        b.set_xlabel("Component"); b.set_ylabel("Eigenvalue (log)"); b.legend(frameon=False)
        b.set_title(f"Effective rank{trunc}: {stats['effective_rank']:.1f}  [full-spectrum bounds {low:.1f}, {high:.1f}]",
                    fontsize=9, color=INK_2)
        for ax in (a, b):
            ax.grid(color=GRID, linewidth=0.6)
        if show:
            plt.show()
    return fig


def plot_reconstruction(summary: pd.DataFrame, per_tile: dict, show: bool = True):
    """
    Reconstruction error and cosine similarity against dimensionality, and the per-tile error distributions.

    * Arguments:
    summary: The first output of reconstruction_metrics
    per_tile: The second output of reconstruction_metrics
    show: Display the figure
    """
    dims = summary["dimensions"].tolist()
    notes = [
        (0, 1, "Normalised mean squared error of the held-out reconstruction (squared error divided by the "
               "squared deviation from the training mean) against the number of components."),
        (1, 2, "Mean cosine similarity between each held-out vector and its reconstruction."),
        (2, 3, "Per-tile mean squared reconstruction error (log scale); boxes show the quartiles and the "
               "whiskers the 5th and 95th percentiles. Dispersion across tiles shows whether the mean "
               "error is driven by a minority of poorly reconstructed tiles."),
    ]
    with plt.rc_context(STYLE):
        fig = plt.figure(figsize=(17, 5.6))
        plots = split_with_notes(fig, notes, [1, 1, 1.2], 17, plot_height=3.8, note_height=1.8)
        a, b, c = plots.subplots(1, 3, width_ratios=[1, 1, 1.2])
        a.plot(dims, summary["nmse"], "o-", color=SERIES[0]); _axis_dims(a, dims, "NMSE")
        b.plot(dims, summary["cosine"], "o-", color=SERIES[1]); _axis_dims(b, dims, "Cosine similarity")
        c.boxplot([per_tile[d]["error"] for d in dims], tick_labels=[str(d) for d in dims], whis=(5, 95),
                  showfliers=False, patch_artist=True, boxprops={"facecolor": "#cde2fb", "edgecolor": SERIES[0]},
                  medianprops={"color": SERIES[1]})
        c.set_yscale("log"); c.set_xlabel("PCA dimensions"); c.set_ylabel("Per-tile MSE"); c.grid(color=GRID, linewidth=0.6)
        if show:
            plt.show()
    return fig


def plot_neighbours(result: dict, show: bool = True):
    """
    Recall@k, Spearman correlation and per-slide recall against dimensionality.

    * Arguments:
    result: The output of neighbour_preservation
    show: Display the figure
    """
    s, ps = result["summary"], result["per_slide"]
    dims = s["dimensions"].tolist()
    ks = [c for c in s.columns if c.startswith("recall@")]
    notes = [
        (0, 1, "Recall@k: the mean fraction of the k nearest neighbours (cosine, original 1536-d space, query "
               "excluded) that are also among the k nearest neighbours in the reduced space, over a fixed "
               "set of query tiles and a fixed reference population."),
        (1, 2, "Spearman rank correlation between the cosine similarities of random tile pairs in the "
               "original and in the reduced space."),
        (2, 3, "Per-slide Recall@10 (mean over the queries of each slide); boxes show the quartiles across "
               "slides and whiskers the 5th and 95th percentiles."),
    ]
    with plt.rc_context(STYLE):
        fig = plt.figure(figsize=(17, 5.6))
        plots = split_with_notes(fig, notes, [1, 1, 1.2], 17, plot_height=3.8, note_height=1.8)
        a, b, c = plots.subplots(1, 3, width_ratios=[1, 1, 1.2])
        for k, col in zip(ks, SERIES):
            a.plot(dims, s[k], "o-", color=col, label=k)
        _axis_dims(a, dims, "Recall"); a.legend(frameon=False)
        b.plot(dims, s["spearman"], "o-", color=SERIES[2]); _axis_dims(b, dims, "Spearman")
        c.boxplot([ps.loc[ps["dimensions"] == d, "recall@10"] for d in dims], tick_labels=[str(d) for d in dims],
                  whis=(5, 95), showfliers=False, patch_artist=True,
                  boxprops={"facecolor": "#cde2fb", "edgecolor": SERIES[0]}, medianprops={"color": SERIES[1]})
        c.set_xlabel("PCA dimensions"); c.set_ylabel("Recall@10 per slide"); c.grid(color=GRID, linewidth=0.6)
        if show:
            plt.show()
    return fig


# Main functions: storage *************************************************************************
def save_pca_embeddings(features_dir: str | Path, out_root: str | Path, model_name: str, pca, dims: list[int],
                        table: pd.DataFrame, config: dict, whiten: bool = False, overwrite: bool = False,
                        batch_size: int = 50000) -> dict:
    """
    Apply the fitted PCA to every slide (one slide in memory at a time) and save the reduced vectors
    to out_root / pca_<d> / model_name / <slide>.pt, in the format of the original files.

    Each file holds cls_features [tiles, d], coordinates, model_encoder, tile_params (original), pca
    (d, whiten, model file) and slide, label, patient, set. Tile order and coordinates are those of the
    original file, which is only read. Nothing is written inside the original folder, and existing
    files are never overwritten unless overwrite=True. The PCA is never refitted.

    * Arguments:
    features_dir: The folder with the original embeddings
    out_root: The folder that receives the pca_<d> folders
    model_name: The model subfolder
    pca: The PCA fitted on the training sample
    dims: The dimensions to store
    table: The slide table (slide, patient, label, set); every slide of it with a file is processed
    config: Reproducibility information written to metadata.json (must be JSON-serialisable)
    whiten: Divide the scores by the square root of their explained variance
    overwrite: Allow replacing existing outputs
    batch_size: The number of tiles per projection batch

    Returns a dict with saved (slides), missing (slides without a file) and folders.
    """

    # STEP 1: Refuse to overwrite and never write into the source folder
    features_dir, out_root = Path(features_dir).resolve(), Path(out_root).resolve()
    if out_root == features_dir or features_dir in out_root.parents:
        raise ValueError("out_root must not be inside the original features folder")
    folders = {d: out_root / f"pca_{d}" / model_name for d in dims}
    existing = []
    for d in dims:
        found = list(folders[d].glob("*.pt")) if folders[d].exists() else []
        found += [out_root / f"pca_{d}" / "pca_model.pt", out_root / f"pca_{d}" / "metadata.json"]
        existing += [p for p in found if p.exists()]
    if existing and not overwrite:
        raise FileExistsError(f"{len(existing)} PCA output files already exist (e.g. {existing[0]}); "
                              "set overwrite=True to replace them.")
    for f in folders.values():
        f.mkdir(parents=True, exist_ok=True)

    # STEP 2: Model and metadata, one per dimension
    scale = (1 / np.sqrt(pca.explained_variance_)).astype(np.float32) if whiten else None
    for d in dims:
        torch.save({"mean": torch.from_numpy(pca.mean_), "components": torch.from_numpy(pca.components_[:d]),
                    "explained_variance": torch.from_numpy(pca.explained_variance_[:d]),
                    "explained_variance_ratio": torch.from_numpy(pca.explained_variance_ratio_[:d]),
                    "whiten": whiten, "n_components": d}, out_root / f"pca_{d}" / "pca_model.pt")

    # STEP 3: Slide by slide
    info = table.set_index("slide")
    saved, missing = [], []
    for s in info.index:
        src = _load_cls(features_dir, model_name, s)
        if src is None:
            missing.append(s)
            continue
        cls = src["cls_features"]
        z = np.concatenate([_scores(pca, cls[i:i + batch_size].float().numpy()) for i in range(0, len(cls), batch_size)])
        if whiten:
            z = z * scale
        meta = {"slide": s, "label": str(info.loc[s, "label"]), "patient": str(info.loc[s, "patient"]),
                "set": str(info.loc[s, "set"])}
        for d in dims:
            out = {"cls_features": torch.from_numpy(np.ascontiguousarray(z[:, :d])),
                   "coordinates": src["coordinates"].clone(), "model_encoder": src.get("model_encoder", model_name),
                   "pca": {"n_components": d, "whiten": whiten, "model_file": "../pca_model.pt"}, **meta}
            if "tile_params" in src:
                out["tile_params"] = src["tile_params"]
            torch.save(out, folders[d] / f"{s}.pt")
        saved.append(s)

    # STEP 4: Metadata with the configuration and the slide lists
    for d in dims:
        info_json = {**config, "n_components": d, "whiten": whiten, "n_slides_saved": len(saved),
                     "slides_missing": missing, "source": str(features_dir / model_name)}
        (out_root / f"pca_{d}" / "metadata.json").write_text(json.dumps(info_json, indent=1, default=str))

    return {"saved": saved, "missing": missing, "folders": folders}


def fit_fold_pca_embeddings(features_dir: str | Path, out_root: str | Path, model_name: str, folds: list[dict], dims: list[int],
                            max_per_slide: int, max_fit_tiles: int, seed: int, solver: str = "randomized", batch_size: int = 20000,
                            whiten: bool = False, overwrite: bool = False, stratify_by_label: bool = False, labels: dict[str, str] | None = None) -> pd.DataFrame:
    """
    One PCA per cross-validation fold, fitted on the fold's fit slides only, and the reduced vectors of the fold's slides.

    The held-out and the early-stopping slides of a fold, and the official Testing slides, never enter its PCA (a global PCA
    has seen them, which leaks into the cross-validation scores). The files of fold k are written to
    out_root/cv_fold_<k>/pca_<d>/model_name/<slide>.pt by save_pca_embeddings for the slides of fold["table7"] (fit, early stopping,
    held-out and the official Testing slides), with the 7-class label and the stored sets Training, Validation, HeldOut, Testing.
    The metadata.json of every folder carries the fold index and its fit / stop / held slide lists.

    * Arguments:
    features_dir: The folder with the original embeddings (level_1)
    out_root: The features root that receives the cv_fold_<k> folders (not inside features_dir)
    model_name: The model subfolder
    folds: The output of per_fold.cv_folds_for_features (each fold needs fit, stop, held and table7)
    dims: The PCA dimensions to store
    max_per_slide: The cap of fit tiles per slide
    max_fit_tiles: The cap of the fit sample
    seed: The seed of the sampling and of the PCA
    solver: "randomized" or "incremental" (fit_pca)
    batch_size: The batch size of the incremental solver
    whiten: Whether to whiten (never in this project)
    overwrite: Allow replacing existing outputs
    stratify_by_label: As sample_tiles
    labels: {slide: label} (only used when stratify_by_label)

    Returns a table with one row per fold: fit slides, fit tiles, and the explained variance ratio of the stored dimensions.
    """
    from utils.per_fold import fold_definition, fold_root

    labels = labels or {}
    rows = []
    for k, fold in enumerate(folds):
        # STEP 1: Sample the fit slides of this fold only and fit the PCA
        sample = sample_tiles(features_dir, model_name, fold["fit"], labels, max_per_slide, max_fit_tiles, seed, stratify_by_label)
        pca, usable = fit_pca(sample["X"], dims, solver, batch_size, seed)
        usable = [d for d in dims if d in usable]

        # STEP 2: Transform every slide of the fold and save
        config = {**fold_definition(fold, k), "seed": seed, "solver": solver, "dimensions": usable, "max_tiles_per_slide": max_per_slide,
                  "max_fit_tiles": max_fit_tiles, "fit_tiles": int(len(sample["X"])), "per_fold": True, "features_dir": str(features_dir),
                  "explained_variance_at_dims": {int(d): float(pca.explained_variance_ratio_[:d].sum()) for d in usable}}
        save_pca_embeddings(features_dir, fold_root(out_root, k), model_name, pca, usable, fold["table7"], config, whiten, overwrite)
        rows.append({"fold": k, "fit slides": len(fold["fit"]), "fit tiles": len(sample["X"]),
                     **{f"explained variance {d}": float(pca.explained_variance_ratio_[:d].sum()) for d in usable}})

    return pd.DataFrame(rows)
