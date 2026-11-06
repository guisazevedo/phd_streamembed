# IMPORTS *****************************************************************************************
# External libraries
from pathlib import Path
import numpy as np
import torch
from tqdm.auto import tqdm


# Main functions **********************************************************************************
def load_cls_features(features_dir: str | Path, model_name: str, max_slides: int | None = None) -> dict:
    """
    Load the CLS vectors and tile coordinates of every slide in features_dir / model_name.

    Files are memory-mapped, so patch tokens (if saved) are never read.

    * Arguments:
    features_dir: The folder that holds one subfolder per model
    model_name: The model subfolder with the per-slide .pt files
    max_slides: The maximum number of slides to load (None = all)

    Returns a dict with X [N, D] (all CLS vectors), coords [N, 2], slide_id [N] (index into names),
    names (slide file stems), counts (tiles per slide) and starts (first row of each slide in X).
    """

    # STEP 1: Find the files
    folder = Path(features_dir) / model_name
    files = sorted(folder.glob("*.pt"))[:max_slides]

    if not files:
        raise FileNotFoundError(f"No .pt files found in {folder.resolve()}")

    # STEP 2: Load every slide's CLS vectors and tile coordinates
    cls_list, coord_list, slide_id = [], [], []
    for k, f in enumerate(tqdm(files, desc="Loading CLS vectors")):
        d = torch.load(f, map_location="cpu", weights_only=True, mmap=True)
        cls_list.append(d["cls_features"].float())
        coord_list.append(d["coordinates"].numpy())
        slide_id.append(np.full(len(d["cls_features"]), k))

    slide_id = np.concatenate(slide_id)
    counts = np.bincount(slide_id, minlength=len(files))

    return {
        "X": torch.cat(cls_list),
        "coords": np.concatenate(coord_list),
        "slide_id": slide_id,
        "names": [f.stem for f in files],
        "counts": counts,
        "starts": np.concatenate([[0], np.cumsum(counts)[:-1]]),
    }


def fit_pca(X: torch.Tensor, max_fit: int = 30000, seed: int = 0, n_components: int = 50) -> dict:
    """
    Fit PCA on a random subset of the rows of X and project all rows.

    * Arguments:
    X: The [N, D] matrix to project
    max_fit: The number of random rows used to fit the PCA
    seed: The random seed
    n_components: The number of principal components to compute

    Returns a dict with Z [N, q] (projection), explained [q] (% variance per component),
    fit_idx (the rows used for fitting, a tensor) and generator (the seeded torch generator).
    """

    g = torch.Generator().manual_seed(seed)
    fit_idx = torch.randperm(len(X), generator=g)[: min(len(X), max_fit)]
    mu = X[fit_idx].mean(0, keepdim=True)
    q = min(n_components, X.shape[1], len(fit_idx) - 1)
    _, S_, V = torch.pca_lowrank(X[fit_idx] - mu, q=q, center=False, niter=4)
    total_var = ((X[fit_idx] - mu) ** 2).sum() / (len(fit_idx) - 1)
    explained = (S_ ** 2 / (len(fit_idx) - 1) / total_var).numpy() * 100        # % variance per PC

    return {"Z": ((X - mu) @ V).numpy(), "explained": explained, "fit_idx": fit_idx, "generator": g}


def kmeans(z, k: int, generator: torch.Generator, iters: int = 40) -> torch.Tensor:
    """Lloyd's k-means with a k-means++ start; returns the [k, d] cluster centres."""
    z = torch.as_tensor(z)
    centers = z[torch.randint(len(z), (1,), generator=generator)]
    for _ in range(k - 1):
        dist = torch.cdist(z, centers).min(1).values ** 2
        centers = torch.cat([centers, z[torch.multinomial(dist / dist.sum(), 1, generator=generator)]])
    for _ in range(iters):
        labels = torch.cdist(z, centers).argmin(1)
        centers = torch.stack([z[labels == c].mean(0) if (labels == c).any() else centers[c] for c in range(k)])
    return centers


def assign(z, centers: torch.Tensor) -> np.ndarray:
    """Label every row of z with the index of its nearest centre."""
    return torch.cdist(torch.as_tensor(z), centers).argmin(1).numpy()


def silhouette(z, labels: np.ndarray) -> np.ndarray:
    """
    Silhouette value of every row of z (in [-1, 1]; higher = better separated from other groups).

    Uses the full pairwise distance matrix, so pass a subsample of a few thousand rows.
    """
    z = torch.as_tensor(z, dtype=torch.float32)
    _, own = np.unique(labels, return_inverse=True)
    own = torch.as_tensor(own)
    onehot = torch.nn.functional.one_hot(own).float()                 # [n, groups]
    size = onehot.sum(0)
    sum_to = torch.cdist(z, z) @ onehot                               # summed distance to each group
    rows = torch.arange(len(z))
    a = sum_to[rows, own] / (size[own] - 1).clamp(min=1)              # mean distance inside the own group
    other = sum_to / size
    other[rows, own] = float("inf")
    b = other.min(1).values if onehot.shape[1] > 1 else torch.zeros(len(z))
    s = (b - a) / torch.maximum(a, b).clamp(min=1e-12)
    s[size[own] <= 1] = 0
    return s.numpy()


def davies_bouldin(z, labels: np.ndarray) -> float:
    """Davies-Bouldin index (lower = more compact, better separated clusters)."""
    z = torch.as_tensor(z, dtype=torch.float32)
    lab = torch.as_tensor(labels)
    groups = lab.unique()
    cen = torch.stack([z[lab == c].mean(0) for c in groups])
    spread = torch.stack([(z[lab == c] - cen[i]).norm(dim=1).mean() for i, c in enumerate(groups)])
    sep = torch.cdist(cen, cen)
    ratio = (spread[:, None] + spread[None]) / sep.clamp(min=1e-12)
    ratio.fill_diagonal_(0)
    return float(ratio.max(1).values.mean())


def contingency(a: np.ndarray, b: np.ndarray, na: int, nb: int) -> np.ndarray:
    """Count table [na, nb] of how often label a[i] co-occurs with label b[i]."""
    table = np.zeros((na, nb))
    np.add.at(table, (a, b), 1)
    return table


def adjusted_rand_index(a: np.ndarray, b: np.ndarray) -> float:
    """Adjusted Rand index between two labelings (1 = identical grouping, about 0 = chance level)."""
    table = contingency(a, b, a.max() + 1, b.max() + 1)
    comb = lambda x: x * (x - 1) / 2
    sum_cells, sum_rows, sum_cols = comb(table).sum(), comb(table.sum(1)).sum(), comb(table.sum(0)).sum()
    expected = sum_rows * sum_cols / comb(table.sum())
    maximum = (sum_rows + sum_cols) / 2
    return float((sum_cells - expected) / (maximum - expected)) if maximum != expected else 1.0


def normalized_mutual_info(a: np.ndarray, b: np.ndarray) -> float:
    """Normalized mutual information between two labelings (0 = independent, 1 = one determines the other)."""
    table = contingency(a, b, a.max() + 1, b.max() + 1)
    p = table / table.sum()
    pa, pb = p.sum(1, keepdims=True), p.sum(0, keepdims=True)
    nz = p > 0
    mi = (p[nz] * np.log(p[nz] / (pa @ pb)[nz])).sum()
    ha, hb = -(pa[pa > 0] * np.log(pa[pa > 0])).sum(), -(pb[pb > 0] * np.log(pb[pb > 0])).sum()
    return float(mi / ((ha + hb) / 2)) if (ha + hb) > 0 else 0.0
