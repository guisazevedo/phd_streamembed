# IMPORTS *****************************************************************************************
# External libraries
from pathlib import Path
import copy
import hashlib
import json
import platform
import time
import matplotlib
import numpy as np
import pandas as pd
import scipy
import sklearn
import torch
import matplotlib.pyplot as plt
from sklearn.cluster import MiniBatchKMeans
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold

# Internal imports ********************************************************************************
from utils.cls_overview import GRID, INK_2, SERIES, STYLE, cluster_colors
from utils.figure_captions import split_with_notes
from utils.pca_analysis import sample_tiles

# Constants ***************************************************************************************
SYNTHETIC_SET = "Synthetic"
LABEL_COLUMNS = ["WSI Filename", "Patient Id", "WSI label", "Set", "method", "pca_dim", "k", "n_tiles",
                 "n_templates", "max_template_weight"]
_CHUNK = 50_000


# Helpers *****************************************************************************************
def _rng(*key: int) -> np.random.Generator:
    """Deterministic generator for a tuple of integers (seed, dimension, k, class index, ...)."""
    return np.random.default_rng(list(key))


def _load(path: Path) -> dict:
    """Memory-map one slide file; patch tokens are never present in PCA files."""
    return torch.load(path, map_location="cpu", weights_only=True, mmap=True)


def _slide_path(pca_dir: Path, model_name: str, slide: str) -> Path:
    return Path(pca_dir) / model_name / f"{slide}.pt"


def _apportion(total: int, proportions: np.ndarray) -> np.ndarray:
    """Largest-remainder rounding: integer counts proportional to `proportions` that sum to `total`."""
    raw = proportions * total
    counts = np.floor(raw).astype(np.int64)
    left = int(total - counts.sum())
    if left > 0:
        counts[np.argsort(-(raw - counts), kind="stable")[:left]] += 1
    return counts


def _within(path_a: Path, path_b: Path) -> bool:
    a, b = Path(path_a).resolve(), Path(path_b).resolve()
    return a == b or b in a.parents


# Main functions: training data *******************************************************************
def load_training_slides(table: pd.DataFrame, pca_dir: str | Path, model_name: str, expected_dim: int) -> dict:
    """
    Select the official Training slides and validate their PCA files (no tile is copied into memory).

    Every Training slide of the label table must have a file with `expected_dim` columns, finite
    values, the label of the table and, when stored, the set "Training". When the folder holds a
    metadata.json written by 01_2, its fit_slides must be the same slides. Any mismatch stops the run,
    so Validation or Testing slides can never enter the statistics silently.

    * Arguments:
    table: The slide table (columns slide, patient, label, set) of build_slide_table
    pca_dir: The folder pca_<d> written by 01_2
    model_name: The model subfolder with the per-slide .pt files
    expected_dim: The number of PCA components of the folder

    Returns a dict with train (DataFrame: slide, patient, label, n_tiles), classes (ordered labels).
    """

    # STEP 1: The Training slides of the table
    pca_dir = Path(pca_dir)
    if "set" not in table or not (table["set"].astype(str) == "Training").any():
        raise ValueError("The slide table has no Training slides: the split cannot be identified.")
    train = table.loc[table["set"].astype(str) == "Training", ["slide", "patient", "label"]].copy()
    train["label"] = train["label"].astype(str)
    classes = [c for c in table["label"].cat.categories if c in set(train["label"])] \
        if hasattr(table["label"], "cat") else sorted(set(train["label"]))

    # STEP 2: Agree with the slides the PCA was fitted on, when that is recorded
    meta_file = pca_dir / "metadata.json"
    if meta_file.exists():
        fitted = set(json.loads(meta_file.read_text()).get("fit_slides", []))
        if fitted and not fitted <= set(train["slide"]):
            raise ValueError(f"{meta_file}: fit_slides holds slides that are not Training slides in the table.")

    # STEP 3: Validate every file
    missing, counts = [], []
    for s, lab in zip(train["slide"], train["label"]):
        path = _slide_path(pca_dir, model_name, s)
        if not path.exists():
            missing.append(s)
            counts.append(0)
            continue
        d = _load(path)
        x = d["cls_features"]
        if x.ndim != 2 or x.shape[1] != expected_dim:
            raise ValueError(f"{path}: expected [tiles, {expected_dim}], found {tuple(x.shape)}")
        if "label" in d and str(d["label"]) != lab:
            raise ValueError(f"{path}: label {d['label']!r} differs from the table ({lab!r})")
        if "set" in d and str(d["set"]) != "Training":
            raise ValueError(f"{path}: stored set is {d['set']!r}, not Training")
        counts.append(int(x.shape[0]))
    if missing:
        raise FileNotFoundError(f"{len(missing)} Training slides have no file in {pca_dir / model_name}: {missing[:5]}")
    train["n_tiles"] = counts
    if (train["n_tiles"] == 0).any():
        raise ValueError("Some Training slides have no tiles.")
    return {"train": train.reset_index(drop=True), "classes": classes}


def sample_class_tiles(pca_dir: str | Path, model_name: str, train: pd.DataFrame, label: str,
                       max_per_slide: int, max_total: int, seed: int, with_slides: bool = False) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """
    Slide-balanced sample of the Training tiles of one class (sample_tiles of 01_2, one class only).

    * Arguments:
    pca_dir: The folder pca_<d>
    model_name: The model subfolder
    train: The Training slides (columns slide, label)
    label: The class
    max_per_slide: The maximum number of tiles taken from one slide
    max_total: The maximum number of tiles of the class
    seed: The random seed
    with_slides: Also return the slide index of every tile (for grouped splits)

    Returns the [N, D] float32 sample, or (sample, slide index [N]) with with_slides.
    """
    slides = train.loc[train["label"] == label, "slide"].tolist()
    sample = sample_tiles(pca_dir, model_name, slides, {s: label for s in slides}, max_per_slide, max_total, seed)
    return (sample["X"], np.asarray(sample["slide"])) if with_slides else sample["X"]


# Main functions: clustering and composition ******************************************************
def fit_class_clusters(X: np.ndarray, k: int, batch_size: int, seed: int, n_init: int = 3,
                       max_iter: int = 100) -> MiniBatchKMeans:
    """
    Fit MiniBatchKMeans on the sample of one class; k is reduced when the sample has fewer tiles.

    * Arguments:
    X: The [N, D] sample of one class
    k: The configured number of clusters
    batch_size: The mini-batch size
    seed: The random seed
    n_init: The number of initialisations (the best inertia is kept)
    max_iter: The maximum number of passes

    Returns the fitted model.
    """
    k_eff = int(min(k, len(X)))
    model = MiniBatchKMeans(n_clusters=k_eff, batch_size=batch_size, n_init=n_init, max_iter=max_iter,
                            random_state=int(seed), reassignment_ratio=0.01)
    return model.fit(np.ascontiguousarray(X, dtype=np.float32))


def fit_shared_clusters(samples: dict[str, np.ndarray], k: int, batch_size: int, seed: int, n_init: int = 3,
                        max_iter: int = 100) -> MiniBatchKMeans:
    """
    One MiniBatchKMeans (a shared codebook) on the tiles of all classes together.

    The classes contribute the same number of tiles (the size of the smallest sample, drawn at random), so a large
    class does not decide the centres. The model is then used for every class: the composition of a class is the
    share of its tiles in each cluster, and a cluster that a class hardly uses falls back to the pooled variance.

    * Arguments:
    samples: Class label -> [N, D] sample (slide-balanced, from sample_class_tiles)
    k: The configured number of clusters
    batch_size: The mini-batch size
    seed: The random seed
    n_init: The number of initialisations
    max_iter: The maximum number of passes

    Returns the fitted model.
    """
    rng = np.random.default_rng(int(seed))
    size = min(len(X) for X in samples.values())
    pooled = np.concatenate([X[rng.choice(len(X), size, replace=False)] for X in samples.values()])

    return fit_class_clusters(pooled, k, batch_size, seed, n_init, max_iter)


def collect_class_statistics(model: MiniBatchKMeans, pca_dir: str | Path, model_name: str, slides: list[str],
                             full_covariance: bool = False) -> dict:
    """
    One streaming pass over the Training slides of a class: cluster composition and Gaussian sums.

    Each slide is read once, every tile is assigned to its nearest centre, and per cluster the tile
    count, the sum and the sum of squares (and the cross-product matrix for a full covariance) are
    accumulated in float64. All Training tiles of the class contribute, without holding them together.

    * Arguments:
    model: The fitted class-specific MiniBatchKMeans
    pca_dir: The folder pca_<d>
    model_name: The model subfolder
    slides: The Training slides of the class
    full_covariance: Also accumulate the cross-product matrices

    Returns a dict with counts [S, K] (tiles per slide and cluster), n [K], sum [K, D], sumsq [K, D],
    xtx [K, D, D] or None, and slide_mean [S, K, D] (mean of the tiles of each slide in each cluster;
    zero where the slide has no tile in the cluster, see counts).
    """

    # STEP 1: Allocate the accumulators
    k, dim = model.cluster_centers_.shape
    counts = np.zeros((len(slides), k), dtype=np.int64)
    total = np.zeros((k, dim)); total_sq = np.zeros((k, dim))
    slide_sum = np.zeros((len(slides), k, dim))
    xtx = np.zeros((k, dim, dim)) if full_covariance else None

    # STEP 2: Stream the slides in chunks
    for i, s in enumerate(slides):
        x_all = _load(_slide_path(pca_dir, model_name, s))["cls_features"]
        for start in range(0, len(x_all), _CHUNK):
            x = x_all[start:start + _CHUNK].float().numpy()
            if not np.isfinite(x).all():
                raise ValueError(f"{s}: NaN or infinite values in the embeddings")
            lab = model.predict(x)
            counts[i] += np.bincount(lab, minlength=k)
            for c in np.unique(lab):
                xc = x[lab == c].astype(np.float64)
                total[c] += xc.sum(0)
                total_sq[c] += (xc * xc).sum(0)
                slide_sum[i, c] += xc.sum(0)
                if xtx is not None:
                    xtx[c] += xc.T @ xc

    # STEP 3: Mean of each slide in each cluster
    slide_mean = slide_sum / np.maximum(counts, 1)[:, :, None]
    return {"counts": counts, "n": counts.sum(0), "sum": total, "sumsq": total_sq, "xtx": xtx, "slide_mean": slide_mean}


def composition_table(stats: dict, slides: list[str]) -> pd.DataFrame:
    """Per Training slide: total tiles and tiles / proportion per cluster (the templates of the generator)."""
    counts = stats["counts"]
    props = {f"prop_{c}": counts[:, c] / counts.sum(1) for c in range(counts.shape[1])}
    return pd.DataFrame({"slide": slides, "n_tiles": counts.sum(1), **props})


# Main functions: generators **********************************************************************
class GeneratorBase:
    """
    Interface of a class-specific generator of synthetic tile vectors.

    A new method (GMM, SMOTE, VAE, ...) implements fit (from the statistics of
    collect_class_statistics, the fitted clustering and the sample) and sample (n vectors for one
    cluster); clustering, composition and saving are independent of it.
    """
    name = "base"

    def fit(self, stats: dict, centers: np.ndarray, X: np.ndarray | None = None) -> "GeneratorBase":
        raise NotImplementedError

    def start_bag(self, rng: np.random.Generator) -> np.ndarray | None:
        """
        Per-bag state drawn once before the clusters of a bag are sampled (for example one offset per cluster).

        The default has no per-bag state: it returns None and does not use the random generator, so a generator
        without the hook produces the same bags as before. The value is passed to `sample` as `offsets`.
        """
        return None

    def sample(self, cluster: int, n: int, rng: np.random.Generator, offsets: np.ndarray | None = None) -> np.ndarray:
        raise NotImplementedError

    def parameters(self) -> dict:
        """Arrays and settings worth saving for reproducibility."""
        raise NotImplementedError


class GaussianGenerator(GeneratorBase):
    """
    x ~ N(mu_k, Sigma_k + lambda I) for each cluster k.

    With covariance "diag" only the variances are stored and a vector is mu + sqrt(var + lambda) * eps.
    With "full" the Cholesky factor of Sigma_k + lambda I is used (jitter is added until it exists).
    A cluster with fewer than min_tiles tiles has no reliable covariance: it uses its mean when it
    has at least one tile (the centre otherwise) and the pooled within-cluster variance of the class.

    * Arguments:
    covariance: "diag" (default) or "full"
    reg: The regularisation lambda added to the variances / the diagonal
    min_tiles: The minimum number of tiles for a cluster-specific covariance
    """
    name = "gaussian"

    def __init__(self, covariance: str = "diag", reg: float = 1e-3, min_tiles: int = 10):
        if covariance not in ("diag", "full"):
            raise ValueError("covariance must be 'diag' or 'full'")
        self.covariance, self.reg, self.min_tiles = covariance, float(reg), int(min_tiles)

    def fit(self, stats: dict, centers: np.ndarray, X: np.ndarray | None = None) -> "GaussianGenerator":
        # STEP 1: Means and (unbiased) variances of the clusters
        n = stats["n"].astype(np.float64)
        k, dim = centers.shape
        safe = np.maximum(n, 1)[:, None]
        mean = np.where(n[:, None] > 0, stats["sum"] / safe, centers)
        scatter = stats["sumsq"] - n[:, None] * mean ** 2
        var = np.maximum(scatter / np.maximum(n - 1, 1)[:, None], 0)

        # STEP 2: Pooled within-cluster variance for the small clusters
        pooled = np.maximum(scatter.sum(0) / max(n.sum() - k, 1), 0)
        small = n < self.min_tiles
        var[small] = pooled
        self.small = small

        # STEP 3: Full covariances (small clusters keep the pooled diagonal)
        self.mean, self.var, self.chol = mean, var, None
        if self.covariance == "full":
            if stats["xtx"] is None:
                raise ValueError("full covariance needs the cross-product matrices")
            self.chol = np.zeros((k, dim, dim))
            for c in range(k):
                if small[c]:
                    cov = np.diag(pooled)
                else:
                    cov = (stats["xtx"][c] - n[c] * np.outer(mean[c], mean[c])) / (n[c] - 1)
                    cov = 0.5 * (cov + cov.T)
                self.chol[c] = _cholesky(cov, self.reg)
        return self

    def sample(self, cluster: int, n: int, rng: np.random.Generator, offsets: np.ndarray | None = None) -> np.ndarray:
        # The offsets are ignored: this generator has no between-slide component
        dim = self.mean.shape[1]
        out = np.empty((n, dim), dtype=np.float32)
        for start in range(0, n, _CHUNK):
            m = min(_CHUNK, n - start)
            eps = rng.standard_normal((m, dim))
            if self.covariance == "diag":
                x = self.mean[cluster] + eps * np.sqrt(self.var[cluster] + self.reg)
            else:
                x = self.mean[cluster] + eps @ self.chol[cluster].T
            out[start:start + m] = x
        return out

    @classmethod
    def from_parameters(cls, p: dict) -> "GaussianGenerator":
        """Rebuild a fitted generator from `parameters()` (the arrays are float32 on disk, so samples agree to float32 precision)."""
        gen = cls(p["covariance"], p["reg"], p["min_tiles"])
        gen.mean, gen.var = _to_numpy(p["mean"]), _to_numpy(p["variance"])
        gen.small, gen.chol = p["small_cluster"].numpy(), (_to_numpy(p["cholesky"]) if "cholesky" in p else None)
        return gen

    def parameters(self) -> dict:
        p = {"covariance": self.covariance, "reg": self.reg, "min_tiles": self.min_tiles,
             "mean": torch.from_numpy(self.mean.astype(np.float32)),
             "variance": torch.from_numpy(self.var.astype(np.float32)),
             "small_cluster": torch.from_numpy(self.small)}
        if self.chol is not None:
            p["cholesky"] = torch.from_numpy(self.chol.astype(np.float32))
        return p


def _total_variance(stats: dict, min_tiles: int) -> tuple[np.ndarray, np.ndarray]:
    """Per-cluster variance of all tiles [K, D]; clusters below min_tiles get the pooled within-cluster variance. Also returns the small flags."""
    n = stats["n"].astype(np.float64)
    k = len(n)
    mean = stats["sum"] / np.maximum(n, 1)[:, None]
    scatter = stats["sumsq"] - n[:, None] * mean ** 2
    var = np.maximum(scatter / np.maximum(n - 1, 1)[:, None], 0)
    pooled = np.maximum(scatter.sum(0) / max(n.sum() - k, 1), 0)
    small = n < min_tiles
    var[small] = pooled

    return var, small


def between_slide_variance(slide_mean: np.ndarray, counts: np.ndarray, total_var: np.ndarray,
                           min_slide_tiles: int = 5) -> tuple[np.ndarray, np.ndarray]:
    """
    Variance of the slide means of each cluster (the part of the variance shared by all tiles of a slide).

    Only slides with at least `min_slide_tiles` tiles in the cluster are used. The slide means are weighted by their
    tile counts. A slide mean also carries sampling noise with variance W / n_s (W: within-slide variance, n_s: tiles),
    which contributes S_valid * W / N to the weighted variance; this term is subtracted with the total variance as the
    upper bound of W (method of moments), and the result is floored at 0. The estimate is therefore conservative,
    and it is 0 for a cluster seen in fewer than two slides.

    * Arguments:
    slide_mean: The [S, K, D] mean of each slide in each cluster
    counts: The [S, K] tiles of each slide in each cluster
    total_var: The [K, D] variance of all tiles of each cluster
    min_slide_tiles: The minimum tiles of a slide in a cluster for its mean to be used

    Returns the between-slide variance [K, D] and the number of slides used per cluster [K].
    """

    # STEP 1: Slides that count and their weights
    valid = counts >= min_slide_tiles
    w = np.where(valid, counts, 0).astype(np.float64)
    n_slides = valid.sum(0)
    total = np.maximum(w.sum(0), 1)

    # STEP 2: Weighted variance of the slide means, minus the sampling-noise term
    centre = np.einsum("sk,skd->kd", w, slide_mean) / total[:, None]
    raw = np.einsum("sk,skd->kd", w, (slide_mean - centre[None]) ** 2) / total[:, None]
    between = np.maximum(raw - (n_slides / total)[:, None] * total_var, 0)
    between[n_slides < 2] = 0

    return between, n_slides


class HierarchicalGenerator(GeneratorBase):
    """
    Tiles of a cluster = cluster mean + one offset shared by the whole bag + within-slide noise.

    The variance of each cluster is split into a between-slide part B (the variance of the slide means,
    `between_slide_variance`) and a within-slide part. A bag draws one offset per cluster from N(0, s^2 B)
    (`start_bag`, s = offset_scale) that shifts every tile of that cluster in the bag; the tiles themselves get
    only the within-slide variance (total - s^2 B, floored at 1% of the total), so the marginal variance of the
    tiles is that of the real tiles for every s, and s = 0 (no offsets) with a diagonal covariance is the
    GaussianGenerator.

    Clusters seen in fewer than `min_slides` slides have an unreliable B: it is shrunk linearly toward the class-pooled
    between-slide share of the variance times the total variance of the cluster (weight min(1, slides / min_slides) on
    the cluster's own estimate). Clusters below `min_tiles` tiles use the pooled variance, as in GaussianGenerator.

    Covariance of the within-slide noise: "diag"; "full" (Cholesky of the scatter covariance, in which the diagonal is
    reduced by s^2 B); or "lowrank" (the top `rank` eigenvectors of that matrix as factors F and a diagonal psi that
    absorbs the rest, so the variances are kept). For "full" and "lowrank" the covariance of a cluster is shrunk toward
    the pooled within-cluster covariance of the class by `shrinkage` (0 = none). Both need the cross-product matrices
    (collect_class_statistics with full_covariance = True). The offsets are diagonal: no correlation between
    coordinates, and independent between clusters.

    * Arguments:
    covariance: "diag" (default), "full" or "lowrank"
    rank: The number of factors of "lowrank"
    offset_scale: s in [0, 1]; 0 = no offsets
    min_slides: Slides below which the between-slide variance of a cluster is shrunk toward the pooled value
    min_slide_tiles: The minimum tiles of a slide in a cluster for its mean to count
    shrinkage: Weight of the pooled covariance in [0, 1] ("full" and "lowrank")
    reg: The regularisation lambda added to the variances / the diagonal
    min_tiles: The minimum number of tiles for a cluster-specific variance
    """
    name = "hier_lowrank"
    _FLOOR = 0.01

    def __init__(self, covariance: str = "diag", rank: int = 16, offset_scale: float = 1.0, min_slides: int = 5,
                 min_slide_tiles: int = 5, shrinkage: float = 0.0, reg: float = 1e-3, min_tiles: int = 10):
        if covariance not in ("diag", "full", "lowrank"):
            raise ValueError("covariance must be 'diag', 'full' or 'lowrank'")
        if not 0.0 <= offset_scale <= 1.0:
            raise ValueError("offset_scale must be in [0, 1]")
        if not 0.0 <= shrinkage <= 1.0:
            raise ValueError("shrinkage must be in [0, 1]")
        if rank < 1:
            raise ValueError("rank must be at least 1")
        self.covariance, self.rank, self.offset_scale = covariance, int(rank), float(offset_scale)
        self.min_slides, self.min_slide_tiles, self.shrinkage = int(min_slides), int(min_slide_tiles), float(shrinkage)
        self.reg, self.min_tiles = float(reg), int(min_tiles)

    def fit(self, stats: dict, centers: np.ndarray, X: np.ndarray | None = None) -> "HierarchicalGenerator":
        # STEP 1: Means and total variances (small clusters: pooled variance)
        n = stats["n"].astype(np.float64)
        k, dim = centers.shape
        mean = np.where(n[:, None] > 0, stats["sum"] / np.maximum(n, 1)[:, None], centers)
        total_var, small = _total_variance(stats, self.min_tiles)
        self.small, self.mean, self.total_var = small, mean, total_var

        # STEP 2: Between-slide variance, shrunk toward the pooled share for clusters seen in few slides
        between, n_slides = between_slide_variance(stats["slide_mean"], stats["counts"], total_var, self.min_slide_tiles)
        reliable = (n_slides >= self.min_slides) & ~small
        if not reliable.any():
            reliable = (n_slides >= 2) & ~small
        share = (n[reliable, None] * between[reliable]).sum(0) / np.maximum((n[reliable, None] * total_var[reliable]).sum(0), 1e-30) \
            if reliable.any() else np.zeros(dim)
        weight = np.clip(n_slides / max(self.min_slides, 1), 0, 1)[:, None]
        between = np.minimum(weight * between + (1 - weight) * share * total_var, total_var)
        self.between, self.n_slides = between, n_slides
        self.offset_var = self.offset_scale ** 2 * between

        # STEP 3: Within-slide variance of the tiles (the offsets carry the rest)
        self.var = np.maximum(total_var - self.offset_var, self._FLOOR * total_var)
        self.chol, self.factors, self.psi = None, None, None
        if self.covariance == "diag":
            return self

        # STEP 4: Covariances for "full" and "lowrank": scatter, shrinkage toward the pooled covariance, diagonal reduced by the offsets
        if stats["xtx"] is None:
            raise ValueError(f"{self.covariance} covariance needs the cross-product matrices (full_covariance=True)")
        scatter = stats["xtx"] - n[:, None, None] * mean[:, :, None] * mean[:, None, :]
        pooled = scatter.sum(0) / max(n.sum() - k, 1)
        covs = np.empty((k, dim, dim))
        for c in range(k):
            if small[c]:
                covs[c] = np.diag(self.var[c])
                continue
            cov = scatter[c] / (n[c] - 1)
            cov = 0.5 * (cov + cov.T)
            cov = (1 - self.shrinkage) * cov + self.shrinkage * pooled
            diag = np.diag(cov)
            cov[np.diag_indices(dim)] = np.maximum(diag - self.offset_var[c], self._FLOOR * diag)
            covs[c] = cov

        # STEP 5: Cholesky factor, or low-rank factors plus a diagonal that absorbs the rest
        if self.covariance == "full":
            self.chol = np.stack([self._factor(covs[c]) for c in range(k)])
            return self
        r = min(self.rank, dim)
        self.factors, self.psi = np.zeros((k, dim, r)), np.zeros((k, dim))
        for c in range(k):
            if small[c]:
                self.psi[c] = self.var[c]
                continue
            values, vectors = np.linalg.eigh(covs[c])
            values, vectors = np.maximum(values[::-1][:r], 0), vectors[:, ::-1][:, :r]
            self.factors[c] = vectors * np.sqrt(values)
            diag = np.diag(covs[c])
            self.psi[c] = np.maximum(diag - (self.factors[c] ** 2).sum(1), self._FLOOR * diag)
        return self

    def _factor(self, cov: np.ndarray) -> np.ndarray:
        """Cholesky factor of cov + reg I; when the covariance minus the offsets is not positive definite, its eigenvalues are first raised to 1% of the mean variance."""
        try:
            return _cholesky(cov, self.reg)
        except np.linalg.LinAlgError:
            values, vectors = np.linalg.eigh(cov)
            values = np.maximum(values, self._FLOOR * float(np.mean(np.diag(cov))))
            return _cholesky((vectors * values) @ vectors.T, self.reg)

    def start_bag(self, rng: np.random.Generator) -> np.ndarray | None:
        """One offset per cluster, N(0, offset_scale^2 B); None (and no random draw) when offset_scale is 0."""
        if self.offset_scale == 0.0:
            return None
        return rng.standard_normal(self.mean.shape) * np.sqrt(self.offset_var)

    def sample(self, cluster: int, n: int, rng: np.random.Generator, offsets: np.ndarray | None = None) -> np.ndarray:
        dim = self.mean.shape[1]
        centre = self.mean[cluster] if offsets is None else self.mean[cluster] + offsets[cluster]
        out = np.empty((n, dim), dtype=np.float32)
        for start in range(0, n, _CHUNK):
            m = min(_CHUNK, n - start)
            if self.covariance == "lowrank":
                x = centre + rng.standard_normal((m, self.factors.shape[2])) @ self.factors[cluster].T \
                    + rng.standard_normal((m, dim)) * np.sqrt(self.psi[cluster] + self.reg)
            else:
                eps = rng.standard_normal((m, dim))
                if self.covariance == "diag":
                    x = centre + eps * np.sqrt(self.var[cluster] + self.reg)
                else:
                    x = centre + eps @ self.chol[cluster].T
            out[start:start + m] = x
        return out

    @classmethod
    def from_parameters(cls, p: dict) -> "HierarchicalGenerator":
        """Rebuild a fitted generator from `parameters()` (float32 on disk; the offset variance is offset_scale^2 times the between-slide variance)."""
        gen = cls(p["covariance"], p["rank"], p["offset_scale"], p["min_slides"], p["min_slide_tiles"], p["shrinkage"], p["reg"], p["min_tiles"])
        gen.mean, gen.total_var, gen.var = _to_numpy(p["mean"]), _to_numpy(p["total_variance"]), _to_numpy(p["variance"])
        gen.between, gen.n_slides, gen.small = _to_numpy(p["between_variance"]), p["n_slides"].numpy(), p["small_cluster"].numpy()
        gen.offset_var = gen.offset_scale ** 2 * gen.between
        gen.chol = _to_numpy(p["cholesky"]) if "cholesky" in p else None
        gen.factors, gen.psi = (_to_numpy(p["factors"]), _to_numpy(p["psi"])) if "factors" in p else (None, None)
        return gen

    def parameters(self) -> dict:
        p = {"covariance": self.covariance, "rank": self.rank, "offset_scale": self.offset_scale, "min_slides": self.min_slides,
             "min_slide_tiles": self.min_slide_tiles, "shrinkage": self.shrinkage, "reg": self.reg, "min_tiles": self.min_tiles,
             "mean": torch.from_numpy(self.mean.astype(np.float32)),
             "total_variance": torch.from_numpy(self.total_var.astype(np.float32)),
             "variance": torch.from_numpy(self.var.astype(np.float32)),
             "between_variance": torch.from_numpy(self.between.astype(np.float32)),
             "n_slides": torch.from_numpy(self.n_slides.astype(np.int64)),
             "small_cluster": torch.from_numpy(self.small)}
        if self.chol is not None:
            p["cholesky"] = torch.from_numpy(self.chol.astype(np.float32))
        if self.factors is not None:
            p["factors"] = torch.from_numpy(self.factors.astype(np.float32))
            p["psi"] = torch.from_numpy(self.psi.astype(np.float32))
        return p


GENERATORS = {"gaussian": GaussianGenerator, "hier_lowrank": HierarchicalGenerator}


def _to_numpy(t: torch.Tensor) -> np.ndarray:
    """A saved float32 tensor as a float64 array."""
    return t.numpy().astype(np.float64)


def build_generator_from_parameters(method: str, parameters: dict) -> GeneratorBase:
    """Rebuild a fitted generator from the `generator` entry of cluster_models.pt (see `save_configuration_metadata`)."""
    if method not in GENERATORS:
        raise ValueError(f"Unknown generation method {method!r}; available: {sorted(GENERATORS)}")
    return GENERATORS[method].from_parameters(parameters)


def composition_from_saved(entry: dict) -> pd.DataFrame:
    """
    The composition table of a class (columns n_tiles, prop_<c>) from the `composition` key of cluster_models.pt.

    The table holds no slide identifiers. A file written before this key existed has none: run 01_3 again.
    """
    if "composition" not in entry:
        raise KeyError("cluster_models.pt has no 'composition' key (written by an older 01_3): run notebook 01_3 again with OVERWRITE = True.")
    props = entry["composition"]["proportions"].numpy().astype(np.float64)
    out = pd.DataFrame({"n_tiles": entry["composition"]["n_tiles"].numpy()})
    for c in range(props.shape[1]):
        out[f"prop_{c}"] = props[:, c]
    return out


def _cholesky(cov: np.ndarray, reg: float) -> np.ndarray:
    """Cholesky factor of cov + reg I; extra jitter is added when the matrix is numerically not positive definite."""
    dim = len(cov)
    jitter = 0.0
    for _ in range(8):
        try:
            return np.linalg.cholesky(cov + (reg + jitter) * np.eye(dim))
        except np.linalg.LinAlgError:
            jitter = max(jitter * 10, 1e-6 * max(float(np.trace(cov)) / dim, 1e-12))
    raise np.linalg.LinAlgError("covariance is not positive definite even with jitter")


def build_generator(method: str, **kwargs) -> GeneratorBase:
    """Instantiate a registered generation method ("gaussian" or "hier_lowrank")."""
    if method not in GENERATORS:
        raise ValueError(f"Unknown generation method {method!r}; available: {sorted(GENERATORS)}")
    return GENERATORS[method](**kwargs)


# Main functions: synthetic slides ****************************************************************
def make_synthetic_wsi(generator: GeneratorBase, composition: pd.DataFrame, rng: np.random.Generator,
                       max_tiles: int | None = None, n_templates: int = 1, concentration: float = 1.0) -> dict:
    """
    One synthetic bag: a random blend of real Training templates of the class decides size and cluster mix.

    n_templates distinct Training slides of the class are drawn (different bags may reuse a slide) and
    combined with Dirichlet weights, so no bag copies the composition of a single slide: the cluster
    proportions are the weighted mean of the template proportions and the tile count is the weighted
    mean of the template tile counts (capped at max_tiles). The tile count is split over the clusters
    (largest-remainder rounding), the vectors of each cluster are sampled from the generator and the
    tiles are shuffled. No real embedding is copied and no template identifier is returned.
    With n_templates = 1 one slide is drawn and its size and proportions are used unchanged.

    * Arguments:
    generator: The fitted class generator
    composition: The composition_table of the class (columns n_tiles, prop_<c>)
    rng: The random generator
    max_tiles: The maximum number of tiles of a synthetic bag (None = no cap)
    n_templates: The number of Training slides blended in one bag (reduced to the number available)
    concentration: The Dirichlet concentration of the weights (smaller = one template dominates)

    Returns a dict with X [N, D] float32, cluster [N] (generating cluster), counts [K], n_templates and
    max_template_weight (the largest weight of a single template).
    """

    # STEP 1: Templates and weights
    prop_cols = [c for c in composition.columns if c.startswith("prop_")]
    m = int(min(max(n_templates, 1), len(composition)))
    if m == 1:
        idx, w = np.array([int(rng.integers(len(composition)))]), np.ones(1)
    else:
        idx = rng.choice(len(composition), size=m, replace=False)
        w = rng.dirichlet(np.full(m, float(concentration)))

    # STEP 2: Blended size and cluster counts
    props = w @ composition[prop_cols].to_numpy(dtype=np.float64)[idx]
    n_tiles = int(round(float(w @ composition["n_tiles"].to_numpy(dtype=np.float64)[idx])))
    size = max(1, n_tiles if max_tiles is None else min(n_tiles, int(max_tiles)))
    counts = _apportion(size, props / props.sum())

    # STEP 3-4: Sample the tiles of the cluster counts
    wsi = sample_bag(generator, counts, rng)
    wsi.update({"n_templates": m, "max_template_weight": float(w.max())})
    return wsi


def sample_bag(generator: GeneratorBase, counts: np.ndarray, rng: np.random.Generator) -> dict:
    """
    The tiles of one synthetic bag with `counts` tiles per cluster (the sampling step of every composition mode).

    The per-bag state of the generator (for example one offset per cluster) is drawn once, each cluster is sampled and the
    tiles are shuffled.

    * Arguments:
    generator: The fitted class generator
    counts: The number of tiles per cluster [K]
    rng: The random generator

    Returns a dict with X [N, D], cluster [N] (generating cluster) and counts [K].
    """
    size = int(counts.sum())
    offsets = generator.start_bag(rng)
    parts = [generator.sample(c, int(n), rng, offsets) for c, n in enumerate(counts) if n > 0]
    cluster = np.repeat(np.arange(len(counts)), counts)
    X = np.concatenate(parts)
    order = rng.permutation(size)
    return {"X": X[order], "cluster": cluster[order], "counts": counts}


def synthetic_id(method: str, dim: int, k: int, label: str, i: int) -> str:
    """Synthetic identifier; it carries no real slide or patient information."""
    return f"SYN_{method}_pca{dim}_k{k}_{label}_{i:04d}"


def check_wsi(x: np.ndarray, dim: int, n_tiles: int) -> None:
    """Validate shape and finiteness of a synthetic bag."""
    if x.shape != (n_tiles, dim):
        raise ValueError(f"shape {x.shape}, expected {(n_tiles, dim)}")
    if not np.isfinite(x).all():
        raise ValueError("NaN or infinite values in a synthetic bag")


def generate_configuration(generators: dict, clusters: dict, compositions: dict, real_samples: dict,
                           settings: dict, out_dir: Path | None, overwrite: bool = False, bag_makers: dict | None = None) -> dict:
    """
    Generate (and optionally save) the synthetic WSIs of one (method, PCA dimension, k) configuration.

    Output goes to <out_dir>/<method>/pca_<D>/k_<K>/. Existing synthetic files are never replaced
    unless overwrite is set. After generation every bag is re-assigned to the class clusters, which
    checks that the sampled vectors fall mostly in the cluster they were drawn for.

    * Arguments:
    generators: Class label -> fitted generator
    clusters: Class label -> fitted MiniBatchKMeans
    compositions: Class label -> composition_table
    real_samples: Class label -> [N, D] Training sample (for the distribution statistics only)
    settings: method, dim, k, seed, n_per_class, max_tiles, n_templates, template_concentration, model_name,
        pca_dir, save_embeddings, plus the configuration recorded in metadata.json
    out_dir: The synthetic root (None = nothing is written)
    overwrite: Replace existing synthetic files of this configuration
    bag_makers: {class label: function(generator, rng) -> bag dict (see make_synthetic_wsi; optional key lesion_share)} for the classes
        whose bags are not blended from templates (composition mode "lesion_share"); the other classes use make_synthetic_wsi

    Returns a dict with wsis (DataFrame), distribution (DataFrame), scatter (class -> real/synthetic PC1-PC2),
    bag_means (class -> mean [B, K, D] and counts [B, K] of the cluster means of the first settings["diag_bags"]
    (default 30) bags, from the cluster each tile was drawn for), folder, and written (number of files).
    """

    # STEP 1: Output folder and overwrite guard
    method, dim, k = settings["method"], settings["dim"], settings["k"]
    folder = None
    if out_dir is not None:
        folder = Path(out_dir) / method / f"pca_{dim}" / f"k_{k}"
        if _within(folder, settings["pca_dir"]):
            raise ValueError("The synthetic output must not be inside the PCA input folder.")
        existing = sorted(folder.glob("SYN_*.pt")) if folder.exists() else []
        if existing and not overwrite:
            raise FileExistsError(f"{folder} holds {len(existing)} synthetic files; set OVERWRITE to replace them.")
        folder.mkdir(parents=True, exist_ok=True)
        for f in existing:
            f.unlink()

    # STEP 2: Generate class by class
    rows, dist, scatter, bag_means = [], [], {}, {}
    start = time.perf_counter()
    for ci, (label, gen) in enumerate(generators.items()):
        rng = _rng(settings["seed"], dim, k, ci, 7)
        real = real_samples[label]
        syn_keep, means, bag_counts = [], [], []
        for i in range(settings["n_per_class"]):
            if bag_makers and label in bag_makers:
                wsi = bag_makers[label](gen, rng)
            else:
                wsi = make_synthetic_wsi(gen, compositions[label], rng, settings["max_tiles"], settings.get("n_templates", 1),
                                         settings.get("template_concentration", 1.0))
            n = len(wsi["X"])
            check_wsi(wsi["X"], dim, n)
            reassigned = np.bincount(clusters[label].predict(wsi["X"]), minlength=len(wsi["counts"]))
            name = synthetic_id(method, dim, k, label, i + 1)
            rows.append({"WSI Filename": name, "Patient Id": name, "WSI label": label, "Set": SYNTHETIC_SET,
                         "method": method, "pca_dim": dim, "k": k, "n_tiles": n,
                         "n_templates": wsi["n_templates"], "max_template_weight": wsi["max_template_weight"],
                         "counts": wsi["counts"], "reassigned": reassigned, **({"lesion_share": wsi["lesion_share"]} if "lesion_share" in wsi else {})})
            if i < 5:
                syn_keep.append(wsi["X"][rng.choice(n, size=min(n, 400), replace=False)])
            if i < settings.get("diag_bags", 30):
                means.append(_cluster_means(wsi["X"], wsi["cluster"], len(wsi["counts"])))
                bag_counts.append(wsi["counts"])
            if folder is not None and settings["save_embeddings"]:
                _save_wsi(folder / f"{name}.pt", name, label, wsi["X"], settings)
        syn = np.concatenate(syn_keep)
        dist.append({"label": label, **distribution_stats(real, syn)})
        scatter[label] = {"real": real[:1000, :2], "synthetic": syn[:1000, :2]}
        bag_means[label] = {"mean": np.stack(means), "counts": np.stack(bag_counts)}
    seconds = time.perf_counter() - start

    # STEP 3: Tables
    wsis = pd.DataFrame(rows)
    return {"wsis": wsis, "distribution": pd.DataFrame(dist), "scatter": scatter, "bag_means": bag_means, "folder": folder,
            "written": len(wsis) if folder is not None and settings["save_embeddings"] else 0,
            "seconds": seconds}


def _cluster_means(X: np.ndarray, cluster: np.ndarray, k: int) -> np.ndarray:
    """Mean of the tiles of each cluster in one bag [K, D] (zero for an empty cluster)."""
    total = np.zeros((k, X.shape[1]))
    np.add.at(total, cluster, X.astype(np.float64))
    return (total / np.maximum(np.bincount(cluster, minlength=k), 1)[:, None]).astype(np.float32)


def _save_wsi(path: Path, name: str, label: str, X: np.ndarray, settings: dict) -> None:
    """Write one synthetic bag with the keys of the PCA files (float32, no real identifiers)."""
    torch.save({
        "cls_features": torch.from_numpy(np.ascontiguousarray(X, dtype=np.float32)),
        "coordinates": torch.zeros(len(X), 2, dtype=torch.long),
        "model_encoder": settings["model_name"],
        "pca": {"n_components": settings["dim"], "whiten": settings.get("whiten", False),
                "model_file": "../pca_model.pt"},
        "tile_params": {"synthetic": True, "method": settings["method"], "k": settings["k"],
                        "seed": settings["seed"], "source_set": "Training", "n_templates": settings.get("n_templates", 1)},
        "slide": name, "label": label, "patient": name, "set": SYNTHETIC_SET, "n_tiles": int(len(X)),
    }, path)


def distribution_stats(real: np.ndarray, synthetic: np.ndarray) -> dict:
    """
    Per-coordinate comparison of real and synthetic tiles.

    * Arguments:
    real: The [N, D] real Training sample of a class
    synthetic: The [M, D] synthetic tiles of the same class

    Returns the mean absolute standardised mean difference (mean |mu_s - mu_r| / sd_r), the median ratio
    of standard deviations (synthetic / real) and the mean vector norms.
    """
    sd = real.std(0) + 1e-12
    return {"mean_abs_smd": float(np.mean(np.abs(synthetic.mean(0) - real.mean(0)) / sd)),
            "median_std_ratio": float(np.median(synthetic.std(0) / sd)),
            "norm_real": float(np.linalg.norm(real, axis=1).mean()),
            "norm_synthetic": float(np.linalg.norm(synthetic, axis=1).mean())}


# Main functions: saving **************************************************************************
def save_configuration_metadata(result: dict, clusters: dict, generators: dict, compositions: dict,
                                settings: dict, source_slides: list[str]) -> None:
    """
    Write metadata.json, cluster_models.pt and labels.csv into the folder of one configuration.

    * Arguments:
    result: The output of generate_configuration (needs folder and wsis)
    clusters: Class label -> fitted MiniBatchKMeans
    generators: Class label -> fitted generator
    compositions: Class label -> composition_table
    settings: The configuration of the run (see generate_configuration)
    source_slides: The Training slides used
    """
    folder = result["folder"]
    if folder is None:
        return
    meta = {key: value for key, value in settings.items() if key not in ("pca_dir", "extra_metadata", "lesion_models")}
    meta.update(settings.get("extra_metadata", {}))
    meta.update({
        "pca_input": str(settings["pca_dir"]), "source_split": "Training", "n_source_slides": len(source_slides),
        "source_slides_sha1": hashlib.sha1("\n".join(sorted(source_slides)).encode()).hexdigest(),
        "effective_k": {c: int(m.n_clusters) for c, m in clusters.items()},
        "n_generated": result["wsis"].groupby("WSI label").size().to_dict(),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "versions": {"python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__,
                     "pandas": pd.__version__, "scikit-learn": sklearn.__version__, "torch": torch.__version__,
                     "matplotlib": matplotlib.__version__},
    })
    (folder / "metadata.json").write_text(json.dumps(meta, indent=1, default=str))
    torch.save({c: {"centers": torch.from_numpy(m.cluster_centers_.astype(np.float32)),
                    "tiles_per_cluster": torch.from_numpy(_cluster_sizes(compositions[c])),
                    "generator": generators[c].parameters(),
                    "composition": _composition_arrays(compositions[c])} for c, m in clusters.items()},
               folder / "cluster_models.pt")
    result["wsis"][LABEL_COLUMNS].to_csv(folder / "labels.csv", index=False)
    if "lesion_share" in result["wsis"]:                      # composition mode "lesion_share" only: the lesion share of every bag
        result["wsis"][["WSI Filename", "WSI label", "n_tiles", "lesion_share"]].to_csv(folder / "lesion_share.csv", index=False)
    if settings.get("lesion_models"):                         # the lesion-share model of the classes that use it
        saved = torch.load(folder / "cluster_models.pt", weights_only=True)
        for label, model in settings["lesion_models"].items():
            saved[label]["lesion"] = model.to_saved()
        torch.save(saved, folder / "cluster_models.pt")


def _composition_arrays(composition: pd.DataFrame) -> dict:
    """The templates of a class (tiles and cluster proportions per Training slide) without slide identifiers, for fresh bags."""
    cols = [c for c in composition.columns if c.startswith("prop_")]
    return {"n_tiles": torch.from_numpy(composition["n_tiles"].to_numpy(dtype=np.int64).copy()),
            "proportions": torch.from_numpy(composition[cols].to_numpy(dtype=np.float32).copy())}


def _cluster_sizes(composition: pd.DataFrame) -> np.ndarray:
    """Training tiles per cluster, from the composition table."""
    cols = [c for c in composition.columns if c.startswith("prop_")]
    return np.rint(composition[cols].to_numpy() * composition[["n_tiles"]].to_numpy()).sum(0).astype(np.int64)


def write_synthetic_labels(results: list[dict], path: str | Path, overwrite: bool = False) -> pd.DataFrame:
    """
    Write the label sheet of all configurations (CSV, like labels/BRACS_mod.csv).

    Rows of the configurations in `results` replace the rows of the same (method, pca_dim, k) already
    in the file; other rows are kept. An existing sheet with rows for one of these configurations is
    only replaced when overwrite is set.

    * Arguments:
    results: The outputs of generate_configuration that were saved
    path: The CSV file
    overwrite: Replace rows of configurations that are already in the file

    Returns the sheet.
    """
    path = Path(path)
    new = pd.concat([r["wsis"][LABEL_COLUMNS] for r in results if r["folder"] is not None], ignore_index=True)
    if path.exists():
        old = pd.read_csv(path)
        keys = set(zip(new["method"], new["pca_dim"], new["k"]))
        clash = old[[key in keys for key in zip(old["method"], old["pca_dim"], old["k"])]]
        if len(clash) and not overwrite:
            raise FileExistsError(f"{path} already lists {len(clash)} synthetic WSIs of these configurations.")
        new = pd.concat([old.drop(clash.index), new], ignore_index=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    new.to_csv(path, index=False)
    return new


# Main functions: tables and checks ***************************************************************
def training_overview(train: pd.DataFrame, classes: list[str]) -> pd.DataFrame:
    """Training WSIs, patients and tiles per class."""
    g = train.groupby("label").agg(wsis=("slide", "size"), patients=("patient", "nunique"),
                                   tiles=("n_tiles", "sum"), median_tiles=("n_tiles", "median"),
                                   min_tiles=("n_tiles", "min"), max_tiles=("n_tiles", "max"))
    return g.reindex(classes).reset_index()


def cluster_size_table(results: dict) -> pd.DataFrame:
    """Cluster sizes (Training tiles per cluster) of every (dim, k, class): smallest, median, largest, empty."""
    rows = []
    for (dim, k), per_class in results.items():
        for label, sizes in per_class.items():
            sizes = np.asarray(sizes)
            rows.append({"pca_dim": dim, "k": k, "label": label, "clusters": len(sizes), "min": int(sizes.min()),
                         "median": float(np.median(sizes)), "max": int(sizes.max()),
                         "largest_share": float(sizes.max() / sizes.sum()), "empty": int((sizes == 0).sum())})
    return pd.DataFrame(rows)


def generation_check(results: dict, n_per_class: int, classes: list[str]) -> pd.DataFrame:
    """Generated WSIs per (dim, k, class) against the configuration; raises when a count differs."""
    rows = []
    for (dim, k), res in results.items():
        got = res["wsis"].groupby("WSI label").size()
        for c in classes:
            rows.append({"pca_dim": dim, "k": k, "label": c, "requested": n_per_class,
                         "generated": int(got.get(c, 0)), "files": int(got.get(c, 0)) if res["written"] > 0 else 0})
    out = pd.DataFrame(rows)
    if (out["requested"] != out["generated"]).any():
        raise AssertionError("A configuration generated a different number of WSIs than requested.")
    return out


def composition_comparison(compositions: dict, result: dict) -> pd.DataFrame:
    """
    Mean (and standard deviation over bags) of the cluster proportions of the real Training slides and of the synthetic bags.

    The synthetic proportions are those of the clusters the tiles were drawn for ("drawn") and those
    obtained by assigning the generated tiles to the nearest centre again ("reassigned").
    """
    rows = []
    wsis = result["wsis"]
    for label, comp in compositions.items():
        real = comp.filter(like="prop_").to_numpy().mean(0)
        sub = wsis[wsis["WSI label"] == label]
        drawn = np.mean([c / c.sum() for c in sub["counts"]], axis=0)
        back = np.mean([c / c.sum() for c in sub["reassigned"]], axis=0)
        real_sd = comp.filter(like="prop_").to_numpy().std(0)
        drawn_sd = np.std([c / c.sum() for c in sub["counts"]], axis=0)
        for c in range(len(real)):
            rows.append({"label": label, "cluster": c, "real": real[c], "drawn": drawn[c], "reassigned": back[c],
                         "real_std": real_sd[c], "drawn_std": drawn_sd[c]})
    return pd.DataFrame(rows)


def between_variance_table(stats: dict, bag_means: dict, min_tiles: int = 10, min_slide_tiles: int = 5) -> pd.DataFrame:
    """
    Between-slide share of the variance of every cluster, real Training slides against synthetic bags.

    The share is the mean over the coordinates of B / total variance, with B from between_slide_variance applied
    in the same way to the slide means of the real slides and to the cluster means of the synthetic bags (the
    total variance of the real tiles is used for the noise correction in both).

    * Arguments:
    stats: Class label -> the output of collect_class_statistics (with slide_mean)
    bag_means: Class label -> {"mean": [B, K, D], "counts": [B, K]} (the bag_means of generate_configuration)
    min_tiles: The minimum tiles of a cluster for its own total variance (as the generator)
    min_slide_tiles: The minimum tiles of a slide or bag in a cluster for its mean to count

    Returns a DataFrame with label, cluster, tiles, slides (real), bags (synthetic), real, synthetic.
    """
    rows = []
    for label, st in stats.items():
        total, _ = _total_variance(st, min_tiles)
        real, n_real = between_slide_variance(st["slide_mean"], st["counts"], total, min_slide_tiles)
        bags = bag_means[label]
        syn, n_syn = between_slide_variance(bags["mean"].astype(np.float64), bags["counts"], total, min_slide_tiles)
        for c in range(len(total)):
            rows.append({"label": label, "cluster": c, "tiles": int(st["n"][c]), "slides": int(n_real[c]), "bags": int(n_syn[c]),
                         "real": float((real[c] / np.maximum(total[c], 1e-30)).mean()),
                         "synthetic": float((syn[c] / np.maximum(total[c], 1e-30)).mean())})

    return pd.DataFrame(rows)


def sample_synthetic_tiles(generator: GeneratorBase, composition: pd.DataFrame, rng: np.random.Generator, n_bags: int = 20,
                           per_bag: int = 250, n_templates: int = 1, concentration: float = 1.0,
                           max_tiles: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    """
    A small in-memory sample of synthetic tiles with their bag, for the diagnostics (nothing is written).

    * Arguments:
    generator: The fitted class generator
    composition: The composition_table of the class
    rng: The random generator
    n_bags: The number of bags generated
    per_bag: The tiles kept per bag
    n_templates: The templates blended per bag (as make_synthetic_wsi)
    concentration: The Dirichlet concentration (as make_synthetic_wsi)
    max_tiles: The cap on the bag size (as make_synthetic_wsi)

    Returns the [N, D] tiles and the bag index [N] of every tile.
    """
    tiles, groups = [], []
    for b in range(n_bags):
        X = make_synthetic_wsi(generator, composition, rng, max_tiles, n_templates, concentration)["X"]
        keep = rng.choice(len(X), size=min(len(X), per_bag), replace=False)
        tiles.append(X[keep])
        groups.append(np.full(len(keep), b))

    return np.concatenate(tiles), np.concatenate(groups)


def fidelity_auc(real: np.ndarray, real_groups: np.ndarray, synthetic: np.ndarray, synthetic_groups: np.ndarray, seed: int,
                 max_tiles: int = 5000, n_splits: int = 5) -> dict:
    """
    How well a classifier tells real tiles from synthetic ones (ROC-AUC on held-out groups).

    A shallow gradient-boosting classifier (HistGradientBoostingClassifier, depth 3) is trained to separate real tiles (label 0) from
    synthetic tiles (label 1) and scored on tiles of slides / bags it has not seen: the folds are grouped by real
    slide and by synthetic bag (StratifiedGroupKFold), so a slide- or bag-specific signature cannot be memorised. Both
    sides are subsampled to the same number of tiles. An AUC of 0.5 means that the tiles cannot be told apart;
    values toward 1 mean that the generator misses structure that the classifier finds.

    * Arguments:
    real: The [N, D] real tiles
    real_groups: The slide of every real tile
    synthetic: The [M, D] synthetic tiles
    synthetic_groups: The bag of every synthetic tile
    seed: The random seed
    max_tiles: The maximum tiles per side
    n_splits: The folds (reduced to the number of slides or bags available)

    Returns a dict with auc (mean over folds), auc_std, n_tiles (per side) and n_folds.
    """

    # STEP 1: Same number of tiles from each side, groups made distinct across the two sides
    rng = np.random.default_rng(seed)
    size = int(min(max_tiles, len(real), len(synthetic)))
    i_r, i_s = rng.choice(len(real), size, replace=False), rng.choice(len(synthetic), size, replace=False)
    X = np.concatenate([real[i_r], synthetic[i_s]]).astype(np.float32)
    y = np.concatenate([np.zeros(size, dtype=int), np.ones(size, dtype=int)])
    g_r = pd.factorize(np.asarray(real_groups)[i_r])[0]
    g_s = pd.factorize(np.asarray(synthetic_groups)[i_s])[0]
    groups = np.concatenate([g_r, g_s + g_r.max() + 1])
    splits = int(min(n_splits, g_r.max() + 1, g_s.max() + 1))
    if splits < 2:
        raise ValueError("fidelity_auc needs at least two slides and two bags.")

    # STEP 2: Grouped cross-validation
    scores = []
    for fit, test in StratifiedGroupKFold(n_splits=splits, shuffle=True, random_state=seed).split(X, y, groups):
        if len(np.unique(y[test])) < 2:
            continue
        model = HistGradientBoostingClassifier(max_iter=60, max_depth=3, min_samples_leaf=100, random_state=seed).fit(X[fit], y[fit])
        scores.append(roc_auc_score(y[test], model.predict_proba(X[test])[:, 1]))

    return {"auc": float(np.mean(scores)), "auc_std": float(np.std(scores)), "n_tiles": size, "n_folds": len(scores)}


# Main functions: plots ***************************************************************************
def plot_cluster_sizes(results: dict, dim: int, classes: list[str], show: bool = True):
    """
    Sorted cluster sizes of every class, one panel per k, for one PCA dimension.

    * Arguments:
    results: (dim, k) -> class -> Training tiles per cluster
    dim: The PCA dimension to show
    classes: The class labels (colour order)
    show: Display the figure
    """
    ks = sorted(k for d, k in results if d == dim)
    colors = dict(zip(classes, cluster_colors(len(classes))))
    notes = [(0, len(ks), "Share of the Training tiles of a class that fall in each cluster, clusters ordered by decreasing size. "
                          "A flat curve means balanced clusters; a steep one, a dominant cluster. Clusters are fitted "
                          "separately for each class, so a rank is not the same cluster across classes.")]
    with plt.rc_context(STYLE):
        fig = plt.figure(figsize=(4.2 * len(ks), 5.6))
        plots = split_with_notes(fig, notes, [1] * len(ks), 4.2 * len(ks), plot_height=3.4, note_height=1.0)
        axes = np.atleast_1d(plots.subplots(1, len(ks), sharey=True))
        for ax, k in zip(axes, ks):
            for label in classes:
                s = np.sort(np.asarray(results[(dim, k)][label]))[::-1]
                ax.plot(np.arange(1, len(s) + 1), s / s.sum(), marker="o", ms=3, lw=1.2, color=colors[label], label=label)
            ax.set_title(f"k = {k}"); ax.set_xlabel("cluster rank"); ax.set_xticks(range(1, k + 1) if k <= 10 else range(5, k + 1, 5))
        axes[0].set_ylabel("share of class tiles"); axes[0].legend(frameon=False, fontsize=8, ncol=2)
        if show:
            plt.show()
    return fig


def plot_tile_counts(real: pd.DataFrame, wsis: pd.DataFrame, classes: list[str], show: bool = True):
    """
    Tile count per WSI, real Training slides and synthetic bags, per class.

    * Arguments:
    real: The Training slides (columns label, n_tiles)
    wsis: The synthetic WSIs of one configuration (columns WSI label, n_tiles)
    classes: The class labels
    show: Display the figure
    """
    notes = [(0, 1, "Tile count per bag on a logarithmic axis, real Training slides (blue) next to synthetic "
                    "bags (orange). The tile count of a synthetic bag is a weighted mean of the tile counts of the real templates "
                    "blended in it, so the mean of the distribution is preserved while its spread is narrower than that of "
                    "the real slides when more than one template is blended (and the optional cap on the bag size applies).")]
    with plt.rc_context(STYLE):
        fig = plt.figure(figsize=(11, 5.6))
        plots = split_with_notes(fig, notes, [1], 11, plot_height=3.8, note_height=1.4)
        ax = plots.subplots(1, 1)
        for i, label in enumerate(classes):
            for off, data, col in ((-0.18, real.loc[real["label"] == label, "n_tiles"], SERIES[0]),
                                   (0.18, wsis.loc[wsis["WSI label"] == label, "n_tiles"], SERIES[1])):
                bp = ax.boxplot([data], positions=[i + off], widths=0.3, patch_artist=True, showfliers=False)
                for patch in bp["boxes"]:
                    patch.set(facecolor=col, alpha=0.35, edgecolor=col)
                for part in ("whiskers", "caps", "medians"):
                    for line in bp[part]:
                        line.set(color=col)
        ax.set_xticks(range(len(classes)), classes); ax.set_yscale("log"); ax.set_ylabel("tiles per WSI")
        ax.plot([], [], color=SERIES[0], label="real (Training)"); ax.plot([], [], color=SERIES[1], label="synthetic")
        ax.legend(frameon=False)
        if show:
            plt.show()
    return fig


def plot_composition(comparison: pd.DataFrame, classes: list[str], title: str = "", show: bool = True):
    """
    Mean cluster proportions: real templates against the synthetic bags of one configuration.

    * Arguments:
    comparison: The output of composition_comparison
    classes: The class labels
    title: A configuration label for the panel titles
    show: Display the figure
    """
    colors = dict(zip(classes, cluster_colors(len(classes))))
    notes = [(0, 1, f"{title} Mean proportion of a cluster over the real Training templates (x) and over the synthetic "
                    "bags (y) as drawn by the generator. Points on the diagonal mean that the cluster mix of the "
                    "templates is reproduced."),
             (1, 2, "The same synthetic proportions after the generated tiles are assigned again to the nearest "
                    "centre of their class. Deviations from the diagonal measure how far the sampled vectors leave "
                    "the cluster they were drawn for, which grows when clusters overlap or when the covariance "
                    "regularisation is large.")]
    with plt.rc_context(STYLE):
        fig = plt.figure(figsize=(11, 6.2))
        plots = split_with_notes(fig, notes, [1, 1], 11, plot_height=3.6, note_height=2.0)
        axes = plots.subplots(1, 2, sharex=True, sharey=True)
        for ax, col, name in zip(axes, ("drawn", "reassigned"), ("drawn", "re-assigned")):
            for label in classes:
                sub = comparison[comparison["label"] == label]
                ax.scatter(sub["real"], sub[col], s=22, color=colors[label], label=label, alpha=0.85)
            top = float(max(comparison["real"].max(), comparison[col].max())) * 1.05
            ax.plot([0, top], [0, top], color=INK_2, lw=0.8, ls="--")
            ax.set_xlabel("real (mean template proportion)"); ax.set_ylabel(f"synthetic, {name}")
        axes[0].legend(frameon=False, fontsize=8, ncol=2)
        if show:
            plt.show()
    return fig


def plot_distribution_stats(table: pd.DataFrame, classes: list[str], show: bool = True):
    """
    Per-coordinate agreement of real and synthetic tiles for every PCA dimension and k.

    * Arguments:
    table: Columns pca_dim, k, label, mean_abs_smd, median_std_ratio
    classes: The class labels
    show: Display the figure
    """
    table = table.assign(config=lambda t: "D" + t["pca_dim"].astype(str) + " k" + t["k"].astype(str))
    configs = list(dict.fromkeys(table["config"]))
    colors = dict(zip(classes, cluster_colors(len(classes))))
    notes = [(0, 1, "Mean over coordinates of |mean_synthetic - mean_real| divided by the real standard deviation, per "
                    "class. Zero means identical coordinate means."),
             (1, 2, "Median over coordinates of the ratio of synthetic to real standard deviation. The generator "
                    "models each cluster separately, so a ratio below one indicates variance that a diagonal "
                    "within-cluster model does not capture (correlations, within-cluster structure); a "
                    "ratio above one, an inflated regularisation term.")]
    with plt.rc_context(STYLE):
        fig = plt.figure(figsize=(13, 5.6))
        plots = split_with_notes(fig, notes, [1, 1], 13, plot_height=3.6, note_height=1.8)
        a, b = plots.subplots(1, 2)
        x = np.arange(len(configs))
        for j, label in enumerate(classes):
            sub = table[table["label"] == label].set_index("config").reindex(configs)
            a.plot(x, sub["mean_abs_smd"], marker="o", ms=4, color=colors[label], label=label)
            b.plot(x, sub["median_std_ratio"], marker="o", ms=4, color=colors[label])
        b.axhline(1, color=INK_2, lw=0.8, ls="--")
        for ax, yl in ((a, "mean |standardised mean difference|"), (b, "median std ratio (synthetic / real)")):
            ax.set_xticks(x, configs, rotation=60, ha="right", fontsize=8); ax.set_ylabel(yl)
        a.legend(frameon=False, fontsize=8, ncol=2)
        if show:
            plt.show()
    return fig


def plot_pc_scatter(scatter: dict, classes: list[str], title: str = "", show: bool = True):
    """
    The first two principal-component scores of real and synthetic tiles, one panel per class.

    * Arguments:
    scatter: Class -> {"real": [n, 2], "synthetic": [m, 2]}
    classes: The class labels
    title: A configuration label
    show: Display the figure
    """
    n = len(classes)
    notes = [(0, n, f"{title} Scores on the first two principal components (PC1, PC2) of a random subsample of the "
                    "real Training tiles (blue) and of the synthetic tiles (orange) of each class. The projection "
                    "shows only the two directions of largest variance; agreement there does not establish "
                    "agreement in the remaining coordinates (see the coordinate-wise statistics).")]
    with plt.rc_context(STYLE):
        fig = plt.figure(figsize=(3.0 * n, 4.8))
        plots = split_with_notes(fig, notes, [1] * n, 3.0 * n, plot_height=2.8, note_height=1.4)
        axes = np.atleast_1d(plots.subplots(1, n, sharex=True, sharey=True))
        for ax, label in zip(axes, classes):
            ax.scatter(*scatter[label]["real"].T, s=4, alpha=0.4, color=SERIES[0], label="real")
            ax.scatter(*scatter[label]["synthetic"].T, s=4, alpha=0.4, color=SERIES[1], label="synthetic")
            ax.set_title(label, fontsize=10); ax.set_xlabel("PC1")
        axes[0].set_ylabel("PC2"); axes[0].legend(frameon=False, fontsize=8, markerscale=3)
        if show:
            plt.show()
    return fig


def plot_between_variance(table: pd.DataFrame, classes: list[str], title: str = "", show: bool = True):
    """
    Between-slide share of the variance of every cluster: real slides (x) against synthetic bags (y).

    * Arguments:
    table: The output of between_variance_table
    classes: The class labels (colour order)
    title: A configuration label
    show: Display the figure
    """
    colors = dict(zip(classes, cluster_colors(len(classes))))
    notes = [(0, 1, f"{title} Share of the variance of a cluster that lies between slides: the variance of the slide "
                    "means of the cluster (weighted by the tiles of each slide, corrected for the sampling noise of a "
                    "mean of few tiles), divided by the total variance of the cluster and averaged over the coordinates. "
                    "Each point is one cluster, estimated over the real Training slides (x) and over the synthetic bags (y) "
                    "with the same estimator. Points on the diagonal mean that a synthetic bag shifts a cluster as much "
                    "as a real slide does; points on the horizontal axis mean that the shift is averaged out within the bag."),
             (1, 2, "The same shares for the clusters seen in at least five real slides and five bags only. Clusters "
                    "seen in fewer slides have an estimate that is dominated by the shrinkage toward the pooled share "
                    "in the generator and by the noise in the real estimate, and are left out here.")]
    with plt.rc_context(STYLE):
        fig = plt.figure(figsize=(11, 6.0))
        plots = split_with_notes(fig, notes, [1, 1], 11, plot_height=3.6, note_height=2.2)
        axes = plots.subplots(1, 2, sharex=True, sharey=True)
        top = float(max(table["real"].max(), table["synthetic"].max(), 1e-3)) * 1.05
        for ax, sub_table in zip(axes, (table, table[(table["slides"] >= 5) & (table["bags"] >= 5)])):
            for label in classes:
                sub = sub_table[sub_table["label"] == label]
                ax.scatter(sub["real"], sub["synthetic"], s=22, color=colors[label], label=label, alpha=0.85)
            ax.plot([0, top], [0, top], color=INK_2, lw=0.8, ls="--")
            ax.set_xlabel("real: between-slide share of the variance"); ax.set_ylabel("synthetic")
        axes[0].legend(frameon=False, fontsize=8, ncol=2)
        if show:
            plt.show()
    return fig


def plot_fidelity_auc(table: pd.DataFrame, classes: list[str], show: bool = True):
    """
    Held-out ROC-AUC of the classifier that separates real from synthetic tiles, per class and generation method.

    * Arguments:
    table: Columns label, method, auc, auc_std (the output of fidelity_auc, one row per class and method)
    classes: The class labels
    show: Display the figure
    """
    methods = list(dict.fromkeys(table["method"]))
    notes = [(0, 1, "Area under the ROC curve of a gradient-boosting classifier that separates real Training tiles from "
                    "synthetic tiles, on tiles of slides and bags that were not used to fit it (folds grouped by slide "
                    "and by bag); bars show the mean over folds and the whiskers one standard deviation. An AUC of 0.5 "
                    "means that real and synthetic tiles cannot be told apart. The real tiles are Training tiles that "
                    "the generator was fitted on, so the value measures how well the generator reproduces its own "
                    "training distribution, not its generalisation to other slides. With few slides the classifier can memorise regions "
                    "of the slides it was fitted on, so a value below 0.5 is possible and is read as no detectable difference; "
                    "the reference row compares two disjoint halves of the real slides and shows the value that two real "
                    "populations of this size obtain.")]
    with plt.rc_context(STYLE):
        fig = plt.figure(figsize=(11, 5.6))
        plots = split_with_notes(fig, notes, [1], 11, plot_height=3.6, note_height=1.6)
        ax = plots.subplots(1, 1)
        width = 0.8 / len(methods)
        for j, method in enumerate(methods):
            sub = table[table["method"] == method].set_index("label").reindex(classes)
            ax.bar(np.arange(len(classes)) + (j - (len(methods) - 1) / 2) * width, sub["auc"], width * 0.9,
                   yerr=sub["auc_std"], color=SERIES[j % len(SERIES)], alpha=0.85, label=method, capsize=2)
        ax.axhline(0.5, color=INK_2, lw=0.8, ls="--")
        ax.set_xticks(range(len(classes)), classes); ax.set_ylabel("held-out ROC-AUC (real vs synthetic)"); ax.set_ylim(0.4, 1.0)
        ax.legend(frameon=False)
        if show:
            plt.show()
    return fig


# Main functions: cluster support and pruning *****************************************************
def _slides_seen(counts: np.ndarray, min_slide_tiles: int) -> np.ndarray:
    """Slides with at least `min_slide_tiles` tiles in each cluster [K], from the per-slide counts [S, K]."""
    return (counts >= min_slide_tiles).sum(0)


def cluster_support_table(stats: dict, classes: list[str], min_slide_tiles: int = 5, few_slides: int = 5) -> pd.DataFrame:
    """
    How many Training slides stand behind each cluster (read-only diagnostic of the clusters of 01_3).

    A slide supports a cluster when it has at least `min_slide_tiles` tiles in it (the rule of between_slide_variance). A cluster
    backed by one or two slides is close to a copy of those slides, which weakens the generalisation of the synthetic bags and the
    privacy of the source slides, and a classifier trained on them can follow the patient rather than the class.

    * Arguments:
    stats: (dim, k) -> class -> the output of collect_class_statistics (needs counts [S, K])
    classes: The class labels (row order)
    min_slide_tiles: The tiles a slide needs in a cluster to support it
    few_slides: A cluster seen in fewer slides than this is "poorly supported"

    Returns one row per (pca_dim, k, label): slides, non-empty clusters, median and minimum slides per cluster, the share of clusters and of
    tiles in poorly supported clusters, and the largest single-slide share of a cluster (maximum over clusters, and mean weighted by cluster size).
    """
    rows = []
    for (dim, k), per in stats.items():
        for label in classes:
            counts = per[label]["counts"]
            tiles = counts.sum(0)
            used = tiles > 0
            seen = _slides_seen(counts, min_slide_tiles)[used]
            single = (counts.max(0)[used] / tiles[used])
            weight = tiles[used] / tiles[used].sum()
            few = seen < few_slides
            rows.append({"pca_dim": dim, "k": k, "label": label, "slides": counts.shape[0], "clusters": int(used.sum()),
                         "median_slides": float(np.median(seen)), "min_slides": int(seen.min()),
                         "share_clusters_few": float(few.mean()), "share_tiles_few": float(weight[few].sum()),
                         "max_single_slide_share": float(single.max()), "mean_single_slide_share": float((single * weight).sum())})
    return pd.DataFrame(rows)


def plot_cluster_support(stats: dict, dim: int, classes: list[str], min_slide_tiles: int = 5, few_slides: int = 5, show: bool = True):
    """
    Cumulative share of the Training tiles of a class by the number of slides that support their cluster, one panel per k.

    * Arguments:
    stats: (dim, k) -> class -> the output of collect_class_statistics
    dim: The PCA dimension to show
    classes: The class labels (colour order)
    min_slide_tiles: The tiles a slide needs in a cluster to support it
    few_slides: The dashed line: clusters left of it are poorly supported
    show: Display the figure
    """
    ks = sorted(k for d, k in stats if d == dim)
    colors = dict(zip(classes, cluster_colors(len(classes))))
    notes = [(0, len(ks), "Cumulative share of the Training tiles of a class (vertical axis) that fall in clusters seen in at most the given "
                          "number of slides (horizontal axis); a slide sees a cluster when it has at least the minimum number of tiles in it. "
                          "The dashed line marks the threshold of a poorly supported cluster. A curve that rises early means that "
                          "much of the class lies in clusters that one or a few slides define, so their synthetic tiles are close to those "
                          "slides; the classes differ in the number of slides, which bounds the right end of each curve.")]
    with plt.rc_context(STYLE):
        fig = plt.figure(figsize=(4.4 * len(ks), 5.8))
        plots = split_with_notes(fig, notes, [1] * len(ks), 4.4 * len(ks), plot_height=3.4, note_height=1.3)
        axes = np.atleast_1d(plots.subplots(1, len(ks), sharey=True))
        for ax, k in zip(axes, ks):
            for label in classes:
                counts = stats[(dim, k)][label]["counts"]
                tiles, seen = counts.sum(0), _slides_seen(counts, min_slide_tiles)
                order = np.argsort(seen, kind="stable")
                ax.step(np.concatenate([[0], seen[order]]), np.concatenate([[0], np.cumsum(tiles[order]) / tiles.sum()]), where="post", lw=1.3,
                        color=colors[label], label=label)
            ax.axvline(few_slides - 0.5, color=INK_2, lw=0.8, ls="--")
            ax.set_title(f"k = {k}"); ax.set_xlabel("slides that see the cluster"); ax.set_xscale("symlog", linthresh=10)
        axes[0].set_ylabel("cumulative share of class tiles"); axes[0].legend(frameon=False, fontsize=8, ncol=2)
        if show:
            plt.show()
    return fig


def reduce_clusters(model, keep: np.ndarray):
    """A copy of a fitted MiniBatchKMeans that keeps only the centres `keep`: its predict assigns a tile to the nearest remaining centre."""
    reduced = copy.deepcopy(model)
    reduced.cluster_centers_ = np.ascontiguousarray(model.cluster_centers_[keep])
    reduced.n_clusters = len(keep)
    return reduced


def prune_clusters(clusters: dict, stats: dict, classes: list[str], min_slides: int, min_slide_tiles: int, recollect) -> tuple[dict, dict, dict]:
    """
    Remove the clusters that fewer than `min_slides` Training slides support, and collect the statistics again.

    The support of a cluster is the number of slides with at least `min_slide_tiles` tiles in it, from the per-slide counts of
    the first statistics pass; with a shared codebook (one model for all classes) the slides of all classes are counted together.
    The centres of the sparse clusters are removed from the model, so their tiles go to the nearest remaining centre, and
    `recollect` computes the statistics of every affected class with the reduced model (one more streaming pass). The pass
    is not repeated: reassigned tiles can leave another cluster with fewer slides than the threshold. At least one cluster is kept.

    * Arguments:
    clusters: class -> fitted MiniBatchKMeans, for one (dim, k)
    stats: class -> collect_class_statistics output, for the same (dim, k)
    classes: The class labels
    min_slides: A cluster seen in fewer slides is removed (0 = nothing is removed)
    min_slide_tiles: The tiles a slide needs in a cluster to support it
    recollect: function(label, model) -> collect_class_statistics output

    Returns (clusters, stats, info): the new dicts (the inputs when nothing was removed) and, per model group, the removed clusters
    {"classes", "n_before", "n_after", "removed" (old indices), "slides_seen" and "tiles" of the removed clusters, "kept" (old index of every new cluster)}.
    """
    if min_slides <= 0:
        return clusters, stats, {}
    groups: dict[int, list[str]] = {}
    for label in classes:
        groups.setdefault(id(clusters[label]), []).append(label)
    new_clusters, new_stats, info = dict(clusters), dict(stats), {}
    for labels in groups.values():
        model = clusters[labels[0]]
        seen = sum(_slides_seen(stats[c]["counts"], min_slide_tiles) for c in labels)
        tiles = sum(stats[c]["counts"].sum(0) for c in labels)
        removed = np.flatnonzero(seen < min_slides)
        if len(removed) == 0:
            continue
        keep = np.setdiff1d(np.arange(len(seen)), removed)
        if len(keep) == 0:
            keep = np.array([int(np.argmax(seen))])
            removed = np.setdiff1d(np.arange(len(seen)), keep)
        reduced = reduce_clusters(model, keep)
        for c in labels:
            new_clusters[c] = reduced
            new_stats[c] = recollect(c, reduced)
        info["+".join(labels)] = {"classes": labels, "n_before": int(len(seen)), "n_after": int(len(keep)), "removed": removed.tolist(),
                                  "slides_seen": seen[removed].astype(int).tolist(), "tiles": tiles[removed].astype(int).tolist(), "kept": keep.tolist()}

    return new_clusters, new_stats, info
