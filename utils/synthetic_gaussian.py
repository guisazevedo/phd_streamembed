# IMPORTS *****************************************************************************************
# External libraries
from pathlib import Path
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
                       max_per_slide: int, max_total: int, seed: int) -> np.ndarray:
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

    Returns the [N, D] float32 sample.
    """
    slides = train.loc[train["label"] == label, "slide"].tolist()
    sample = sample_tiles(pca_dir, model_name, slides, {s: label for s in slides}, max_per_slide, max_total, seed)
    return sample["X"]


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

    Returns a dict with counts [S, K] (tiles per slide and cluster), n [K], sum [K, D], sumsq [K, D]
    and xtx [K, D, D] or None.
    """

    # STEP 1: Allocate the accumulators
    k, dim = model.cluster_centers_.shape
    counts = np.zeros((len(slides), k), dtype=np.int64)
    total = np.zeros((k, dim)); total_sq = np.zeros((k, dim))
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
                if xtx is not None:
                    xtx[c] += xc.T @ xc
    return {"counts": counts, "n": counts.sum(0), "sum": total, "sumsq": total_sq, "xtx": xtx}


def composition_table(stats: dict, slides: list[str]) -> pd.DataFrame:
    """Per Training slide: total tiles and tiles / proportion per cluster (the templates of the generator)."""
    counts = stats["counts"]
    out = pd.DataFrame({"slide": slides, "n_tiles": counts.sum(1)})
    for c in range(counts.shape[1]):
        out[f"prop_{c}"] = counts[:, c] / counts.sum(1)
    return out


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

    def sample(self, cluster: int, n: int, rng: np.random.Generator) -> np.ndarray:
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

    def sample(self, cluster: int, n: int, rng: np.random.Generator) -> np.ndarray:
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

    def parameters(self) -> dict:
        p = {"covariance": self.covariance, "reg": self.reg, "min_tiles": self.min_tiles,
             "mean": torch.from_numpy(self.mean.astype(np.float32)),
             "variance": torch.from_numpy(self.var.astype(np.float32)),
             "small_cluster": torch.from_numpy(self.small)}
        if self.chol is not None:
            p["cholesky"] = torch.from_numpy(self.chol.astype(np.float32))
        return p


GENERATORS = {"gaussian": GaussianGenerator}


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
    """Instantiate a registered generation method (only "gaussian" exists)."""
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

    # STEP 3: Sample per cluster, then shuffle
    parts = [generator.sample(c, int(n), rng) for c, n in enumerate(counts) if n > 0]
    cluster = np.repeat(np.arange(len(counts)), counts)
    X = np.concatenate(parts)
    order = rng.permutation(size)
    return {"X": X[order], "cluster": cluster[order], "counts": counts, "n_templates": m,
            "max_template_weight": float(w.max())}


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
                           settings: dict, out_dir: Path | None, overwrite: bool = False) -> dict:
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

    Returns a dict with wsis (DataFrame), distribution (DataFrame), scatter (class -> real/synthetic PC1-PC2),
    folder, and written (number of files).
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
    rows, dist, scatter = [], [], {}
    start = time.perf_counter()
    for ci, (label, gen) in enumerate(generators.items()):
        rng = _rng(settings["seed"], dim, k, ci, 7)
        real = real_samples[label]
        syn_keep = []
        for i in range(settings["n_per_class"]):
            wsi = make_synthetic_wsi(gen, compositions[label], rng, settings["max_tiles"], settings.get("n_templates", 1),
                                     settings.get("template_concentration", 1.0))
            n = len(wsi["X"])
            check_wsi(wsi["X"], dim, n)
            reassigned = np.bincount(clusters[label].predict(wsi["X"]), minlength=len(wsi["counts"]))
            name = synthetic_id(method, dim, k, label, i + 1)
            rows.append({"WSI Filename": name, "Patient Id": name, "WSI label": label, "Set": SYNTHETIC_SET,
                         "method": method, "pca_dim": dim, "k": k, "n_tiles": n,
                         "n_templates": wsi["n_templates"], "max_template_weight": wsi["max_template_weight"],
                         "counts": wsi["counts"], "reassigned": reassigned})
            if i < 5:
                syn_keep.append(wsi["X"][rng.choice(n, size=min(n, 400), replace=False)])
            if folder is not None and settings["save_embeddings"]:
                _save_wsi(folder / f"{name}.pt", name, label, wsi["X"], settings)
        syn = np.concatenate(syn_keep)
        dist.append({"label": label, **distribution_stats(real, syn)})
        scatter[label] = {"real": real[:1000, :2], "synthetic": syn[:1000, :2]}
    seconds = time.perf_counter() - start

    # STEP 3: Tables
    wsis = pd.DataFrame(rows)
    return {"wsis": wsis, "distribution": pd.DataFrame(dist), "scatter": scatter, "folder": folder,
            "written": len(wsis) if folder is not None and settings["save_embeddings"] else 0,
            "seconds": seconds}


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
    meta = {key: value for key, value in settings.items() if key not in ("pca_dir",)}
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
                    "generator": generators[c].parameters()} for c, m in clusters.items()},
               folder / "cluster_models.pt")
    result["wsis"][LABEL_COLUMNS].to_csv(folder / "labels.csv", index=False)


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
