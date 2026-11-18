# IMPORTS *****************************************************************************************
# External libraries
from dataclasses import dataclass, field
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from scipy import stats as sps
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# Internal imports ********************************************************************************
from utils.cls_overview import INK_2, SERIES, STYLE
from utils.figure_captions import split_with_notes
from utils.synthetic_gaussian import _apportion, make_synthetic_wsi, sample_bag

# Constants ***************************************************************************************
LESION_CLASSES = ("ADH", "DCIS")                 # the classes that get lesion-share bags
REFERENCE_CLASSES = ("N", "PB", "UDH", "FEA")    # the Non-cancer classes of the 3-class problem: the benign background
SHARE_BOUNDS = (1e-4, 1 - 1e-4)                  # the lesion share is kept inside (0, 1) for the Beta fit and draw
COMPOSITION_MODES = ("blend", "lesion_share")


# Main functions: lesion clusters *****************************************************************
def lesion_enrichment(counts: dict[str, np.ndarray], lesion_classes: tuple[str, ...] = LESION_CLASSES, reference_classes: tuple[str, ...] = REFERENCE_CLASSES,
                      min_lesion_slides: int = 5, min_slide_tiles: int = 5, log2_threshold: float = 1.0, alpha: float = 0.05) -> pd.DataFrame:
    """
    The enrichment of every cluster of the shared codebook in the lesion slides (ADH, DCIS) against the Non-cancer slides.

    The unit is the slide: for every cluster the per-slide share of tiles (the slide's tiles in the cluster over all its tiles) of the
    lesion slides is compared with that of the reference slides. A cluster is a **lesion cluster** when (i) at least `min_lesion_slides`
    lesion slides have at least `min_slide_tiles` tiles in it (so that it is not the signature of one slide), (ii) the log2 ratio of the mean
    per-slide share in the lesion slides to that in the reference slides (with a small pseudo-count of 0.1 / K) is at least `log2_threshold`
    (1 = twice as frequent), and (iii) a one-sided Mann-Whitney U test of the per-slide shares (lesion greater than reference), corrected over the
    eligible clusters by the Benjamini-Hochberg procedure (Benjamini and Hochberg 1995), has q <= alpha. The test uses slides, not tiles, as the unit.

    * Arguments:
    counts: class -> per-slide tile counts [slides, K] of the shared codebook (collect_class_statistics)
    lesion_classes: The lesion classes
    reference_classes: The reference (Non-cancer) classes that are present in `counts`
    min_lesion_slides: The minimum number of lesion slides that support a cluster
    min_slide_tiles: The tiles a slide needs in a cluster to support it
    log2_threshold: The minimum log2 enrichment
    alpha: The false discovery rate of the test

    Returns one row per cluster: cluster, lesion_slides, reference_slides (slides that support it), mean_share_lesion, mean_share_reference,
    log2_enrichment, p, q (NaN when not eligible), eligible and lesion.
    """
    lesion = np.concatenate([counts[c] for c in lesion_classes if c in counts])
    reference = np.concatenate([counts[c] for c in reference_classes if c in counts])
    if len(lesion) == 0 or len(reference) == 0:
        raise ValueError("Both lesion and reference slides are needed.")
    share_l, share_r = lesion / lesion.sum(1, keepdims=True), reference / reference.sum(1, keepdims=True)
    k = lesion.shape[1]
    pseudo = 0.1 / k
    supporting = (lesion >= min_slide_tiles).sum(0)
    mean_l, mean_r = share_l.mean(0), share_r.mean(0)
    table = pd.DataFrame({"cluster": np.arange(k), "lesion_slides": supporting, "reference_slides": (reference >= min_slide_tiles).sum(0),
                          "mean_share_lesion": mean_l, "mean_share_reference": mean_r,
                          "log2_enrichment": np.log2((mean_l + pseudo) / (mean_r + pseudo))})
    table["eligible"] = table["lesion_slides"] >= min_lesion_slides
    p = np.full(k, np.nan)
    for c in np.flatnonzero(table["eligible"]):
        try:
            p[c] = sps.mannwhitneyu(share_l[:, c], share_r[:, c], alternative="greater").pvalue
        except ValueError:                                 # all values identical
            p[c] = 1.0
    table["p"], table["q"] = p, _benjamini_hochberg(p)
    table["lesion"] = table["eligible"] & (table["log2_enrichment"] >= log2_threshold) & (table["q"] <= alpha)

    return table


def _benjamini_hochberg(p: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg adjusted p-values of the finite entries of p (NaN stays NaN)."""
    q = np.full(len(p), np.nan)
    ok = np.flatnonzero(np.isfinite(p))
    if len(ok) == 0:
        return q
    order = ok[np.argsort(p[ok])]
    adjusted = np.minimum.accumulate((p[order] * len(ok) / np.arange(1, len(ok) + 1))[::-1])[::-1]
    q[order] = np.clip(adjusted, 0, 1)

    return q


# Main functions: lesion-share model **************************************************************
@dataclass
class LesionShareModel:
    """
    The composition model of one lesion class (ADH or DCIS): lesion share, size, lesion mix and benign background.

    A bag is built in four draws: the lesion share lambda from a Beta distribution fitted to the lambda of the real slides of the class, the
    tile count from a lognormal fitted to their tile counts (independently of lambda; the fitted values and the empirical correlation are reported by
    lesion_share_table), the proportions inside the lesion clusters (summing to lambda) from a Dirichlet blend of the real lesion mixes of the class,
    and the proportions of the other clusters (summing to 1 - lambda) from a Dirichlet blend of Non-cancer slides whose lesion clusters were removed and
    that were renormalised. The tiles are then sampled by the generator of the class.

    * Arguments:
    label: The class
    lesion_mask: Boolean [K], the lesion clusters
    beta: (a, b) of the lesion-share distribution
    lognormal: (mu, sigma) of the log tile count
    tile_range: (lowest, highest) tile count that is allowed
    lesion_mixes: [M, K] real lesion mixes (zero outside the lesion clusters, rows sum to 1)
    benign: [T, K] Non-cancer templates (zero in the lesion clusters, rows sum to 1)
    n_lesion_templates: The lesion mixes blended per bag
    n_benign_templates: The benign templates blended per bag
    concentration: The Dirichlet concentration of both blends
    max_tiles: The cap of the bag size (None = no cap)
    """
    label: str
    lesion_mask: np.ndarray
    beta: tuple[float, float]
    lognormal: tuple[float, float]
    tile_range: tuple[float, float]
    lesion_mixes: np.ndarray
    benign: np.ndarray
    n_lesion_templates: int = 3
    n_benign_templates: int = 5
    concentration: float = 1.0
    max_tiles: int | None = None
    real_share: np.ndarray = field(default_factory=lambda: np.zeros(0))
    real_tiles: np.ndarray = field(default_factory=lambda: np.zeros(0))

    def _blend(self, rng: np.random.Generator, pool: np.ndarray, m: int) -> tuple[np.ndarray, float]:
        """A Dirichlet blend of m distinct rows of pool, and the largest weight."""
        m = int(min(max(m, 1), len(pool)))
        if m == 1:
            return pool[int(rng.integers(len(pool)))], 1.0
        w = rng.dirichlet(np.full(m, float(self.concentration)))
        return w @ pool[rng.choice(len(pool), size=m, replace=False)], float(w.max())

    def make(self, generator, rng: np.random.Generator) -> dict:
        """One synthetic bag of the class (the keys of make_synthetic_wsi, and lesion_share: the share of the tiles drawn for lesion clusters)."""
        lam = float(np.clip(rng.beta(*self.beta), *SHARE_BOUNDS))
        size = float(np.exp(rng.normal(*self.lognormal)))
        size = max(1, int(round(float(np.clip(size, *self.tile_range)))))
        size = size if self.max_tiles is None else min(size, int(self.max_tiles))
        lesion_part, w_lesion = self._blend(rng, self.lesion_mixes, self.n_lesion_templates)
        benign_part, w_benign = self._blend(rng, self.benign, self.n_benign_templates)
        props = lam * lesion_part + (1 - lam) * benign_part
        counts = _apportion(size, props / props.sum())
        wsi = sample_bag(generator, counts, rng)
        wsi.update({"n_templates": int(min(self.n_lesion_templates, len(self.lesion_mixes)) + min(self.n_benign_templates, len(self.benign))),
                    "max_template_weight": float(max(lam * w_lesion, (1 - lam) * w_benign)),
                    "lesion_share": float(counts[self.lesion_mask].sum() / counts.sum())})
        return wsi

    def to_saved(self) -> dict:
        """The model as tensors and numbers (the `lesion` key of cluster_models.pt)."""
        return {"lesion_mask": torch.from_numpy(self.lesion_mask.astype(np.uint8)), "beta": [float(x) for x in self.beta],
                "lognormal": [float(x) for x in self.lognormal], "tile_range": [float(x) for x in self.tile_range],
                "lesion_mixes": torch.from_numpy(self.lesion_mixes.astype(np.float32)), "benign": torch.from_numpy(self.benign.astype(np.float32)),
                "n_lesion_templates": int(self.n_lesion_templates), "n_benign_templates": int(self.n_benign_templates),
                "concentration": float(self.concentration)}


def fit_lesion_models(counts: dict[str, np.ndarray], compositions: dict[str, pd.DataFrame], lesion_mask: np.ndarray, lesion_classes: tuple[str, ...] = LESION_CLASSES,
                      reference_classes: tuple[str, ...] = REFERENCE_CLASSES, min_lesion_tiles: int = 5, n_lesion_templates: int = 3, n_benign_templates: int = 5,
                      concentration: float = 1.0, max_tiles: int | None = None) -> dict[str, LesionShareModel]:
    """
    Fit the LesionShareModel of every lesion class from the real Training slides.

    * Arguments:
    counts: class -> per-slide tile counts [slides, K] (collect_class_statistics)
    compositions: class -> composition_table (columns n_tiles, prop_<c>); the rows are the slides of `counts`
    lesion_mask: The lesion clusters (lesion_enrichment)
    lesion_classes: The classes to model
    reference_classes: The Non-cancer classes that give the benign background
    min_lesion_tiles: A lesion mix is taken from a slide with at least this many tiles in the lesion clusters
    n_lesion_templates, n_benign_templates, concentration, max_tiles: See LesionShareModel

    Returns {class: LesionShareModel}.
    """
    if not lesion_mask.any():
        raise ValueError("No lesion cluster was found: lower the enrichment threshold or the minimum number of slides, or change k.")

    # STEP 1: The benign background: Non-cancer slides without their lesion clusters, renormalised
    pool = np.concatenate([counts[c] for c in reference_classes if c in counts]).astype(np.float64)
    pool[:, lesion_mask] = 0
    pool = pool[pool.sum(1) > 0]
    benign = pool / pool.sum(1, keepdims=True)

    # STEP 2: Lesion share, size and lesion mixes of every lesion class
    models = {}
    for label in lesion_classes:
        c = counts[label].astype(np.float64)
        tiles = c.sum(1)
        share = np.clip(c[:, lesion_mask].sum(1) / tiles, *SHARE_BOUNDS)
        a, b, _, _ = sps.beta.fit(share, floc=0, fscale=1)
        mu, sigma = float(np.log(tiles).mean()), float(np.log(tiles).std(ddof=1)) if len(tiles) > 1 else 0.0
        has_lesion = c[:, lesion_mask].sum(1) >= min_lesion_tiles
        mixes = np.zeros_like(c[has_lesion])
        mixes[:, lesion_mask] = c[has_lesion][:, lesion_mask]
        mixes /= mixes.sum(1, keepdims=True)
        models[label] = LesionShareModel(label, lesion_mask.copy(), (float(a), float(b)), (mu, sigma), (0.5 * float(tiles.min()), 2.0 * float(tiles.max())),
                                         mixes, benign, n_lesion_templates, n_benign_templates, concentration, max_tiles, share, tiles)
        if len(mixes) == 0:
            raise ValueError(f"No {label} slide has {min_lesion_tiles} tiles in the lesion clusters.")

    return models


def lesion_share_table(models: dict[str, LesionShareModel]) -> pd.DataFrame:
    """The fitted lesion-share (Beta) and tile-count (lognormal) distributions against the real slides, one row per class."""
    rows = []
    for label, m in models.items():
        a, b = m.beta
        rows.append({"label": label, "slides": len(m.real_share), "share_mean_real": float(m.real_share.mean()), "share_mean_beta": a / (a + b),
                     "share_sd_real": float(m.real_share.std(ddof=1)), "share_sd_beta": float(np.sqrt(a * b / ((a + b) ** 2 * (a + b + 1)))),
                     "beta_a": a, "beta_b": b, "ks_beta": float(sps.kstest(m.real_share, sps.beta(a, b).cdf).statistic),
                     "tiles_median_real": float(np.median(m.real_tiles)), "tiles_median_lognormal": float(np.exp(m.lognormal[0])), "lognormal_sigma": m.lognormal[1],
                     "ks_lognormal": float(sps.kstest(np.log(m.real_tiles), sps.norm(*m.lognormal).cdf).statistic),
                     "corr_share_log_tiles": float(np.corrcoef(m.real_share, np.log(m.real_tiles))[0, 1]) if len(m.real_share) > 2 else np.nan,
                     "lesion_mixes": len(m.lesion_mixes), "benign_templates": len(m.benign)})
    return pd.DataFrame(rows)


# Main functions: diagnostics *********************************************************************
def lesion_share_comparison(models: dict[str, LesionShareModel], wsis: pd.DataFrame, classes: list[str]) -> pd.DataFrame:
    """
    Real against synthetic lesion share and tile count per class, from the real slides and the bags that were generated.

    The synthetic lesion share is the share of the tiles that the clusters of the generator's bag fall in, after the bag was
    assigned again to the nearest cluster centre (the `reassigned` counts), like the real slides.

    * Arguments:
    models: The output of fit_lesion_models (the real lesion share and tile counts are stored in it)
    wsis: The wsis table of generate_configuration (columns WSI label, n_tiles, reassigned)
    classes: The classes to report (those without a lesion model are skipped)

    Returns one row per class and source with the quantiles of the lesion share and of the tile count and the Kolmogorov-Smirnov distance between real and synthetic.
    """
    rows = []
    for label in classes:
        if label not in models:
            continue
        m = models[label]
        sub = wsis[wsis["WSI label"] == label]
        counts = np.stack(sub["reassigned"].to_numpy()).astype(np.float64)
        share = counts[:, m.lesion_mask].sum(1) / counts.sum(1)
        tiles = sub["n_tiles"].to_numpy(dtype=np.float64)
        for source, s, t in (("real", m.real_share, m.real_tiles), ("synthetic", share, tiles)):
            rows.append({"label": label, "source": source, "bags": len(s), **{f"share_q{int(q * 100)}": float(np.quantile(s, q)) for q in (0.1, 0.5, 0.9)},
                         **{f"tiles_q{int(q * 100)}": float(np.quantile(t, q)) for q in (0.1, 0.5, 0.9)},
                         "ks_share_vs_real": float(sps.ks_2samp(s, m.real_share).statistic), "ks_tiles_vs_real": float(sps.ks_2samp(t, m.real_tiles).statistic)})
    return pd.DataFrame(rows)


def proportion_correlation(real: dict[str, np.ndarray], synthetic: dict[str, np.ndarray]) -> pd.DataFrame:
    """
    How well the synthetic bags reproduce the cluster proportions of the real slides, per class.

    Two Pearson correlations over clusters: of the mean proportion vectors (the average composition), and of the entries of the
    cluster-by-cluster correlation matrices across slides or bags (the co-occurrence structure; clusters without variance on
    either side are left out).

    * Arguments:
    real: class -> real slide proportions [slides, K]
    synthetic: class -> synthetic bag proportions [bags, K] (same clusters)
    """
    rows = []
    for label, r in real.items():
        s = synthetic[label]
        mean_r = np.corrcoef(r.mean(0), s.mean(0))[0, 1]
        ok = (r.std(0) > 0) & (s.std(0) > 0)
        cr, cs = np.corrcoef(r[:, ok].T), np.corrcoef(s[:, ok].T)
        upper = np.triu_indices(int(ok.sum()), 1)
        rows.append({"label": label, "slides": len(r), "bags": len(s), "corr_mean_proportions": float(mean_r),
                     "corr_cluster_correlations": float(np.corrcoef(cr[upper], cs[upper])[0, 1]) if len(upper[0]) > 2 else np.nan})
    return pd.DataFrame(rows)


def bag_summaries(mean_embedding: np.ndarray, proportions: np.ndarray) -> np.ndarray:
    """The summary of a bag for the bag-level classifier: its mean embedding and its cluster proportions, side by side [bags, D + K]."""
    return np.hstack([np.asarray(mean_embedding, dtype=np.float64), np.asarray(proportions, dtype=np.float64)])


def real_bag_summaries(stats: dict, composition: pd.DataFrame) -> np.ndarray:
    """The summaries (mean embedding and proportions) of the real Training slides of a class, from the statistics of collect_class_statistics (no file is read)."""
    counts, slide_mean = stats["counts"].astype(np.float64), stats["slide_mean"]
    mean = (counts[:, :, None] * slide_mean).sum(1) / counts.sum(1, keepdims=True)
    return bag_summaries(mean, counts / counts.sum(1, keepdims=True))


def blend_summaries(generator, composition: pd.DataFrame, model, rng: np.random.Generator, n: int, n_templates: int, concentration: float,
                    max_tiles: int | None) -> np.ndarray:
    """The summaries of n bags of the template-blending mode (the reference of the bag-level AUC); proportions from the nearest cluster centre."""
    rows = []
    for _ in range(n):
        wsi = make_synthetic_wsi(generator, composition, rng, max_tiles, n_templates, concentration)
        labels = model.predict(wsi["X"])
        rows.append(bag_summaries(wsi["X"].mean(0, keepdims=True), np.bincount(labels, minlength=len(wsi["counts"]))[None] / len(wsi["X"]))[0])
    return np.stack(rows)


def bag_level_auc(real: np.ndarray, synthetic: np.ndarray, seed: int, n_splits: int = 5, repeats: int = 5) -> dict:
    """
    How well a regularised classifier tells real bag summaries from synthetic ones (held-out ROC-AUC).

    The two sides are balanced: each repeat draws as many synthetic bags as there are real slides. A standardised, L2-regularised logistic
    regression (C = 0.1) is scored by stratified cross-validation; every slide and every bag is one row, so the folds are grouped by slide or bag.
    The classifier is deliberately simple, because a class has few real slides (a flexible classifier would memorise them). Compare the value with
    the real-vs-real reference of `bag_level_reference` (two disjoint halves of the real slides), which is the value that two real populations of this size obtain.

    * Arguments:
    real: The real summaries [n, F]
    synthetic: The synthetic summaries [m, F], m >= n
    seed: The random seed
    n_splits: The cross-validation folds
    repeats: The number of synthetic subsamples

    Returns auc (mean), auc_std (over repeats and folds), n_real, n_synthetic.
    """
    rng = np.random.default_rng(seed)
    n = len(real)
    scores = []
    for _ in range(repeats):
        pick = rng.choice(len(synthetic), min(n, len(synthetic)), replace=False)
        scores += _cv_auc(np.vstack([real, synthetic[pick]]), np.r_[np.zeros(n), np.ones(len(pick))].astype(int), seed, n_splits)
    return {"auc": float(np.mean(scores)), "auc_std": float(np.std(scores)), "n_real": n, "n_synthetic": int(len(synthetic))}


def bag_level_reference(real: np.ndarray, seed: int, n_splits: int = 5, repeats: int = 5) -> dict:
    """The real-vs-real reference of bag_level_auc: random disjoint halves of the real slides are told apart; a value near 0.5 is what identical populations give."""
    rng = np.random.default_rng(seed)
    scores = []
    for _ in range(repeats):
        order = rng.permutation(len(real))
        half = len(real) // 2
        scores += _cv_auc(real[order[:2 * half]], np.r_[np.zeros(half), np.ones(half)].astype(int), seed, n_splits)
    return {"auc": float(np.mean(scores)), "auc_std": float(np.std(scores)), "n_real": len(real), "n_synthetic": 0}


def _cv_auc(X: np.ndarray, y: np.ndarray, seed: int, n_splits: int) -> list[float]:
    """Held-out ROC-AUC of every fold of a stratified cross-validation of a standardised logistic regression."""
    splits = int(min(n_splits, np.bincount(y).min()))
    if splits < 2:
        raise ValueError("The bag-level AUC needs at least two bags on each side.")
    scores = []
    for fit, test in StratifiedKFold(splits, shuffle=True, random_state=seed).split(X, y):
        model = make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=2000)).fit(X[fit], y[fit])
        scores.append(roc_auc_score(y[test], model.predict_proba(X[test])[:, 1]))
    return scores


# Main functions: plots ***************************************************************************
def plot_lesion_fit(models: dict[str, LesionShareModel], wsis: pd.DataFrame | None = None, show: bool = True):
    """
    The fitted lesion-share and tile-count distributions against the real slides (and the generated bags when given), one row per lesion class.

    * Arguments:
    models: The output of fit_lesion_models
    wsis: The wsis table of generate_configuration (optional; adds the synthetic bags)
    show: Display the figure
    """
    labels = list(models)
    notes = [(0, 1, "Lesion share lambda of the real Training slides of each lesion class (the summed proportion of the tiles in the lesion clusters; bars, normalised to a density) "
                    "with the fitted Beta density (line) and, when generated, the lesion share of the synthetic bags (step line). With few slides the histogram is coarse and the "
                    "fit is indicative; the Kolmogorov-Smirnov distances are in the table."),
             (1, 2, "Tile count of the same slides on a logarithmic axis, with the fitted lognormal density (line) and the tile counts of the generated bags (step line). "
                    "The tile count is drawn independently of the lesion share; the correlation of the two in the real slides is reported in the table.")]
    with plt.rc_context(STYLE):
        fig = plt.figure(figsize=(11, 3.4 * len(labels) + 2.6))
        plots = split_with_notes(fig, notes, [1, 1], 11, plot_height=3.4 * len(labels), note_height=2.4)
        axes = np.atleast_2d(plots.subplots(len(labels), 2))
        for row, label in enumerate(labels):
            m = models[label]
            sub = wsis[wsis["WSI label"] == label] if wsis is not None else None
            a, b = axes[row]
            grid = np.linspace(0.001, 0.999, 300)
            a.hist(m.real_share, bins=np.linspace(0, 1, 11), density=True, color=SERIES[0], alpha=0.6, label="real")
            a.plot(grid, sps.beta.pdf(grid, *m.beta), color=SERIES[1], lw=2, label="Beta fit")
            if sub is not None:
                counts = np.stack(sub["reassigned"].to_numpy()).astype(np.float64)
                a.hist(counts[:, m.lesion_mask].sum(1) / counts.sum(1), bins=np.linspace(0, 1, 11), density=True, histtype="step", color=INK_2, lw=1.5, label="synthetic")
            a.set_title(f"{label}: lesion share"); a.set_xlabel("lesion share"); a.set_ylabel("density"); a.legend(frameon=False, fontsize=8)
            x = np.log(m.real_tiles)
            span = np.linspace(x.min() - 1, x.max() + 1, 300)
            b.hist(x, bins=10, density=True, color=SERIES[0], alpha=0.6)
            b.plot(span, sps.norm.pdf(span, *m.lognormal), color=SERIES[1], lw=2)
            if sub is not None:
                b.hist(np.log(sub["n_tiles"].to_numpy(dtype=np.float64)), bins=10, density=True, histtype="step", color=INK_2, lw=1.5)
            b.set_title(f"{label}: tile count"); b.set_xlabel("log tiles per slide")
        if show:
            plt.show()
    return fig
