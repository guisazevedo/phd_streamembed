# IMPORTS *****************************************************************************************
# External libraries
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

# Internal imports ********************************************************************************
from utils.cls_overview import INK_2, SERIES, STYLE, SURFACE, cluster_colors
from utils.figure_captions import split_with_notes
from utils.labels import BRACS_CLASSES, load_bracs_labels, load_bracs_roi_distribution

# Constants ***************************************************************************************
SETS = ["Training", "Validation", "Testing"]
SET_COLORS = dict(zip(SETS, SERIES))
# Diverging map for standardized residuals: blue (fewer than expected), neutral, orange (more than expected)
DIVERGING = LinearSegmentedColormap.from_list("div_blue_orange", ["#2a78d6", "#f2f1ee", "#eb6834"])


# Helper functions ********************************************************************************
def _contingency(rows: np.ndarray, cols: np.ndarray, n_rows: int, n_cols: int) -> np.ndarray:
    """Count table [n_rows, n_cols] of how often rows[i] co-occurs with cols[i]."""
    return np.bincount(rows * n_cols + cols, minlength=n_rows * n_cols).reshape(n_rows, n_cols).astype(float)


def _chi2(table: np.ndarray) -> float:
    """Pearson chi-square statistic of a count table."""
    expected = table.sum(1, keepdims=True) * table.sum(0, keepdims=True) / table.sum()
    return float(((table - expected) ** 2 / np.where(expected > 0, expected, 1)).sum())


def _patient_set(table: pd.DataFrame) -> pd.Series:
    """The set of each patient (the most frequent one when a patient's slides are in several sets)."""
    return table.groupby("patient")["set"].agg(lambda s: s.value_counts().index[0])


# Main functions **********************************************************************************
def build_slide_table(labels_file: str | Path, scan: pd.DataFrame | None = None) -> pd.DataFrame:
    """
    Build one tidy table with a row per labelled slide: slide, patient, label, set and, when given,
    the technical properties from scan_slides plus the total number of RoIs from the second sheet.

    * Arguments:
    labels_file: The path to BRACS.xlsx
    scan: The output of utils.slide_stats.scan_slides (None = labels only)
    """

    # STEP 1: Labels, in the order of the spreadsheet
    labels = load_bracs_labels(labels_file)
    table = pd.DataFrame([{"slide": k, **v} for k, v in labels.items()]).rename(columns={"label": "label"})
    table["label"] = pd.Categorical(table["label"], categories=BRACS_CLASSES, ordered=True)
    table["set"] = pd.Categorical(table["set"], categories=SETS, ordered=True)

    # STEP 2: Number of RoIs per slide (only some slides have a breakdown) and the technical properties
    roi = load_bracs_roi_distribution(labels_file)
    table["n_rois"] = table["slide"].map(lambda s: roi[s]["total"] if s in roi else np.nan)
    if scan is not None:
        table = table.merge(scan, on="slide", how="left")

    return table


def validate_labels(table: pd.DataFrame, data_dir: str | Path | None = None) -> dict:
    """
    Check the label table and print a short report: size, duplicates, unknown classes or sets, and
    agreement with the .svs files in data_dir.

    * Arguments:
    table: The output of build_slide_table
    data_dir: The folder with the .svs files (None = skip the file check)

    Returns a dict with the checks (name -> value or list of offending slides).
    """

    # STEP 1: Checks on the table itself
    checks = {
        "slides": len(table),
        "patients": table["patient"].nunique(),
        "duplicate slides": table["slide"][table["slide"].duplicated()].tolist(),
        "unknown class": table["slide"][table["label"].isna()].tolist(),
        "unknown set": table["slide"][table["set"].isna()].tolist(),
    }

    # STEP 2: Agreement with the files on disk
    if data_dir is not None:
        on_disk = {p.stem for p in Path(data_dir).glob("*.svs")}
        checks["labelled but no file"] = sorted(set(table["slide"]) - on_disk)
        checks["file but no label"] = sorted(on_disk - set(table["slide"]))

    # STEP 3: Report
    for name, value in checks.items():
        status = value if isinstance(value, int) else (f"{len(value)}" + (f"  {value[:5]}" if value else ""))
        print(f"{name:<22}: {status}")

    return checks


def split_audit_stats(table: pd.DataFrame, n_perm: int = 5000, seed: int = 0) -> dict:
    """
    Test whether the official split is stratified by class, and check it at patient level.

    The class-by-set association is measured with Pearson's chi-square and Cramer's V. The p-value comes
    from a permutation test that reassigns whole patients to sets (keeping the number of patients per
    set), because slides of one patient are not independent.

    * Arguments:
    table: The output of build_slide_table
    n_perm: The number of permutations
    seed: The random seed

    Returns a dict with the class-by-set count table, chi2, cramers_v, p_value, standardized residuals,
    and patient-level counts (patients_in_several_sets, patients_with_mixed_classes, slides_per_patient).
    """

    # STEP 1: Observed class-by-set table and standardized residuals (O - E) / sqrt(E)
    K, S = len(BRACS_CLASSES), len(SETS)
    cls = table["label"].cat.codes.to_numpy()
    sets = table["set"].cat.codes.to_numpy()
    observed = _contingency(cls, sets, K, S)
    expected = observed.sum(1, keepdims=True) * observed.sum(0, keepdims=True) / observed.sum()
    chi2 = _chi2(observed)
    cramers_v = float(np.sqrt(chi2 / (observed.sum() * (min(K, S) - 1))))

    # STEP 2: Permutation test at patient level
    rng = np.random.default_rng(seed)
    patient_ids, patient_index = np.unique(table["patient"].to_numpy(), return_inverse=True)
    patient_set = _patient_set(table).reindex(patient_ids).cat.codes.to_numpy() if hasattr(_patient_set(table), "cat") else \
        pd.Categorical(_patient_set(table).reindex(patient_ids), categories=SETS).codes
    permuted = np.array([_chi2(_contingency(cls, rng.permutation(patient_set)[patient_index], K, S)) for _ in range(n_perm)])
    p_value = float((1 + (permuted >= chi2).sum()) / (1 + n_perm))

    # STEP 3: Patient-level checks
    per_patient = table.groupby("patient")
    several_sets = per_patient["set"].agg(lambda s: s.nunique() > 1)
    mixed = per_patient["label"].agg(lambda s: s.nunique() > 1)

    return {
        "observed": observed, "expected": expected, "residuals": (observed - expected) / np.sqrt(np.where(expected > 0, expected, 1)),
        "chi2": chi2, "cramers_v": cramers_v, "p_value": p_value, "n_perm": n_perm,
        "patients": len(patient_ids),
        "patients_in_several_sets": several_sets[several_sets].index.tolist(),
        "patients_with_mixed_classes": int(mixed.sum()),
        "slides_per_patient": per_patient.size(),
        "classes_per_patient": per_patient["label"].nunique(),
    }


def plot_class_distribution(table: pd.DataFrame, show: bool = True) -> dict:
    """
    Show how many slides each class has, overall and within each official set.

    * Arguments:
    table: The output of build_slide_table
    show: Whether to call plt.show() (the figure is returned either way)

    Returns a dict with the figure and the class-by-set count table (DataFrame).
    """

    # STEP 1: Counts and within-set shares
    counts = pd.crosstab(table["label"], table["set"]).reindex(index=BRACS_CLASSES, columns=SETS, fill_value=0)
    share = counts / counts.sum() * 100
    print(counts.assign(total=counts.sum(1)).to_string())
    print("\nShare of each set (%):\n" + share.round(1).to_string())

    # STEP 2: Figure
    notes = [
        (0, 1, "Number of slides per class, split by the official set. The slide label is the highest grade among the regions of interest of the slide.\n"
               "The classes are unequally frequent: the commonest class (PB) has several times more slides than the rarest (FEA). "
               "Rare classes also have few slides in the Validation and Testing sets, so estimates for them are imprecise."),
        (1, 2, "Class composition of each set, in percent of the slides of that set. If the split had been stratified by class, the three bars of a class would be equal.\n"
               "The composition differs visibly between sets (for example, class N is a small share of Training but a larger share of Validation); whether the "
               "differences exceed what a random assignment of patients would produce is tested in Section 3.3. With sets of 65 and 87 slides, a few slides "
               "more or fewer change the percentages of the rarer classes considerably, so results for those classes carry wide uncertainty."),
    ]
    x = np.arange(len(BRACS_CLASSES))
    with plt.rc_context(STYLE):
        fig = plt.figure(figsize=(15, 6.2), layout="constrained")
        fig.suptitle(f"Class distribution · {len(table)} slides, {table['patient'].nunique()} patients", fontsize=14, weight="bold", x=0.01, ha="left")
        top = split_with_notes(fig, notes, [1, 1], 15, plot_height=3.6, note_height=1.5)
        ax_a, ax_b = top.subplots(1, 2)

        bottom = np.zeros(len(x))
        for s in SETS:
            ax_a.bar(x, counts[s], bottom=bottom, color=SET_COLORS[s], edgecolor=SURFACE, linewidth=1.5, width=0.7, label=s)
            bottom += counts[s].to_numpy()
        for i, total in enumerate(bottom):
            ax_a.text(i, total, f"{int(total)}", ha="center", va="bottom", fontsize=8, color=INK_2)
        ax_a.set_xticks(x); ax_a.set_xticklabels(BRACS_CLASSES); ax_a.set_ylabel("slides"); ax_a.set_ylim(0, bottom.max() * 1.12)
        ax_a.set_title("How many slides per class?"); ax_a.legend(frameon=False)

        width = 0.8 / len(SETS)
        for j, s in enumerate(SETS):
            ax_b.bar(x - 0.4 + width * (j + 0.5), share[s], width=width, color=SET_COLORS[s], edgecolor=SURFACE, linewidth=1, label=f"{s} (n = {int(counts[s].sum())})")
        ax_b.axhline(100 / len(BRACS_CLASSES), color=INK_2, ls="--", lw=1)
        ax_b.set_xticks(x); ax_b.set_xticklabels(BRACS_CLASSES); ax_b.set_ylabel("% of the slides of the set")
        ax_b.set_title("Is the class mix the same in every set?"); ax_b.legend(frameon=False)

    if show:
        plt.show()

    return {"figure": fig, "counts": counts, "share": share}


def plot_split_audit(table: pd.DataFrame, n_perm: int = 5000, seed: int = 0, show: bool = True) -> dict:
    """
    Audit the official Training/Validation/Testing split: class stratification and patient-level checks.

    * Arguments:
    table: The output of build_slide_table
    n_perm: The number of patient-level permutations for the p-value
    seed: The random seed
    show: Whether to call plt.show() (the figure is returned either way)

    Returns the dict of split_audit_stats plus the figure.
    """

    # STEP 1: Statistics
    stats = split_audit_stats(table, n_perm, seed)
    leaks = stats["patients_in_several_sets"]
    print(f"Class x set : chi2 = {stats['chi2']:.1f}, Cramer's V = {stats['cramers_v']:.2f}, patient-level permutation p = {stats['p_value']:.3f} ({n_perm} permutations)")
    print(f"Patients    : {stats['patients']}  |  with slides in several sets: {len(leaks)} {leaks}  |  with slides of several classes: {stats['patients_with_mixed_classes']}")
    for patient in leaks:
        print(f"  patient {patient}: " + ", ".join(f"{s} {n}" for s, n in table[table['patient'] == patient]['set'].value_counts().items() if n))

    # STEP 2: Figure
    notes = [
        (0, 1, f"Standardized residuals (observed − expected) / √expected of the class-by-set count table; expected counts assume the class mix is the same in every set. "
               f"Orange, more slides than expected; blue, fewer. Numbers are the observed slide counts. Pearson χ² = {stats['chi2']:.1f}, Cramér's V = {stats['cramers_v']:.2f}, "
               f"p = {stats['p_value']:.3f} (permutation test that reassigns whole patients, {n_perm:,} permutations).\n"
               "Residuals beyond about ±2 mark cells that depart noticeably from a proportional split. A non-significant test is not proof of stratification: "
               "with these sample sizes only large departures are detectable, and individual cells of the small Validation and Testing sets can still differ visibly."),
        (1, 2, "Number of different classes among the slides of each patient. A patient can have slides of several diagnoses (a slide is labelled with its highest-grade region), "
               "so the label is a property of the slide, not of the patient.\n"
               "Consequences: patient-level cross-validation must group slides by patient id, and class counts per patient are not independent of the patient's other slides."),
        (0, 1, "Number of slides per patient. Most patients contribute one to a few slides; a small number contribute many.\n"
               "Patients with many slides weigh more in training and in any slide-level statistic, and their slides are correlated (same case, same staining batch). "
               "Splits and cross-validation must keep all slides of a patient together."),
        (1, 2, (f"Patients and slides per set. The official split keeps patients disjoint, with {len(leaks)} exception"
                f"{'s' if len(leaks) != 1 else ''}: patient{'s' if len(leaks) != 1 else ''} {', '.join(leaks) if leaks else '-'} "
                "ha" + ("ve" if len(leaks) != 1 else "s") + " slides in more than one set.\n"
                "The overlap is small, and the official assignment is kept for comparability with published results; it should be mentioned when reporting results, "
                "and patient-grouped folds should be used for any model selection inside the Training set.")),
    ]
    with plt.rc_context(STYLE):
        fig = plt.figure(figsize=(16, 11.5), layout="constrained")
        fig.suptitle("Audit of the official split", fontsize=14, weight="bold", x=0.01, ha="left")
        rows = fig.subfigures(2, 1)

        top = split_with_notes(rows[0], notes[:2], [1.2, 1], 16, plot_height=3.6, note_height=2.3)
        ax_r, ax_c = top.subplots(1, 2, width_ratios=[1.2, 1])
        residuals, observed = stats["residuals"], stats["observed"]
        im = ax_r.imshow(residuals, cmap=DIVERGING, vmin=-3, vmax=3, aspect="auto")
        ax_r.set_xticks(range(len(SETS))); ax_r.set_xticklabels(SETS); ax_r.set_yticks(range(len(BRACS_CLASSES))); ax_r.set_yticklabels(BRACS_CLASSES); ax_r.grid(False)
        for i in range(len(BRACS_CLASSES)):
            for j in range(len(SETS)):
                ax_r.text(j, i, f"{int(observed[i, j])}", ha="center", va="center", fontsize=9, color=INK_2)
        top.colorbar(im, ax=ax_r, shrink=0.8, label="standardized residual")
        ax_r.set_title("Does each set have its share of each class?")

        classes_per_patient = stats["classes_per_patient"].value_counts().sort_index()
        ax_c.bar(classes_per_patient.index, classes_per_patient.values, color=SERIES[0], edgecolor=SURFACE, linewidth=1.5, width=0.7)
        for k, v in classes_per_patient.items():
            ax_c.text(k, v, f"{v}", ha="center", va="bottom", fontsize=8, color=INK_2)
        ax_c.set_xticks(classes_per_patient.index); ax_c.set_xlabel("different classes among a patient's slides"); ax_c.set_ylabel("patients")
        ax_c.set_title("Do a patient's slides share one class?")

        bottom = split_with_notes(rows[1], notes[2:], [1.2, 1], 16, plot_height=3.6, note_height=2.3)
        ax_s, ax_p = bottom.subplots(1, 2, width_ratios=[1.2, 1])
        per_patient = stats["slides_per_patient"]
        values = np.arange(per_patient.min(), per_patient.max() + 1)
        ax_s.bar(values, [(per_patient == v).sum() for v in values], color=SERIES[0], edgecolor=SURFACE, linewidth=1.5, width=0.7)
        ax_s.set_xlabel("slides per patient"); ax_s.set_ylabel("patients"); ax_s.set_title("How many slides does a patient contribute?")

        patients_per_set = table.groupby("set", observed=False)["patient"].nunique().reindex(SETS)
        slides_per_set = table.groupby("set", observed=False).size().reindex(SETS)
        x = np.arange(len(SETS))
        ax_p.bar(x - 0.2, slides_per_set, width=0.4, color=SERIES[0], edgecolor=SURFACE, linewidth=1, label="slides")
        ax_p.bar(x + 0.2, patients_per_set, width=0.4, color=SERIES[1], edgecolor=SURFACE, linewidth=1, label="patients")
        for xi, v in zip(x - 0.2, slides_per_set): ax_p.text(xi, v, f"{v}", ha="center", va="bottom", fontsize=8, color=INK_2)
        for xi, v in zip(x + 0.2, patients_per_set): ax_p.text(xi, v, f"{v}", ha="center", va="bottom", fontsize=8, color=INK_2)
        ax_p.set_xticks(x); ax_p.set_xticklabels(SETS); ax_p.set_ylabel("count"); ax_p.legend(frameon=False)
        ax_p.set_title("How large is each set?")

    if show:
        plt.show()

    return {**stats, "figure": fig}


def plot_roi_composition(table: pd.DataFrame, labels_file: str | Path, show: bool = True) -> dict:
    """
    Show what the regions of interest (RoIs) inside slides of each slide-level class look like.

    A slide is labelled with its highest-grade RoI, but it also contains RoIs of other classes. This
    figure quantifies that mixture (only slides with an RoI breakdown in the second sheet are used).

    * Arguments:
    table: The output of build_slide_table
    labels_file: The path to BRACS.xlsx
    show: Whether to call plt.show() (the figure is returned either way)

    Returns a dict with the figure and the mean RoI share (DataFrame: slide class x RoI class).
    """

    # STEP 1: Share of each RoI class inside every slide, then the mean per slide class
    roi = load_bracs_roi_distribution(labels_file)
    slides = table[table["slide"].isin(roi)].copy()
    shares = np.array([[roi[s][c] / max(roi[s]["total"], 1) for c in BRACS_CLASSES] for s in slides["slide"]])
    mean_share = pd.DataFrame(shares, columns=BRACS_CLASSES, index=slides["label"].to_numpy()).groupby(level=0, observed=False).mean().reindex(BRACS_CLASSES) * 100
    n_per_class = slides["label"].value_counts().reindex(BRACS_CLASSES)
    print(f"{len(slides)} of {len(table)} slides have an RoI breakdown; RoIs per slide: median {slides['n_rois'].median():.0f}, range {slides['n_rois'].min():.0f} to {slides['n_rois'].max():.0f}")
    print("Mean RoI share (%) by slide class (rows) and RoI class (columns):\n" + mean_share.round(1).to_string())

    # STEP 2: Figure
    notes = [
        (0, 1, "Mean share of the regions of interest of each class (columns) inside slides of each slide-level class (rows), in percent; rows sum to 100%. "
               "Only slides with an RoI breakdown in the second sheet are used (n per class on the left).\n"
               "The diagonal is the share of RoIs that actually have the slide's label. Large off-diagonal values show that a slide label summarises a mixture: "
               "for example, slides labelled IC contain many benign regions. Labels at slide level are therefore weak labels for tiles, and tile-level accuracy "
               "against the slide label has a ceiling below 100%."),
        (1, 2, "Number of regions of interest per slide, by slide-level class (box: quartiles; line: median; points: slides).\n"
               "The number of RoIs reflects how many regions the annotators outlined, not only tissue size. Differences between classes show that the amount of "
               "annotation is correlated with the label, so RoI counts must not be used as an input feature, and RoI-level and slide-level analyses should not be mixed without care."),
    ]
    colors = cluster_colors(len(BRACS_CLASSES))
    with plt.rc_context(STYLE):
        fig = plt.figure(figsize=(15, 7), layout="constrained")
        fig.suptitle(f"Composition of the slide labels · {len(slides)} slides with RoI breakdown", fontsize=14, weight="bold", x=0.01, ha="left")
        top = split_with_notes(fig, notes, [1.2, 1], 15, plot_height=4.0, note_height=2.4)
        ax_h, ax_b = top.subplots(1, 2, width_ratios=[1.2, 1])

        from utils.cls_overview import SEQ
        im = ax_h.imshow(mean_share.to_numpy(), cmap=SEQ, vmin=0, vmax=100, aspect="auto")
        ax_h.set_xticks(range(len(BRACS_CLASSES))); ax_h.set_xticklabels(BRACS_CLASSES)
        ax_h.set_yticks(range(len(BRACS_CLASSES))); ax_h.set_yticklabels([f"{c} (n = {int(n_per_class[c])})" for c in BRACS_CLASSES]); ax_h.grid(False)
        for i in range(len(BRACS_CLASSES)):
            for j in range(len(BRACS_CLASSES)):
                v = mean_share.iloc[i, j]
                ax_h.text(j, i, f"{v:.0f}", ha="center", va="center", fontsize=8, color="white" if v > 55 else INK_2)
        top.colorbar(im, ax=ax_h, shrink=0.8, label="% of the RoIs")
        ax_h.set_xlabel("class of the RoI"); ax_h.set_ylabel("class of the slide"); ax_h.set_title("What do slides of each class contain?")

        rng = np.random.default_rng(0)
        data = [slides.loc[slides["label"] == c, "n_rois"].to_numpy() for c in BRACS_CLASSES]
        bp = ax_b.boxplot(data, positions=range(len(BRACS_CLASSES)), widths=0.55, patch_artist=True, showfliers=False,
                          medianprops={"color": INK_2, "lw": 1.5}, whiskerprops={"color": INK_2}, capprops={"color": INK_2})
        for patch, color in zip(bp["boxes"], colors):
            patch.set_facecolor(color); patch.set_alpha(0.35); patch.set_edgecolor(INK_2)
        for i, values in enumerate(data):
            ax_b.scatter(i + rng.uniform(-0.18, 0.18, len(values)), values, s=8, color=colors[i], alpha=0.7, linewidths=0)
        ax_b.set_xticks(range(len(BRACS_CLASSES))); ax_b.set_xticklabels(BRACS_CLASSES); ax_b.set_ylabel("RoIs per slide")
        ax_b.set_title("How many regions does a slide have?")

    if show:
        plt.show()

    return {"figure": fig, "mean_share": mean_share}


def plot_slide_properties(table: pd.DataFrame, tile_size: int = 224, show: bool = True) -> dict:
    """
    Summarise the technical properties of the slides and screen them for batch effects.

    Needs the columns added by slide_stats.scan_slides. A property that differs between classes or sets
    could be learned by a classifier instead of the pathology (confounding).

    * Arguments:
    table: The output of build_slide_table with the scan columns
    tile_size: The tile edge in pixels, used for the upper bound on the number of tiles per level
    show: Whether to call plt.show() (the figure is returned either way)

    Returns a dict with the figure and the statistics shown in the captions.
    """

    # STEP 1: Statistics for the captions
    levels = range(1, 5)
    year = table["scan_date"].dt.year
    class_by_year = pd.crosstab(table["label"], year).reindex(BRACS_CLASSES, fill_value=0)
    set_by_year = pd.crosstab(table["set"], year).reindex(SETS, fill_value=0)
    cls_codes, year_codes = table["label"].cat.codes.to_numpy(), pd.factorize(year, sort=True)[0]
    tab = _contingency(cls_codes, year_codes, len(BRACS_CLASSES), year_codes.max() + 1)
    v_year = float(np.sqrt(_chi2(tab) / (tab.sum() * (min(tab.shape) - 1))))
    n_levels = table["n_levels"].value_counts().sort_index()
    print(f"Scanner(s)       : {table['scanner'].value_counts().to_dict()}")
    print(f"Magnification    : {table['app_mag'].value_counts().to_dict()}   JPEG quality: {table['jpeg_quality'].value_counts().to_dict()}")
    print(f"MPP (um/pixel)   : {table['mpp'].value_counts().to_dict()}")
    print(f"Scan years       : {year.value_counts().sort_index().to_dict()}   (class x year Cramer's V = {v_year:.2f})")
    print(f"Pyramid levels   : {n_levels.to_dict()}   |   downsample at level 4: {table['downsample_4'].round().value_counts().to_dict()}")
    print(f"Full resolution  : median {table['gigapixels'].median():.1f} gigapixels, max {table['gigapixels'].max():.1f}   |   file size median {table['file_gb'].median():.2f} GB, total {table['file_gb'].sum():.0f} GB")

    # STEP 2: Figure
    notes = [
        (0, 1, "Size of the slide at full resolution in gigapixels, by slide-level class (box: quartiles; line: median; points: slides).\n"
               "Slides are large (a median of several gigapixels), which is why tiles are taken from a downsampled pyramid level. "
               "Systematic differences in size between classes would let a classifier use slide size as a shortcut."),
        (1, 2, f"Number of slides by downsample factor of pyramid levels 2 to 4 relative to level 1. Levels 2 and 3 are almost always 4× and 16×; level 4 is 32× or 64× "
               f"depending on the slide, and {int((table['n_levels'] < 4).sum())} slides have no level 4.\n"
               "Because the factor differs, a tile at level 4 covers a different tissue area in different slides; this should be remembered when level 4 is used."),
        (0, 1, f"Slides by microns per pixel at full resolution (MPP), coloured by set. All slides share one scanner and magnification "
               f"({', '.join(sorted(table['scanner'].dropna().unique()))}, {', '.join(f'{m:g}×' for m in sorted(table['app_mag'].dropna().unique()))}); "
               f"MPP takes {table['mpp'].nunique()} values.\n"
               "A narrow MPP range means that tiles from different slides cover the same physical size, which supports comparing them directly."),
        (1, 2, f"Slides by year of scanning, coloured by set. Scanning date is a proxy for staining and handling batches.\n"
               f"The association between class and scan year is Cramér's V = {v_year:.2f}. Values near 0 mean that batch and class are unrelated; "
               "larger values mean that a classifier could partly predict the class from batch effects, so batch should be checked after feature extraction (slide-effect analysis)."),
        (0, 1, f"Upper bound on the number of non-overlapping {tile_size} px tiles per slide at each pyramid level (whole slide rectangle, before removing background); box: quartiles; logarithmic axis.\n"
               "The number of tiles grows about 16-fold from one level to the next finer one, and so do the time and disk needed for feature extraction. "
               "The real count is lower because background tiles are filtered out."),
        (1, 2, "Class composition of the slides scanned in each year (100% = all slides of that year).\n"
               "If the composition is similar across years, batch and class are not confounded. Differences, together with the Cramér's V above, indicate "
               "that some classes were digitised mainly in particular periods."),
    ]
    colors = cluster_colors(len(BRACS_CLASSES))
    with plt.rc_context(STYLE):
        fig = plt.figure(figsize=(15, 16.5), layout="constrained")
        fig.suptitle(f"Technical properties of the {len(table)} slides", fontsize=14, weight="bold", x=0.01, ha="left")
        rows = fig.subfigures(3, 1)
        rng = np.random.default_rng(0)

        # Row 1: size by class, downsample factors
        top = split_with_notes(rows[0], notes[:2], [1, 1], 15, plot_height=3.4, note_height=2.2)
        ax_a, ax_b = top.subplots(1, 2)
        data = [table.loc[table["label"] == c, "gigapixels"].to_numpy() for c in BRACS_CLASSES]
        bp = ax_a.boxplot(data, positions=range(len(BRACS_CLASSES)), widths=0.55, patch_artist=True, showfliers=False,
                          medianprops={"color": INK_2, "lw": 1.5}, whiskerprops={"color": INK_2}, capprops={"color": INK_2})
        for patch, color in zip(bp["boxes"], colors):
            patch.set_facecolor(color); patch.set_alpha(0.35); patch.set_edgecolor(INK_2)
        for i, values in enumerate(data):
            ax_a.scatter(i + rng.uniform(-0.18, 0.18, len(values)), values, s=6, color=colors[i], alpha=0.6, linewidths=0)
        ax_a.set_xticks(range(len(BRACS_CLASSES))); ax_a.set_xticklabels(BRACS_CLASSES); ax_a.set_ylabel("gigapixels at level 1")
        ax_a.set_title("How large are the slides?")

        factors = {lvl: table[f"downsample_{lvl}"].dropna().round().astype(int).value_counts().sort_index() for lvl in (2, 3, 4)}
        labels_b, counts_b, colors_b = [], [], []
        for k, lvl in enumerate((2, 3, 4)):
            for factor, n in factors[lvl].items():
                labels_b.append(f"L{lvl}\n{factor}×"); counts_b.append(n); colors_b.append(SERIES[k])
        ax_b.bar(range(len(counts_b)), counts_b, color=colors_b, edgecolor=SURFACE, linewidth=1.5, width=0.7)
        for i, v in enumerate(counts_b):
            ax_b.text(i, v, f"{v}", ha="center", va="bottom", fontsize=8, color=INK_2)
        ax_b.set_xticks(range(len(counts_b))); ax_b.set_xticklabels(labels_b); ax_b.set_ylabel("slides"); ax_b.set_title("How are the pyramid levels downsampled?")

        # Row 2: MPP and scan year (batch screen)
        middle = split_with_notes(rows[1], notes[2:4], [1, 1], 15, plot_height=3.4, note_height=2.2)
        ax_m, ax_y = middle.subplots(1, 2)
        mpp_values = sorted(table["mpp"].dropna().unique())
        bottom_m = np.zeros(len(mpp_values))
        for s in SETS:
            counts_m = np.array([((table["mpp"] == m) & (table["set"] == s)).sum() for m in mpp_values])
            ax_m.bar(range(len(mpp_values)), counts_m, bottom=bottom_m, color=SET_COLORS[s], edgecolor=SURFACE, linewidth=1.5, width=0.6, label=s)
            bottom_m += counts_m
        for i, v in enumerate(bottom_m):
            ax_m.text(i, v, f"{int(v)}", ha="center", va="bottom", fontsize=8, color=INK_2)
        ax_m.set_xticks(range(len(mpp_values))); ax_m.set_xticklabels([f"{m:g}" for m in mpp_values]); ax_m.set_xlabel("MPP (µm per pixel, level 1)")
        ax_m.set_ylabel("slides"); ax_m.set_title("Is the resolution the same in every slide?"); ax_m.legend(frameon=False)

        years = list(set_by_year.columns)
        bottom_y = np.zeros(len(years))
        for s in SETS:
            ax_y.bar(range(len(years)), set_by_year.loc[s], bottom=bottom_y, color=SET_COLORS[s], edgecolor=SURFACE, linewidth=1.5, width=0.6, label=s)
            bottom_y += set_by_year.loc[s].to_numpy()
        for i, v in enumerate(bottom_y):
            ax_y.text(i, v, f"{int(v)}", ha="center", va="bottom", fontsize=8, color=INK_2)
        ax_y.set_xticks(range(len(years))); ax_y.set_xticklabels([str(int(y)) for y in years]); ax_y.set_xlabel("year of scanning")
        ax_y.set_ylabel("slides"); ax_y.set_title("When were the slides scanned?"); ax_y.legend(frameon=False)

        # Row 3: tiles per level, class mix per year
        last = split_with_notes(rows[2], notes[4:], [1, 1], 15, plot_height=3.4, note_height=2.2)
        ax_t, ax_c = last.subplots(1, 2)
        tiles = [((table[f"width_{lvl}"] // tile_size) * (table[f"height_{lvl}"] // tile_size)).dropna().to_numpy() for lvl in levels]
        bp = ax_t.boxplot(tiles, positions=range(4), widths=0.55, patch_artist=True, showfliers=False,
                          medianprops={"color": INK_2, "lw": 1.5}, whiskerprops={"color": INK_2}, capprops={"color": INK_2})
        for patch in bp["boxes"]:
            patch.set_facecolor(SERIES[0]); patch.set_alpha(0.35); patch.set_edgecolor(INK_2)
        for i, values in enumerate(tiles):
            ax_t.scatter(i + rng.uniform(-0.18, 0.18, len(values)), values, s=5, color=SERIES[0], alpha=0.5, linewidths=0)
            ax_t.text(i + 0.33, np.median(values), f"median {np.median(values):,.0f}", ha="left", va="center", fontsize=7, color=INK_2)
        ax_t.set_yscale("log"); ax_t.set_xlim(-0.5, 3.9); ax_t.set_xticks(range(4)); ax_t.set_xticklabels([f"level {l}" for l in levels]); ax_t.set_ylabel(f"{tile_size} px tiles per slide (upper bound)")
        ax_t.set_title("How many tiles does each level give?")

        share_year = class_by_year / class_by_year.sum() * 100
        left = np.zeros(len(years))
        for c, color in zip(BRACS_CLASSES, colors):
            ax_c.barh(range(len(years)), share_year.loc[c].to_numpy(), left=left, color=color, edgecolor=SURFACE, linewidth=1.5, label=c)
            left += share_year.loc[c].to_numpy()
        ax_c.set_yticks(range(len(years))); ax_c.set_yticklabels([f"{int(y)} (n = {int(class_by_year[y].sum())})" for y in years]); ax_c.invert_yaxis()
        ax_c.set_xlim(0, 100); ax_c.set_xlabel("% of the slides of the year"); ax_c.grid(False)
        ax_c.legend(frameon=False, ncol=7, loc="upper center", bbox_to_anchor=(0.5, -0.18), fontsize=8)
        ax_c.set_title("Does the class mix change with scan year?")

    if show:
        plt.show()

    return {"figure": fig, "class_by_year": class_by_year, "set_by_year": set_by_year, "cramers_v_class_year": v_year}


def build_summary(table: pd.DataFrame, split_stats: dict, class_weights: dict | None = None) -> pd.DataFrame:
    """
    Collect the main facts about the data set in one table (item, value), for the thesis text.

    * Arguments:
    table: The output of build_slide_table (with the scan columns for the technical items)
    split_stats: The output of plot_split_audit or split_audit_stats
    class_weights: Optional, the "weights" of plot_loss_weighting (adds the weight range)
    """

    # STEP 1: Labels and split
    counts = table["label"].value_counts().reindex(BRACS_CLASSES)
    train = table[table["set"] == "Training"]["label"].value_counts().reindex(BRACS_CLASSES)
    items = {
        "slides / patients": f"{len(table)} / {table['patient'].nunique()}",
        "slides per set (Training / Validation / Testing)": " / ".join(str(int((table["set"] == s).sum())) for s in SETS),
        "slides per class": ", ".join(f"{c} {int(n)}" for c, n in counts.items()),
        "imbalance ratio in Training (largest / smallest class)": f"{train.max() / train.min():.1f}",
        "class x set association (Cramer's V, permutation p)": f"{split_stats['cramers_v']:.2f}, p = {split_stats['p_value']:.2f}",
        "patients with slides in several sets": ", ".join(split_stats["patients_in_several_sets"]) or "none",
        "patients with slides of several classes": f"{split_stats['patients_with_mixed_classes']} of {split_stats['patients']}",
    }

    # STEP 2: Technical properties (when the scan columns are present)
    if "gigapixels" in table:
        items.update({
            "scanner / magnification": f"{', '.join(table['scanner'].dropna().unique())} / {', '.join(f'{m:g}x' for m in table['app_mag'].dropna().unique())}",
            "MPP values (um per pixel)": ", ".join(f"{m:g} ({int((table['mpp'] == m).sum())})" for m in sorted(table["mpp"].dropna().unique())),
            "scan years": ", ".join(f"{int(y)} ({n})" for y, n in table["scan_date"].dt.year.value_counts().sort_index().items()),
            "full-resolution size (median, max Gpx)": f"{table['gigapixels'].median():.1f}, {table['gigapixels'].max():.1f}",
            "slides without pyramid level 4": str(int((table["n_levels"] < 4).sum())),
            "total size on disk (GB)": f"{table['file_gb'].sum():.0f}",
        })
    if class_weights:
        items["inverse-frequency class weights (min - max)"] = f"{class_weights['inverse'].min():.2f} - {class_weights['inverse'].max():.2f}"

    return pd.DataFrame({"item": list(items), "value": list(items.values())})
