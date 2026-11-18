# IMPORTS *****************************************************************************************
# External libraries
from pathlib import Path
import numpy as np
import pandas as pd

# Internal imports ********************************************************************************
from utils.dataset_overview import build_slide_table
from utils.per_fold import fold_labels_file, fold_root, fold_synthetic_root, fold_table_from_metadata
from utils.pca_analysis import timed
from utils.synthetic_gaussian import (build_generator, collect_class_statistics, composition_table, fit_class_clusters, fit_shared_clusters,
                                      generate_configuration, load_training_slides, prune_clusters, sample_class_tiles, save_configuration_metadata,
                                      write_synthetic_labels)
from utils.synthetic_lesion import (blend_summaries, bag_level_auc, bag_level_reference, bag_summaries, fit_lesion_models, lesion_enrichment, real_bag_summaries)


# Helpers *****************************************************************************************
def method_name(params: dict) -> str:
    """The name of the output folder and of the label-sheet rows: the generation method, the output tag and "_lesion" in the lesion-share mode."""
    return params["method"] + params.get("output_tag", "") + ("_lesion" if params.get("composition_mode", "blend") == "lesion_share" else "")


def check_params(params: dict) -> None:
    """Stop on settings that cannot work together."""
    if params["method"] != "hier_lowrank" and params["covariance_type"] not in ("diag", "full"):
        raise ValueError("the 'gaussian' method has no low-rank covariance")
    if params.get("composition_mode", "blend") not in ("blend", "lesion_share"):
        raise ValueError("composition_mode must be 'blend' or 'lesion_share'")
    if params.get("composition_mode", "blend") == "lesion_share" and not params["shared_codebook"]:
        raise ValueError("composition_mode='lesion_share' needs shared_codebook=True (the lesion clusters must be the same clusters in every class)")


def _options(params: dict) -> dict:
    """The generator options of the configuration."""
    if params["method"] == "hier_lowrank":
        return dict(covariance=params["covariance_type"], rank=params["lowrank_rank"], offset_scale=params["offset_scale"], min_slides=params["min_slides"],
                    min_slide_tiles=params["min_slide_tiles"], shrinkage=params["covariance_shrinkage"], reg=params["covariance_reg"], min_tiles=params["min_cluster_tiles"])
    return dict(covariance=params["covariance_type"], reg=params["covariance_reg"], min_tiles=params["min_cluster_tiles"])


# Main functions: sources *************************************************************************
def build_sources(params: dict, per_fold: bool, n_folds: int = 5) -> list[dict]:
    """
    The data sources that the generator is fitted on: the global one (official Training slides) or one per cross-validation fold.

    A source holds the name, the folder with the PCA files (pca_<D> below it), the slide table whose Training slides are fitted, the output
    root, the label sheet and the seed offset (0 for the global source, so the default run keeps its seeds; the fold index in the fold sources).
    A fold source takes the fit slides of its fold from the metadata.json of the fold's PCA folder (01_2 with PER_FOLD), so its generator and its PCA
    see the same slides.

    * Arguments:
    params: The pipeline parameters (see the notebook); uses pca_input_dir, synthetic_output_dir, labels_file, labels_synthetic_file, pca_dimensions
    per_fold: One source per fold instead of the global source
    n_folds: The number of folds
    """
    table = build_slide_table(params["labels_file"])
    if not per_fold:
        return [{"name": "global", "fold": None, "pca_root": Path(params["pca_input_dir"]), "table": table, "out_root": Path(params["synthetic_output_dir"]),
                 "labels_file": Path(params["labels_synthetic_file"]), "seed_offset": 0}]
    root, dim = Path(params["pca_input_dir"]), params["pca_dimensions"][0]
    return [{"name": f"fold {k}", "fold": k, "pca_root": fold_root(root, k), "table": fold_table_from_metadata(root, k, dim, table),
             "out_root": fold_synthetic_root(root, k), "labels_file": fold_labels_file(params["labels_synthetic_file"], k), "seed_offset": k} for k in range(n_folds)]


# Main functions: stages **************************************************************************
def select_training(source: dict, params: dict) -> dict:
    """The Training slides of the source, validated in every PCA dimension (memory-mapped, no tile is loaded). Returns {training, classes, train}."""
    training = {}
    for dim in params["pca_dimensions"]:
        training[dim] = load_training_slides(source["table"], source["pca_root"] / f"pca_{dim}", params["model_name"], dim)
    first = params["pca_dimensions"][0]
    classes, train = training[first]["classes"], training[first]["train"]
    assert all(training[d]["train"][["slide", "n_tiles"]].equals(train[["slide", "n_tiles"]]) for d in params["pca_dimensions"])   # same slides and tile counts in every dimension

    return {"source": source, "training": training, "classes": classes, "train": train}


def fit_clusters(state: dict, params: dict) -> dict:
    """Sample once per (dimension, class) and fit one MiniBatchKMeans per k (per class, or one shared codebook per k); adds clusters, real_diag and real_groups."""
    source, classes, train, seed0 = state["source"], state["classes"], state["train"], params["seed"] + source_offset(state)
    clusters, real_diag, real_groups = {}, {}, {}
    for dim in params["pca_dimensions"]:
        pca_dir = source["pca_root"] / f"pca_{dim}"
        class_samples = {}
        for ci, label in enumerate(classes):
            with timed(f"{source['name']} D={dim} {label}: sample" + ("" if params["shared_codebook"] else f" + fit k={params['k_values']}")):
                X, slide_index = sample_class_tiles(pca_dir, params["model_name"], train, label, params["max_tiles_per_wsi_fit"], params["max_tiles_per_class"],
                                                    seed0 + dim + ci, with_slides=True)

                # A small real reference (and the slide of each tile) for the diagnostics
                rng = np.random.default_rng([seed0, dim, ci])
                pick = rng.choice(len(X), min(len(X), 5000), replace=False)
                real_diag[(dim, label)], real_groups[(dim, label)] = X[pick], slide_index[pick]

                # Class-specific clustering, one per k
                if params["shared_codebook"]:
                    class_samples[label] = X
                    continue
                for k in params["k_values"]:
                    seed = int(np.random.default_rng([seed0, dim, k, ci]).integers(2**31 - 1))
                    clusters.setdefault((dim, k), {})[label] = fit_class_clusters(X, k, params["kmeans_batch_size"], seed, params["kmeans_n_init"])
            del X

        # Shared codebook: one clustering per k for all classes
        if params["shared_codebook"]:
            with timed(f"{source['name']} D={dim}: shared codebook k={params['k_values']}"):
                for k in params["k_values"]:
                    seed = int(np.random.default_rng([seed0, dim, k]).integers(2**31 - 1))
                    shared = fit_shared_clusters(class_samples, k, params["kmeans_batch_size"], seed, params["kmeans_n_init"])
                    clusters[(dim, k)] = {label: shared for label in classes}
            del class_samples

    return {**state, "clusters": clusters, "real_diag": real_diag, "real_groups": real_groups}


def source_offset(state: dict) -> int:
    """The seed offset of the source of a state (0 for the global source)."""
    return int(state["source"]["seed_offset"])


def collect_statistics(state: dict, params: dict) -> dict:
    """One streaming pass per (dimension, k, class): composition and sums; adds stats and compositions."""
    source, classes, train = state["source"], state["classes"], state["train"]
    full = params["covariance_type"] in ("full", "lowrank")
    stats, compositions = {}, {}
    for dim in params["pca_dimensions"]:
        pca_dir = source["pca_root"] / f"pca_{dim}"
        for k in params["k_values"]:
            with timed(f"{source['name']} D={dim} k={k}: composition and statistics"):
                for label in classes:
                    slides = train.loc[train["label"] == label, "slide"].tolist()
                    st = collect_class_statistics(state["clusters"][(dim, k)][label], pca_dir, params["model_name"], slides, full)
                    stats.setdefault((dim, k), {})[label] = st
                    compositions.setdefault((dim, k), {})[label] = composition_table(st, slides)
    for (dim, k), per in stats.items():                      # every Training tile was assigned
        for label, s in per.items():
            assert s["n"].sum() == state["training"][dim]["train"].query("label == @label")["n_tiles"].sum()

    return {**state, "stats": stats, "compositions": compositions, "pruned": {}}


def prune_state(state: dict, params: dict) -> dict:
    """
    MIN_CLUSTER_SLIDES: remove the clusters that fewer slides support and collect the statistics again (prune_clusters); a state is returned unchanged when
    min_cluster_slides is 0. The removed clusters are kept in state["pruned"] for metadata.json.
    """
    if params.get("min_cluster_slides", 0) <= 0:
        return state
    source, classes, train = state["source"], state["classes"], state["train"]
    full = params["covariance_type"] in ("full", "lowrank")
    clusters, stats, compositions, pruned = dict(state["clusters"]), dict(state["stats"]), dict(state["compositions"]), {}
    for (dim, k) in list(state["stats"]):
        pca_dir = source["pca_root"] / f"pca_{dim}"

        def recollect(label, model, pca_dir=pca_dir):
            return collect_class_statistics(model, pca_dir, params["model_name"], train.loc[train["label"] == label, "slide"].tolist(), full)

        with timed(f"{source['name']} D={dim} k={k}: prune clusters seen in < {params['min_cluster_slides']} slides"):
            new_clusters, new_stats, info = prune_clusters(clusters[(dim, k)], stats[(dim, k)], classes, params["min_cluster_slides"], params["min_slide_tiles"], recollect)
        clusters[(dim, k)], stats[(dim, k)], pruned[(dim, k)] = new_clusters, new_stats, info
        compositions[(dim, k)] = {label: composition_table(new_stats[label], train.loc[train["label"] == label, "slide"].tolist()) for label in classes}
        for label, s in new_stats.items():
            assert s["n"].sum() == state["training"][dim]["train"].query("label == @label")["n_tiles"].sum()

    return {**state, "clusters": clusters, "stats": stats, "compositions": compositions, "pruned": pruned}


def fit_generators(state: dict, params: dict) -> dict:
    """Fit one generator per (dimension, k, class) and release the large accumulators; adds generators and options."""
    options = _options(params)
    generators = {}
    with timed(f"{state['source']['name']}: Gaussian parameters"):
        for key, per in state["stats"].items():
            for label, st in per.items():
                g = build_generator(params["method"], **options)
                generators.setdefault(key, {})[label] = g.fit(st, state["clusters"][key][label].cluster_centers_)
                st["xtx"] = None

    return {**state, "generators": generators, "options": options}


def fit_lesion(state: dict, params: dict) -> dict:
    """
    The lesion clusters and the lesion-share model of every (dimension, k) (composition_mode 'lesion_share'); adds lesion = {key: {enrichment, models}}.
    A state of the 'blend' mode is returned unchanged.
    """
    if params.get("composition_mode", "blend") != "lesion_share":
        return state
    lesion = {}
    for key, per in state["stats"].items():
        counts = {c: per[c]["counts"] for c in state["classes"]}
        table = lesion_enrichment(counts, params["lesion_classes"], params["reference_classes"], params["min_lesion_slides"], params["min_slide_tiles"],
                                  params["lesion_log2_threshold"], params["lesion_alpha"])
        models = fit_lesion_models(counts, {c: state["compositions"][key][c] for c in state["classes"]}, table["lesion"].to_numpy(), params["lesion_classes"],
                                   params["reference_classes"], params["min_slide_tiles"], params["n_lesion_templates"], params["n_templates_per_wsi"],
                                   params["template_concentration"], params["max_synthetic_tiles"])
        lesion[key] = {"enrichment": table, "models": models}

    return {**state, "lesion": lesion}


def generate(state: dict, params: dict, out_dir: Path | None, overwrite: bool) -> dict:
    """Generate (and optionally save the bags of) every configuration; adds results (key -> generate_configuration output, with its settings)."""
    source, classes, method = state["source"], state["classes"], method_name(params)
    mode = params.get("composition_mode", "blend")
    results = {}
    for dim in params["pca_dimensions"]:
        for k in params["k_values"]:
            key = (dim, k)
            settings = {"method": method, "dim": dim, "k": k, "seed": params["seed"] + source_offset(state), "n_per_class": params["n_synthetic_wsis_per_class"],
                        "max_tiles": params["max_synthetic_tiles"], "n_templates": params["n_templates_per_wsi"], "template_concentration": params["template_concentration"],
                        "model_name": params["model_name"], "pca_dir": source["pca_root"] / f"pca_{dim}", "save_embeddings": params["save_synthetic_embeddings"],
                        "covariance": params["covariance_type"], "covariance_reg": params["covariance_reg"], "min_cluster_tiles": params["min_cluster_tiles"],
                        "shared_codebook": params["shared_codebook"], "generator_options": state["options"],
                        "kmeans": {"batch_size": params["kmeans_batch_size"], "n_init": params["kmeans_n_init"]},
                        "sampling": {"max_tiles_per_wsi_fit": params["max_tiles_per_wsi_fit"], "max_tiles_per_class": params["max_tiles_per_class"]}}

            # Additions that exist only when a new option is used, so the default run writes the files it always wrote
            extra, bag_makers = {}, None
            if method != params["method"]:
                extra["generator_method"] = params["method"]
            if source["fold"] is not None:
                extra["cv_fold"] = source["fold"]
            if mode == "lesion_share":
                models, table = state["lesion"][key]["models"], state["lesion"][key]["enrichment"]
                bag_makers = {label: (lambda gen, rng, m=m: m.make(gen, rng)) for label, m in models.items()}
                settings.update({"lesion_models": models, "diag_bags": max(100, params["n_synthetic_wsis_per_class"] // 5)})
                extra.update({"composition_mode": mode, "lesion_classes": list(models), "reference_classes": list(params["reference_classes"]),
                              "lesion_rule": {"min_lesion_slides": params["min_lesion_slides"], "min_slide_tiles": params["min_slide_tiles"], "log2_threshold": params["lesion_log2_threshold"],
                                              "alpha": params["lesion_alpha"], "test": "one-sided Mann-Whitney U on per-slide shares, Benjamini-Hochberg"},
                              "lesion_clusters": np.flatnonzero(table["lesion"]).tolist(), "n_lesion_templates": params["n_lesion_templates"]})
            if params.get("min_cluster_slides", 0) > 0:
                extra.update({"min_cluster_slides": params["min_cluster_slides"], "pruned_clusters": state["pruned"].get(key, {})})
            settings["extra_metadata"] = extra
            with timed(f"{source['name']} D={dim} k={k}: generation"):
                results[key] = generate_configuration(state["generators"][key], state["clusters"][key], state["compositions"][key],
                                                      {c: state["real_diag"][(dim, c)] for c in classes}, settings, out_dir, overwrite, bag_makers)
            results[key]["settings"] = settings

    return {**state, "results": results}


def save_state(state: dict, params: dict, overwrite: bool) -> pd.DataFrame | None:
    """Write metadata.json, cluster_models.pt, labels.csv of every configuration and the label sheet of the source; returns the sheet (None when nothing is saved)."""
    if not params["save_synthetic_embeddings"]:
        return None
    for key, r in state["results"].items():
        save_configuration_metadata(r, state["clusters"][key], state["generators"][key], state["compositions"][key], r["settings"], state["train"]["slide"].tolist())

    return write_synthetic_labels(list(state["results"].values()), state["source"]["labels_file"], overwrite)


def run_source(source: dict, params: dict, overwrite: bool) -> dict:
    """All stages for one source, then release the large arrays; returns {name, configurations, sheet rows, pruned}. The notebook uses it for the folds that it does not inspect."""
    check_params(params)
    state = select_training(source, params)
    state = fit_clusters(state, params)
    state = collect_statistics(state, params)
    state = prune_state(state, params)
    state = fit_generators(state, params)
    state = fit_lesion(state, params)
    state = generate(state, params, Path(source["out_root"]) if params["save_synthetic_embeddings"] else None, overwrite)
    sheet = save_state(state, params, overwrite)

    return {"name": source["name"], "configurations": list(state["results"]), "bags": int(sum(len(r["wsis"]) for r in state["results"].values())), "sheet": None if sheet is None else len(sheet),
            "pruned": state["pruned"]}


# Main functions: diagnostics *********************************************************************
def lesion_bag_auc(state: dict, key: tuple, params: dict, n_reference_bags: int = 100) -> pd.DataFrame:
    """
    Bag-level real-vs-synthetic AUC of the lesion classes for one configuration: lesion-share bags, template-blend bags and the real-vs-real reference.

    The summary of a bag is its mean embedding and its cluster proportions (synthetic proportions: the bag assigned again to the nearest centre). The
    real summaries come from the statistics of the Training slides; the template-blend bags are generated here with the same generator.

    * Arguments:
    state: A state with stats, compositions, generators, clusters, results and lesion
    key: (dim, k)
    params: The pipeline parameters
    n_reference_bags: The number of template-blend bags per class

    Returns columns label, method, auc, auc_std (the input of synthetic_gaussian.plot_fidelity_auc), n_real, n_synthetic.
    """
    rows, seed = [], params["seed"] + source_offset(state)
    result, models = state["results"][key], state["lesion"][key]["models"]
    for label, model in models.items():
        real = real_bag_summaries(state["stats"][key][label], state["compositions"][key][label])
        sub = result["wsis"][result["wsis"]["WSI label"] == label].iloc[: len(result["bag_means"][label]["mean"])]
        counts, mean = result["bag_means"][label]["counts"].astype(np.float64), result["bag_means"][label]["mean"].astype(np.float64)
        mean_embedding = (counts[:, :, None] * mean).sum(1) / counts.sum(1, keepdims=True)
        reassigned = np.stack(sub["reassigned"].to_numpy()).astype(np.float64)
        synthetic = bag_summaries(mean_embedding, reassigned / reassigned.sum(1, keepdims=True))
        rng = np.random.default_rng([seed, key[0], key[1], 31])
        blend = blend_summaries(state["generators"][key][label], state["compositions"][key][label], state["clusters"][key][label], rng, n_reference_bags,
                                params["n_templates_per_wsi"], params["template_concentration"], params["max_synthetic_tiles"])
        for name, other in (("lesion_share", synthetic), ("blend (reference)", blend)):
            rows.append({"label": label, "method": name, **bag_level_auc(real, other, seed)})
        rows.append({"label": label, "method": "real vs real (disjoint slides)", **bag_level_reference(real, seed)})

    return pd.DataFrame(rows)
