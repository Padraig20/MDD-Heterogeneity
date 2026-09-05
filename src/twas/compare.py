from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd

from src.twas.weights import ModelSpec

"""
compare.py

Head-to-head against ctPred, the single-cell-type predictor from scPrediXcan.

Rather than consuming the prediction model DBs scPrediXcan ships, the ctPred
arm is distilled here: `src/training/models/ctpred.py` supplies the teacher and
`src/distillation/train.py` writes the elastic net exactly as it does for our
own models. The comparison therefore takes a directory of weights JSONs and
nothing else, and `discover_ctpred_models` / `match_model` are the whole of the
plumbing.

That is not just a simplification. Both arms are then distilled from the same
individuals, over the same cis-windows, into the same identifier space, so the
only thing that differs between them is the teacher -- which is the comparison
the numbers are supposed to be making. Reading a foreign DB instead means the
two arms also differ in their reference panel, their variant identifiers and
their genome build, and any one of those can dominate the result.

The gene universe is the intersection
-------------------------------------
VariantFormer simply never sees some genes, so our models are a subset of
ctPred's. Leaving those extras in would let ctPred report more tests, a
different Bonferroni threshold and a longer hit list for a reason that has
nothing to do with the teacher. `restrict_to_shared_genes` therefore drops
every gene that is not in both models *before* either arm is run, so the two
TWAS are the same list of hypotheses. The covariances stay as built -- they
are a property of the model directory -- and unused genes in them are simply
never looked up.

One reference panel, one covariance each
----------------------------------------
Each arm carries its own covariance, built over the SNPs its own elastic net
selected by `src/twas/get_covariance_matrices.py`. What has to match is not the
covariance but the *cohort* underneath it, since an LD estimate from a
different panel is not comparable; `warn_on_reference_mismatch` checks that the
two directories were prepared from the same genotypes and the same individual
selection, which they are when both were built by one run of that script.

The percentile target needs no rescaling
----------------------------------------
ctPred is distilled against rank/percentile-normalized expression
(`--norm-targets percentiles`) while our models are usually distilled against
log expression, so its coefficients come out on a completely different scale.
That scale does not reach the z-score. As `model_db.py` sets out, S-PrediXcan
forms

    z = sum_l w_l z_l sigma_l / sqrt(w' COV w)

which is invariant to multiplying a gene's whole weight vector by a constant:
the numerator and the square root pick up the same factor. A change in the
units of the *target* is exactly such a per-gene constant, so it cancels, and
the z-scores and p-values of the two arms are directly comparable.

What does not cancel is a per-SNP rescale, which is the standardized-design
correction `model_db.py` applies. That is driven by `ModelSpec.standardized`
and is read off the JSON layout, so it is handled per arm without either side
needing to know what the other did.

The one quantity that is *not* comparable across the arms is `effect_size`: it
is an effect per unit of expression and so carries the target's units. Every
comparison here is on z-scores, p-values and significance calls for that
reason.
"""

# Suffixes distinguishing the two arms once their result tables are joined.
SUFFIX_OURS = "_ours"
SUFFIX_CTPRED = "_ctpred"
SUFFIXES = (SUFFIX_OURS, SUFFIX_CTPRED)


def normalize_cell_type(name: str) -> str:
    """
    Fold a cell-type name to a comparable key.

    The two directories hold the same labels written by two runs, so
    'CD14-low_CD16-positive_monocyte' and 'CD14 low CD16 positive monocyte'
    have to land on the same key.
    """
    return re.sub(r"[^a-z0-9]+", "", str(name).lower())


def discover_ctpred_models(directory: Path) -> dict[str, Path]:
    """
    Index a directory of ctPred weights JSONs by folded cell-type name.

    Same layout as `--models-dir`: one `<cell type>.json` per cell type, as
    written by `src/distillation/train.py`, with the covariance files beside
    it. `discover_models` does the globbing so that the covariance metadata
    sidecar -- also a `.json` in this directory -- is filtered out here too.
    """
    from src.twas.weights import discover_models

    directory = Path(directory)
    if not directory.is_dir():
        raise NotADirectoryError(
            f"--ctpred-models-dir is not a directory: {directory}"
        )

    models = {
        normalize_cell_type(path.stem): path
        for path in discover_models(directory, requested=None)
    }
    logging.info("Found %d ctPred model(s) in %s.", len(models), directory)
    return models


def match_model(models: dict[str, Path], cell_type: str) -> Optional[Path]:
    """The ctPred JSON for one of our cell types, matched on the folded name."""
    return models.get(normalize_cell_type(cell_type))


# Fields of a covariance's `reference` block that must agree between the arms
# for their LD estimates to be comparable. `individuals_hash` is the one that
# matters and is compared on the set of individuals rather than on how it was
# chosen, since the two arms legitimately read different target files.
REFERENCE_FIELDS = (
    "genotype_dir",
    "genotype_template",
    "n_individuals",
    "individuals_hash",
)


def restrict_to_shared_genes(
    ours: ModelSpec, theirs: ModelSpec
) -> tuple[ModelSpec, ModelSpec, dict]:
    """
    Restrict both model specs to the genes they both define.

    Matching is on the versionless Ensembl id, the same key `gene_overlap_report`
    uses. Genes only ctPred has -- the ones VariantFormer never produced a
    model for -- are dropped from that arm; genes only we have are dropped
    from ours, so both TWAS test the same hypotheses and share one
    multiple-testing burden.

    The covariance files are not touched: they were built for the full models
    and a later hash check would fail if we rewrote them. Extra genes in a
    covariance are ignored by S-PrediXcan.
    """
    our_keys = gene_keys(ours.snp_sets)
    their_keys = gene_keys(theirs.snp_sets)
    shared = our_keys & their_keys
    dropped_ours = our_keys - shared
    dropped_theirs = their_keys - shared
    if not shared:
        raise ValueError(
            f"The two models for '{ours.cell_type}' share no gene, so there is "
            "nothing to compare. Check that both were distilled against the "
            "same --select-genes and the same targets."
        )

    ours_kept = ours.restrict_to_gene_keys(shared)
    theirs_kept = theirs.restrict_to_gene_keys(shared)
    stats = {
        "n_genes_ours_before_intersection": len(our_keys),
        "n_genes_ctpred_before_intersection": len(their_keys),
        "n_genes_shared": len(shared),
        "n_genes_dropped_from_ours": len(dropped_ours),
        "n_genes_dropped_from_ctpred": len(dropped_theirs),
    }
    logging.info(
        "Comparing on the %d gene(s) both models define (ours had %d, ctPred "
        "had %d; dropped %d of ours and %d of ctPred).",
        len(shared), len(our_keys), len(their_keys),
        len(dropped_ours), len(dropped_theirs),
    )
    if dropped_theirs:
        logging.info(
            "ctPred-only genes are the ones VariantFormer never modelled; "
            "they are not tested on either arm."
        )
    return ours_kept, theirs_kept, stats


def warn_on_reference_mismatch(ours: dict, theirs: dict, cell_type: str) -> None:
    """
    Check that both arms' covariances were estimated on the same cohort.

    The two arms use separate covariances -- each over the SNPs its own elastic
    net selected -- which is fine, but only as long as the LD underneath them
    comes from the same individuals. If one directory was prepared with a
    different `--num-individuals`, or against different genotypes entirely,
    then a difference in z-scores between the arms is partly a difference in
    reference panel rather than in the models, and the comparison is not clean.
    """
    ours_reference = ours.get("reference") or {}
    theirs_reference = theirs.get("reference") or {}
    differing = [
        (field, ours_reference.get(field), theirs_reference.get(field))
        for field in REFERENCE_FIELDS
        if ours_reference.get(field) != theirs_reference.get(field)
    ]
    if not differing:
        return
    logging.warning(
        "The two arms' covariances for '%s' were built against different "
        "reference panels (%s). Rebuild both model directories with one run of "
        "src/twas/get_covariance_matrices.py, or the comparison also measures "
        "the difference between the panels.",
        cell_type,
        "; ".join(f"{field}: ours={ours!r}, ctPred={theirs!r}"
                  for field, ours, theirs in differing),
    )


def two_sample_quantiles(
    left: np.ndarray, right: np.ndarray, n_points: Optional[int] = None
) -> tuple[np.ndarray, np.ndarray]:
    """
    Matched quantiles of two samples, for a quantile-quantile comparison.

    This deliberately does not pair genes. A Q-Q plot asks whether one method's
    *distribution* of evidence is shifted relative to the other's, which is the
    question "does it have more power" -- a gene-level pairing answers the
    different question of whether the two agree case by case, and
    `matched_pvalues` covers that.
    """
    left = np.asarray(left, dtype=float)
    right = np.asarray(right, dtype=float)
    left = left[np.isfinite(left)]
    right = right[np.isfinite(right)]
    if left.size == 0 or right.size == 0:
        return np.array([]), np.array([])

    n = n_points or min(left.size, right.size)
    probabilities = (np.arange(1, n + 1) - 0.5) / n
    return np.quantile(left, probabilities), np.quantile(right, probabilities)


def matched_pvalues(
    ours: pd.DataFrame, theirs: pd.DataFrame, suffixes: tuple[str, str] = SUFFIXES
) -> pd.DataFrame:
    """
    Inner join of the two result tables on the versionless Ensembl id.

    Both sides carry their own significance flags, so the join is enough to read
    off which genes each method finds and which they share.
    """
    def prepare(frame: pd.DataFrame) -> pd.DataFrame:
        columns = [
            c for c in
            ("gene", "gene_name", "zscore", "pvalue", "qvalue", "effect_size",
             "n_snps_used", "significant_fdr", "significant_bonferroni")
            if c in frame.columns
        ]
        subset = frame[columns].copy()
        subset["gene_key"] = subset["gene"].astype(str).str.split(".").str[0]
        return subset.drop_duplicates("gene_key")

    merged = prepare(ours).merge(
        prepare(theirs), on="gene_key", how="inner", suffixes=suffixes
    )
    return merged


def gene_keys(values: Iterable) -> set[str]:
    """Versionless, case-folded Ensembl ids, for comparing gene sets."""
    return {
        str(value).split(".")[0].strip().upper()
        for value in values
        if value is not None and str(value) != "nan"
    }


def gene_overlap_report(
    our_model_genes: Iterable,
    their_model_genes: Iterable,
    our_results: pd.DataFrame,
    their_results: pd.DataFrame,
) -> dict:
    """
    Where genes are lost between the two model definitions and the two results.

    A small shared-gene count has two very different explanations and this
    separates them. If the two *models* already barely overlap, the two
    distillation runs were given different gene lists. If the models overlap
    well but the *results* do not, genes are being dropped downstream instead --
    by `--max-snps-in-gene`, or by every variant of a gene being absent from the
    GWAS -- and `*_lost_from_model` says which side is doing the dropping.
    """
    our_model, their_model = gene_keys(our_model_genes), gene_keys(their_model_genes)
    our_tested = gene_keys(our_results["gene"]) if len(our_results) else set()
    their_tested = gene_keys(their_results["gene"]) if len(their_results) else set()

    report = {
        "n_genes_in_our_model": len(our_model),
        "n_genes_in_ctpred_model": len(their_model),
        "n_genes_shared_by_models": len(our_model & their_model),
        "n_genes_in_our_results": len(our_tested),
        "n_genes_in_ctpred_results": len(their_tested),
        "n_genes_shared_by_results": len(our_tested & their_tested),
        "n_genes_ours_lost_from_model": len(our_model - our_tested),
        "n_genes_ctpred_lost_from_model": len(their_model - their_tested),
    }
    shared_models = report["n_genes_shared_by_models"]
    if shared_models:
        report["frac_shared_models_reaching_both_results"] = (
            report["n_genes_shared_by_results"] / shared_models
        )
    return report


def log_overlap_report(report: dict) -> None:
    """Explain the attrition at INFO, since a low overlap is easy to misread."""
    logging.info(
        "Gene overlap: %d in our model, %d in the ctPred model, %d shared. "
        "After the association: %d ours, %d theirs, %d shared.",
        report["n_genes_in_our_model"], report["n_genes_in_ctpred_model"],
        report["n_genes_shared_by_models"], report["n_genes_in_our_results"],
        report["n_genes_in_ctpred_results"], report["n_genes_shared_by_results"],
    )
    if report["n_genes_shared_by_models"] == 0:
        logging.warning(
            "The two models share no gene at all, which for two runs of the same "
            "distillation should not happen -- check that both were given the "
            "same --select-genes and the same targets directory."
        )
        return

    fraction = report.get("frac_shared_models_reaching_both_results", 1.0)
    if fraction >= 0.5:
        return
    logging.warning(
        "Only %.0f%% of the %d gene(s) both models define survive into both "
        "result tables (%d ours, %d theirs dropped between model and result). "
        "Both arms share one covariance and one GWAS, so this is a per-gene "
        "weight difference: genes whose elastic net kept no variant the GWAS "
        "also carries, or genes skipped by --max-snps-in-gene.",
        100 * fraction, report["n_genes_shared_by_models"],
        report["n_genes_ours_lost_from_model"],
        report["n_genes_ctpred_lost_from_model"],
    )


def comparison_metrics(
    ours: pd.DataFrame,
    theirs: pd.DataFrame,
    matched: pd.DataFrame,
    suffixes: tuple[str, str] = SUFFIXES,
) -> dict:
    """
    How the two arms' hit lists relate on the genes they both test.

    Each arm's own significance counts and LD-block coverage are produced by the
    per-arm analysis; what is left here is strictly the overlap. Both arms are
    restricted to the same genes before TWAS, so they share one multiple-testing
    burden and the two Bonferroni thresholds agree.

    `effect_size` is deliberately absent: it carries the units of the
    distillation target, and ctPred's percentile target is not our log target.
    """
    ours_suffix, theirs_suffix = suffixes
    statistics: dict = {"n_genes_shared": int(len(matched))}

    for criterion in ("fdr", "bonferroni"):
        ours_column = f"significant_{criterion}{ours_suffix}"
        theirs_column = f"significant_{criterion}{theirs_suffix}"
        if ours_column not in matched.columns or theirs_column not in matched.columns:
            continue
        mine = matched[ours_column].fillna(False)
        yours = matched[theirs_column].fillna(False)
        statistics[f"{criterion}_both"] = int((mine & yours).sum())
        statistics[f"{criterion}_ours_only"] = int((mine & ~yours).sum())
        statistics[f"{criterion}_ctpred_only"] = int((~mine & yours).sum())
        union = int((mine | yours).sum())
        statistics[f"{criterion}_jaccard"] = (
            statistics[f"{criterion}_both"] / union if union else float("nan")
        )

    for suffix, side in ((ours_suffix, "ours"), (theirs_suffix, "ctpred")):
        column = f"zscore{suffix}"
        if column in matched.columns:
            statistics[f"mean_abs_zscore_{side}"] = float(
                matched[column].abs().mean()
            )
    left, right = f"zscore{ours_suffix}", f"zscore{theirs_suffix}"
    if left in matched.columns and right in matched.columns:
        pair = matched[[left, right]].dropna()
        if len(pair) > 2:
            statistics["zscore_correlation"] = float(pair[left].corr(pair[right]))
    return statistics


__all__ = [
    "REFERENCE_FIELDS",
    "SUFFIXES",
    "SUFFIX_CTPRED",
    "SUFFIX_OURS",
    "comparison_metrics",
    "discover_ctpred_models",
    "gene_keys",
    "gene_overlap_report",
    "log_overlap_report",
    "match_model",
    "matched_pvalues",
    "normalize_cell_type",
    "restrict_to_shared_genes",
    "two_sample_quantiles",
    "warn_on_reference_mismatch",
]
