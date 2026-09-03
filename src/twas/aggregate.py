from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
import pandas as pd
from scipy.stats import norm

"""
aggregate.py

Post-processing of S-PrediXcan output: multiple-testing correction for a single
run, and the multiple-imputation summary across the member-bootstrap draws.

MI aggregation happens on the z-score. Each draw is a complete TWAS, so the
per-gene z-scores are directly comparable (unlike `effect_size`, whose units
depend on the gene's expression scale), and the expectation of the z-score is
the natural point summary.

A caveat worth keeping in mind when reading the numbers: the draws are
bootstrap refits of one cohort and are therefore strongly correlated, so
`zscore_var` measures how *stable* a gene's association is across fits, not the
sampling variance of an estimator. The reported p-value comes from `E[z]` alone
and does not absorb that spread, which makes it anti-conservative.
"""

# The columns S-PrediXcan emits (see `MetaxcanUtilities._results_column_order`).
GENE_COLUMNS = ["gene", "gene_name"]

# Per-draw columns carried into the tidy `long` frame.
DRAW_COLUMNS = ["zscore", "pvalue", "effect_size", "n_snps_used", "best_gwas_p"]

Z_CI_MULTIPLIER = 1.959963984540054  # two-sided 95%

# Default cut points for the model-agreement curve.
AGREEMENT_THRESHOLDS = (0.0, 0.2, 0.4, 0.5, 0.6, 0.8, 0.9, 1.0)


def benjamini_hochberg(pvalues: Sequence[float]) -> np.ndarray:
    """
    BH-adjusted p-values, with NaNs passed through.

    Implemented here rather than pulled from statsmodels to keep the TWAS
    package's dependency surface identical to the rest of the repository.
    """
    values = np.asarray(pvalues, dtype=float)
    qvalues = np.full(values.shape, np.nan)
    finite = np.isfinite(values)
    n = int(finite.sum())
    if n == 0:
        return qvalues

    order = np.flatnonzero(finite)[np.argsort(values[finite], kind="mergesort")]
    ranked = values[order] * n / np.arange(1, n + 1)
    # Enforce monotonicity from the largest p-value downwards.
    qvalues[order] = np.minimum.accumulate(ranked[::-1])[::-1].clip(max=1.0)
    return qvalues


def annotate_significance(
    frame: pd.DataFrame, fdr: float = 0.05, pvalue_column: str = "pvalue"
) -> pd.DataFrame:
    """Add BH q-values plus Bonferroni threshold/flag over the tested genes."""
    frame = frame.copy()
    pvalues = frame[pvalue_column].to_numpy(dtype=float)
    n_tested = int(np.isfinite(pvalues).sum())

    frame["qvalue"] = benjamini_hochberg(pvalues)
    frame["significant_fdr"] = frame["qvalue"] < fdr
    bonferroni = 0.05 / n_tested if n_tested else np.nan
    frame["bonferroni_threshold"] = bonferroni
    frame["significant_bonferroni"] = pvalues < bonferroni
    return frame.sort_values(pvalue_column, na_position="last").reset_index(drop=True)


def genomic_inflation(pvalues: Sequence[float]) -> float:
    """
    Genomic control lambda: the median chi-square (1 df) over its null median.

    A value near 1 means the p-values behave like the null away from the true
    signal; a large value points at residual confounding or an LD reference
    that does not match the GWAS cohort.
    """
    values = np.asarray(pvalues, dtype=float)
    values = values[np.isfinite(values) & (values > 0)]
    if values.size == 0:
        return float("nan")
    chi2 = norm.isf(values / 2.0) ** 2
    return float(np.median(chi2) / 0.4549364231195728)


def aggregate_draws(
    per_draw: dict[str, pd.DataFrame], fdr: float = 0.05
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Summarise the member-bootstrap draws into one table per gene.

    Returns `(aggregated, long)` where `long` is the tidy per-gene-per-draw
    frame the MI figures are drawn from.

    `zscore_var` is only reported for genes that come out significant under
    `E[z]`; for everything else the spread of a null association is not a
    number worth reading, so it is left as NA.
    """
    if not per_draw:
        raise ValueError("No draws to aggregate.")

    frames = []
    for draw_id, frame in per_draw.items():
        columns = [c for c in GENE_COLUMNS if c in frame.columns]
        subset = frame[columns + _present(frame, DRAW_COLUMNS)].copy()
        subset["draw"] = draw_id
        if "pvalue" in subset.columns:
            # Each draw is a complete, independently corrected TWAS, so its own
            # gene count sets its own multiple-testing burden.
            subset = annotate_significance(subset, fdr=fdr)
        frames.append(subset)
    long = pd.concat(frames, ignore_index=True)

    specification = {
        "zscore_mean": ("zscore", "mean"),
        "zscore_var": ("zscore", lambda values: values.var(ddof=1)),
        "zscore_min": ("zscore", "min"),
        "zscore_max": ("zscore", "max"),
        "n_draws": ("zscore", "count"),
    }
    if "effect_size" in long.columns:
        specification["effect_size_mean"] = ("effect_size", "mean")
    if "n_snps_used" in long.columns:
        specification["mean_n_snps_used"] = ("n_snps_used", "mean")
    if "best_gwas_p" in long.columns:
        # Identical across draws up to which SNPs each fit kept, so the minimum
        # is the strength of the GWAS signal available at the locus.
        specification["best_gwas_p"] = ("best_gwas_p", "min")
    # How many member-bootstrap fits independently call the gene significant.
    for criterion in ("bonferroni", "fdr"):
        column = f"significant_{criterion}"
        if column in long.columns:
            specification[f"n_draws_significant_{criterion}"] = (column, "sum")
    aggregated = long.groupby("gene", sort=False).agg(**specification).reset_index()

    if "gene_name" in long.columns:
        names = long.drop_duplicates("gene").set_index("gene")["gene_name"]
        aggregated["gene_name"] = aggregated["gene"].map(names)

    # A single draw has no between-fit spread to report.
    aggregated.loc[aggregated["n_draws"] < 2, "zscore_var"] = np.nan
    aggregated["zscore_sd"] = np.sqrt(aggregated["zscore_var"])

    # Model agreement: the share of fits that call the gene significant on their
    # own. This is a stability measure independent of E[z] -- a gene can carry a
    # large mean z-score that only a handful of fits actually support.
    draw_counts = aggregated["n_draws"].to_numpy(dtype=float)
    for criterion in ("bonferroni", "fdr"):
        column = f"n_draws_significant_{criterion}"
        if column in aggregated.columns:
            aggregated[column] = aggregated[column].astype(int)
            aggregated[f"agreement_{criterion}"] = np.divide(
                aggregated[column].to_numpy(dtype=float),
                draw_counts,
                out=np.full(len(aggregated), np.nan),
                where=draw_counts > 0,
            )

    zscores = aggregated["zscore_mean"].to_numpy(dtype=float)
    aggregated["zscore"] = zscores
    aggregated["pvalue"] = 2.0 * norm.sf(np.abs(zscores))
    if "effect_size_mean" in aggregated.columns:
        aggregated["effect_size"] = aggregated["effect_size_mean"]

    sds = aggregated["zscore_sd"].to_numpy(dtype=float)
    aggregated["zscore_ci_low"] = zscores - Z_CI_MULTIPLIER * sds
    aggregated["zscore_ci_high"] = zscores + Z_CI_MULTIPLIER * sds

    aggregated = annotate_significance(aggregated, fdr=fdr)

    # Report the between-fit spread only where the expectation is significant.
    insignificant = ~aggregated["significant_fdr"].fillna(False)
    for column in ("zscore_var", "zscore_sd", "zscore_ci_low", "zscore_ci_high"):
        aggregated.loc[insignificant, column] = np.nan

    ordered = [
        "gene",
        "gene_name",
        "zscore",
        "pvalue",
        "qvalue",
        "effect_size",
        "zscore_var",
        "zscore_sd",
        "zscore_ci_low",
        "zscore_ci_high",
        "zscore_min",
        "zscore_max",
        "n_draws",
        "n_draws_significant_bonferroni",
        "agreement_bonferroni",
        "n_draws_significant_fdr",
        "agreement_fdr",
        "mean_n_snps_used",
        "best_gwas_p",
        "significant_fdr",
        "significant_bonferroni",
        "bonferroni_threshold",
    ]
    aggregated = aggregated[[c for c in ordered if c in aggregated.columns]]
    return aggregated, long


def agreement_strata(
    frame: pd.DataFrame,
    agreement_column: str = "agreement_bonferroni",
    edges: Sequence[float] = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0),
) -> pd.DataFrame:
    """
    Compare genes by how many member-bootstrap fits agree they are significant.

    Genes are binned on the share of fits calling them significant, and each bin
    is described by the quantities that plausibly separate a 5-of-25 gene from a
    20-of-25 one:

    `abs_zscore_*`     how strong the pooled association is,
    `zscore_sd_*`      how much the association moves between fits,
    `stability_*`      |E[z]| / sd(z), i.e. signal relative to that movement,
    `best_gwas_p_*`    the strongest GWAS signal in the gene's cis window,
    `n_snps_used_*`    how many variants the fits actually put weight on.

    The distinction that matters is between a gene the fits disagree on because
    the underlying GWAS locus is weak (`best_gwas_p` is unremarkable) and one
    they disagree on because the elastic net keeps reshuffling which correlated
    variant carries the weight (`best_gwas_p` is strong but `stability` is low).
    Only the second is a fine-mapping problem.

    `n_ld_blocks` is filled in when the frame has been annotated with a
    `block_index` column by `ld_blocks.assign_frame`.
    """
    if agreement_column not in frame.columns:
        raise KeyError(
            f"{agreement_column} is not present; agreement is only defined for "
            "multiple-imputation runs (--model-kind mi)."
        )

    data = frame[frame[agreement_column].notna()].copy()
    if data.empty:
        return pd.DataFrame()

    edges = list(edges)
    labels = [f"({edges[i]:.0%}, {edges[i + 1]:.0%}]" for i in range(len(edges) - 1)]
    # `include_lowest=False` keeps the zero-agreement genes -- the ones no fit
    # ever calls significant -- in their own row rather than folded into the
    # weakest bin.
    data["agreement_bin"] = pd.cut(
        data[agreement_column], bins=edges, labels=labels, include_lowest=False
    )
    data["agreement_bin"] = data["agreement_bin"].cat.add_categories(["0%"]).fillna("0%")
    data["agreement_bin"] = data["agreement_bin"].cat.reorder_categories(
        ["0%"] + labels, ordered=True
    )

    data["abs_zscore"] = data["zscore"].abs()
    if "zscore_sd" in data.columns:
        with np.errstate(divide="ignore", invalid="ignore"):
            data["stability"] = data["abs_zscore"] / data["zscore_sd"]

    rows = []
    for label, group in data.groupby("agreement_bin", observed=False, sort=True):
        row = {
            "agreement_bin": str(label),
            "n_genes": int(len(group)),
            "n_draws_significant_median": _median(group.get(
                agreement_column.replace("agreement_", "n_draws_significant_")
            )),
        }
        if "block_index" in group.columns:
            assigned = group.loc[group["block_index"] >= 0, "block_index"]
            row["n_ld_blocks"] = int(assigned.nunique())
            row["genes_per_ld_block"] = (
                len(assigned) / assigned.nunique() if assigned.nunique() else float("nan")
            )
        for column, name in (
            ("abs_zscore", "abs_zscore"),
            ("zscore_sd", "zscore_sd"),
            ("stability", "stability"),
            ("best_gwas_p", "best_gwas_p"),
            ("mean_n_snps_used", "n_snps_used"),
            ("n_snps_used", "n_snps_used"),
        ):
            if column in group.columns and name + "_median" not in row:
                row[f"{name}_median"] = _median(group[column])
                row[f"{name}_mean"] = _mean(group[column])
        row["frac_significant_pooled"] = (
            float(group["significant_bonferroni"].mean())
            if "significant_bonferroni" in group.columns
            else float("nan")
        )
        rows.append(row)
    return pd.DataFrame(rows)


def agreement_summary(
    frame: pd.DataFrame, agreement_column: str = "agreement_bonferroni"
) -> dict:
    """Scalar model-agreement statistics over the genes any fit calls significant."""
    if agreement_column not in frame.columns:
        return {}
    agreement = frame[agreement_column]
    ever = frame[agreement.fillna(0.0) > 0.0]
    summary = {
        "n_genes_significant_in_any_draw": int(len(ever)),
        "n_genes_significant_in_all_draws": int((agreement >= 1.0).sum()),
        "n_genes_agreement_at_least_80pct": int((agreement >= 0.8).sum()),
        "n_genes_agreement_at_least_50pct": int((agreement >= 0.5).sum()),
    }
    if len(ever):
        summary["mean_agreement_among_ever_significant"] = float(
            ever[agreement_column].mean()
        )
        summary["median_agreement_among_ever_significant"] = float(
            ever[agreement_column].median()
        )
    return summary


def _median(values) -> float:
    return _reduce(values, "median")


def _mean(values) -> float:
    return _reduce(values, "mean")


def _reduce(values, how: str) -> float:
    """`values.median()`/`.mean()`, but NaN rather than a warning when empty."""
    if values is None or len(values) == 0:
        return float("nan")
    numeric = pd.to_numeric(values, errors="coerce").dropna()
    if numeric.empty:
        return float("nan")
    return float(getattr(numeric, how)())


def draw_spread(long: pd.DataFrame) -> pd.DataFrame:
    """
    Per-draw stability diagnostics: how far each draw's z-scores sit from the
    across-draw mean, plus each draw's own inflation.
    """
    means = long.groupby("gene")["zscore"].transform("mean")
    residuals = long["zscore"] - means
    frame = long.assign(residual=residuals)
    return (
        frame.groupby("draw", sort=False)
        .agg(
            n_genes=("gene", "count"),
            mean_zscore=("zscore", "mean"),
            sd_zscore=("zscore", "std"),
            rmse_vs_mean=("residual", lambda values: float(np.sqrt(np.mean(values**2)))),
        )
        .reset_index()
    )


def summarize(
    frame: pd.DataFrame, fdr: float = 0.05, extra: Optional[dict] = None
) -> dict:
    """Scalar statistics for one cell type's final table."""
    pvalues = frame["pvalue"].to_numpy(dtype=float)
    tested = int(np.isfinite(pvalues).sum())
    summary = {
        "n_genes": int(len(frame)),
        "n_genes_tested": tested,
        "n_significant_fdr": int(frame.get("significant_fdr", pd.Series(dtype=bool)).sum()),
        "n_significant_bonferroni": int(
            frame.get("significant_bonferroni", pd.Series(dtype=bool)).sum()
        ),
        "fdr_level": fdr,
        "lambda_gc": genomic_inflation(pvalues),
        "min_pvalue": float(np.nanmin(pvalues)) if tested else float("nan"),
        "max_abs_zscore": (
            float(np.nanmax(np.abs(frame["zscore"].to_numpy(dtype=float))))
            if tested
            else float("nan")
        ),
    }
    if "n_snps_used" in frame.columns:
        summary["mean_n_snps_used"] = float(frame["n_snps_used"].mean())
    elif "mean_n_snps_used" in frame.columns:
        summary["mean_n_snps_used"] = float(frame["mean_n_snps_used"].mean())
    if "zscore_sd" in frame.columns:
        sds = frame["zscore_sd"].to_numpy(dtype=float)
        if np.isfinite(sds).any():
            summary["mean_zscore_sd_significant"] = float(np.nanmean(sds))
            summary["max_zscore_sd_significant"] = float(np.nanmax(sds))
    if extra:
        summary.update(extra)
    return summary


def _present(frame: pd.DataFrame, columns: Sequence[str]) -> list[str]:
    return [column for column in columns if column in frame.columns]


__all__ = [
    "AGREEMENT_THRESHOLDS",
    "aggregate_draws",
    "agreement_strata",
    "agreement_summary",
    "annotate_significance",
    "benjamini_hochberg",
    "draw_spread",
    "genomic_inflation",
    "summarize",
]
