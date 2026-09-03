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

Z_CI_MULTIPLIER = 1.959963984540054  # two-sided 95%


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
        subset = frame[columns + _present(frame, ["zscore", "effect_size", "n_snps_used"])].copy()
        subset["draw"] = draw_id
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
    aggregated = long.groupby("gene", sort=False).agg(**specification).reset_index()

    if "gene_name" in long.columns:
        names = long.drop_duplicates("gene").set_index("gene")["gene_name"]
        aggregated["gene_name"] = aggregated["gene"].map(names)

    # A single draw has no between-fit spread to report.
    aggregated.loc[aggregated["n_draws"] < 2, "zscore_var"] = np.nan
    aggregated["zscore_sd"] = np.sqrt(aggregated["zscore_var"])

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
        "mean_n_snps_used",
        "significant_fdr",
        "significant_bonferroni",
        "bonferroni_threshold",
    ]
    aggregated = aggregated[[c for c in ordered if c in aggregated.columns]]
    return aggregated, long


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
    "aggregate_draws",
    "annotate_significance",
    "benjamini_hochberg",
    "draw_spread",
    "genomic_inflation",
    "summarize",
]
