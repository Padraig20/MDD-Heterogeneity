from __future__ import annotations

from typing import Optional

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.twas.aggregate import genomic_inflation

"""
plots.py

The standard TWAS figures for one cell type, plus the multiple-imputation
diagnostics.

Every function returns a matplotlib figure (or None when there is nothing to
draw); the caller owns saving and closing it.
"""

CHROM_ORDER = [str(c) for c in range(1, 23)]
MAX_LOG10P = 320.0  # beyond float64's smallest normal p-value


def _neg_log10(pvalues: np.ndarray) -> np.ndarray:
    with np.errstate(divide="ignore", invalid="ignore"):
        values = -np.log10(pvalues)
    return np.clip(values, None, MAX_LOG10P)


def _chrom_key(chrom: str) -> int:
    text = str(chrom).removeprefix("chr")
    return int(text) if text.isdigit() else 99


def manhattan(
    frame: pd.DataFrame,
    positions: dict[str, tuple[str, int]],
    cell_type: str,
    fdr: float = 0.05,
    label_top: int = 10,
) -> Optional[plt.Figure]:
    """
    Gene-level Manhattan plot, using each gene's cis-window midpoint in the LD
    reference as its x position.
    """
    data = frame.dropna(subset=["pvalue"]).copy()
    data["chrom"] = data["gene"].map(lambda gene: positions.get(gene, (None, None))[0])
    data["bp"] = data["gene"].map(lambda gene: positions.get(gene, (None, None))[1])
    data = data.dropna(subset=["chrom", "bp"])
    if data.empty:
        return None

    data["chrom_key"] = data["chrom"].map(_chrom_key)
    data = data.sort_values(["chrom_key", "bp"]).reset_index(drop=True)

    offsets: dict[int, float] = {}
    ticks: list[float] = []
    tick_labels: list[str] = []
    cumulative = 0.0
    x = np.empty(len(data))
    for chrom_key, block in data.groupby("chrom_key", sort=True):
        offsets[chrom_key] = cumulative
        span = float(block["bp"].max()) or 1.0
        x[block.index] = cumulative + block["bp"].to_numpy(dtype=float)
        ticks.append(cumulative + span / 2.0)
        tick_labels.append(str(block["chrom"].iloc[0]).removeprefix("chr"))
        cumulative += span * 1.02

    y = _neg_log10(data["pvalue"].to_numpy(dtype=float))

    fig, ax = plt.subplots(figsize=(12, 5))
    for index, (chrom_key, block) in enumerate(data.groupby("chrom_key", sort=True)):
        ax.scatter(
            x[block.index],
            y[block.index],
            s=6,
            alpha=0.7,
            color="0.30" if index % 2 == 0 else "0.60",
        )

    threshold = data["bonferroni_threshold"].dropna()
    if not threshold.empty:
        ax.axhline(
            -np.log10(float(threshold.iloc[0])),
            color="firebrick",
            linestyle="--",
            linewidth=1,
            label="Bonferroni (0.05)",
        )
    significant = data[data.get("significant_fdr", False) == True]  # noqa: E712
    if not significant.empty:
        ax.axhline(
            float(_neg_log10(significant["pvalue"].to_numpy(dtype=float)).min()),
            color="steelblue",
            linestyle=":",
            linewidth=1,
            label=f"BH FDR {fdr:g}",
        )

    if label_top:
        top = data.nsmallest(label_top, "pvalue")
        for row_index, row in top.iterrows():
            ax.annotate(
                row.get("gene_name") or row["gene"],
                (x[row_index], y[row_index]),
                fontsize=7,
                xytext=(0, 4),
                textcoords="offset points",
                ha="center",
            )

    ax.set_xticks(ticks)
    ax.set_xticklabels(tick_labels, fontsize=8)
    ax.set_xlabel("Chromosome")
    ax.set_ylabel(r"$-\log_{10}$ p")
    ax.set_title(f"{cell_type} — S-PrediXcan gene-level association")
    ax.set_xlim(-cumulative * 0.01, cumulative * 1.01)
    if ax.get_legend_handles_labels()[0]:
        ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    return fig


def qq(frame: pd.DataFrame, cell_type: str) -> Optional[plt.Figure]:
    """QQ plot of the gene-level p-values against the uniform null."""
    pvalues = frame["pvalue"].to_numpy(dtype=float)
    pvalues = np.sort(pvalues[np.isfinite(pvalues) & (pvalues > 0)])
    if pvalues.size == 0:
        return None

    n = pvalues.size
    expected = _neg_log10((np.arange(1, n + 1) - 0.5) / n)
    observed = _neg_log10(pvalues)
    lambda_gc = genomic_inflation(pvalues)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(expected, observed, s=6, alpha=0.6)
    limit = float(max(expected.max(), observed.max())) * 1.05
    ax.plot([0, limit], [0, limit], "k--", linewidth=1, label="null")
    ax.set_xlim(0, limit)
    ax.set_ylim(0, limit)
    ax.set_xlabel(r"Expected $-\log_{10}$ p")
    ax.set_ylabel(r"Observed $-\log_{10}$ p")
    ax.set_title(f"{cell_type} — TWAS QQ")
    ax.text(
        0.03, 0.95,
        f"$\\lambda_{{GC}}$ = {lambda_gc:.3f}\n{n:,} genes",
        transform=ax.transAxes,
        va="top",
    )
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    return fig


def volcano(
    frame: pd.DataFrame, cell_type: str, fdr: float = 0.05, label_top: int = 10
) -> Optional[plt.Figure]:
    """Direction against strength: predicted-expression effect size vs p-value."""
    data = frame.dropna(subset=["pvalue", "effect_size"]).reset_index(drop=True)
    if data.empty:
        return None

    y = _neg_log10(data["pvalue"].to_numpy(dtype=float))
    effect = data["effect_size"].to_numpy(dtype=float)
    significant = data.get("significant_fdr", pd.Series(False, index=data.index)).to_numpy(
        dtype=bool
    )

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(effect[~significant], y[~significant], s=6, alpha=0.4, color="0.6", label="ns")
    if significant.any():
        ax.scatter(
            effect[significant],
            y[significant],
            s=10,
            alpha=0.8,
            color="firebrick",
            label=f"BH FDR < {fdr:g}",
        )
    ax.axvline(0.0, color="grey", linestyle=":", linewidth=1)

    if label_top:
        top = data.nsmallest(label_top, "pvalue")
        for row_index, row in top.iterrows():
            ax.annotate(
                row.get("gene_name") or row["gene"],
                (effect[row_index], y[row_index]),
                fontsize=7,
                xytext=(0, 4),
                textcoords="offset points",
                ha="center",
            )

    ax.set_xlabel("Effect size (per unit predicted expression)")
    ax.set_ylabel(r"$-\log_{10}$ p")
    ax.set_title(f"{cell_type} — TWAS volcano")
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    return fig


def zscore_histogram(frame: pd.DataFrame, cell_type: str) -> Optional[plt.Figure]:
    """Distribution of gene-level z-scores against the standard normal null."""
    zscores = frame["zscore"].to_numpy(dtype=float)
    zscores = zscores[np.isfinite(zscores)]
    if zscores.size == 0:
        return None

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.hist(zscores, bins=60, density=True, edgecolor="white", color="0.45")
    grid = np.linspace(zscores.min(), zscores.max(), 300)
    ax.plot(
        grid,
        np.exp(-0.5 * grid**2) / np.sqrt(2 * np.pi),
        "r--",
        linewidth=1,
        label="N(0, 1)",
    )
    ax.set_xlabel("TWAS z-score")
    ax.set_ylabel("Density")
    ax.set_title(f"{cell_type} — gene-level z-scores")
    ax.text(
        0.03, 0.95,
        f"mean = {zscores.mean():.3f}\nsd = {zscores.std(ddof=1):.3f}\n"
        f"n_genes = {zscores.size:,}",
        transform=ax.transAxes,
        va="top",
    )
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    return fig


def mi_stability(frame: pd.DataFrame, cell_type: str) -> Optional[plt.Figure]:
    """
    Between-fit spread against association strength, for the genes where the
    expectation is significant (the only ones `zscore_sd` is reported for).
    """
    data = frame.dropna(subset=["zscore_sd", "zscore"])
    if data.empty:
        return None

    absolute_z = np.abs(data["zscore"].to_numpy(dtype=float))
    sds = data["zscore_sd"].to_numpy(dtype=float)

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(absolute_z, sds, s=14, alpha=0.7)
    limit = float(absolute_z.max()) * 1.05
    ax.plot([0, limit], [0, limit], "k--", linewidth=1, label="sd = |E[z]|")
    ax.set_xlim(0, limit)
    ax.set_xlabel(r"$|\mathrm{E}[z]|$ across member-bootstrap fits")
    ax.set_ylabel(r"$\mathrm{sd}[z]$ across member-bootstrap fits")
    ax.set_title(f"{cell_type} — MI stability of significant genes")
    ax.text(
        0.03, 0.95,
        f"{len(data):,} significant gene(s)\nmean sd = {np.nanmean(sds):.3f}",
        transform=ax.transAxes,
        va="top",
    )
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    return fig


def mi_draw_spread(long: pd.DataFrame, cell_type: str, top_n: int = 20) -> Optional[plt.Figure]:
    """Per-draw z-scores of the strongest genes, one box per gene."""
    if long.empty:
        return None

    means = long.groupby("gene")["zscore"].mean()
    top_genes = means.abs().nlargest(top_n).index.tolist()
    if not top_genes:
        return None
    ordered = means.loc[top_genes].sort_values().index.tolist()

    series = [
        long.loc[long["gene"] == gene, "zscore"].dropna().to_numpy(dtype=float)
        for gene in ordered
    ]
    series = [values for values in series if values.size]
    if not series:
        return None

    names = long.drop_duplicates("gene").set_index("gene").get("gene_name")
    labels = [
        (names.get(gene) if names is not None else None) or gene for gene in ordered
    ]

    fig, ax = plt.subplots(figsize=(9, max(4.0, 0.32 * len(series) + 2.0)))
    ax.boxplot(series, vert=False, showmeans=True, meanline=True, widths=0.6,
               tick_labels=labels, flierprops=dict(marker=".", markersize=3, alpha=0.5))
    ax.axvline(0.0, color="grey", linestyle=":", linewidth=1)
    ax.set_xlabel("TWAS z-score per member-bootstrap fit")
    ax.set_title(f"{cell_type} — z-score spread across MI draws (top {len(series)})")
    ax.tick_params(axis="y", labelsize=7)
    fig.tight_layout()
    return fig


def mi_draw_summary(spread: pd.DataFrame, cell_type: str) -> Optional[plt.Figure]:
    """How far each individual draw sits from the across-draw consensus."""
    if spread.empty:
        return None

    fig, ax = plt.subplots(figsize=(9, 5))
    positions = np.arange(len(spread))
    ax.bar(positions, spread["rmse_vs_mean"].to_numpy(dtype=float), color="0.45")
    ax.set_xticks(positions)
    ax.set_xticklabels(spread["draw"].tolist(), rotation=90, fontsize=6)
    ax.set_ylabel("RMSE of z vs across-draw mean")
    ax.set_xlabel("Member-bootstrap fit")
    ax.set_title(f"{cell_type} — per-draw deviation from the MI consensus")
    fig.tight_layout()
    return fig


def top_genes(frame: pd.DataFrame, n: int = 25) -> pd.DataFrame:
    """The strongest associations, for logging as a table."""
    columns = [
        c
        for c in (
            "gene", "gene_name", "zscore", "pvalue", "qvalue", "effect_size",
            "zscore_sd", "n_snps_used", "mean_n_snps_used", "n_draws",
        )
        if c in frame.columns
    ]
    return frame.nsmallest(n, "pvalue")[columns].reset_index(drop=True)


__all__ = [
    "manhattan",
    "mi_draw_spread",
    "mi_draw_summary",
    "mi_stability",
    "qq",
    "top_genes",
    "volcano",
    "zscore_histogram",
]
