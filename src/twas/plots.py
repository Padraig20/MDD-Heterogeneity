from __future__ import annotations

import logging
from typing import Optional

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import MaxNLocator

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


def _find_y_break(
    y: np.ndarray, floor: float, min_gap_fraction: float = 0.35
) -> Optional[tuple[float, float]]:
    """
    The widest empty band on the y-axis worth breaking, or None.

    A handful of enormous associations (an MHC hit will do it on its own)
    stretch the axis so far that everything at the significance threshold is
    squashed into the bottom centimetre. Cutting the empty band out gives that
    region the room back.

    Only bands above `floor` -- the significance line -- are considered, so the
    break can never fall through the part of the plot being read. The band also
    has to be wide enough to be worth the discontinuity, and the points above it
    few enough to really be a detached tail rather than the top of a dense
    column that happens to have a hole in it.
    """
    finite = y[np.isfinite(y)]
    values = np.unique(finite[finite > floor])
    if values.size < 2:
        return None

    gaps = np.diff(values)
    index = int(np.argmax(gaps))
    low, high = float(values[index]), float(values[index + 1])
    ymax = float(values[-1])
    if (high - low) < max(min_gap_fraction * ymax, 5.0):
        return None
    if int((finite > low).sum()) > max(10, 0.02 * finite.size):
        return None
    return low, high


def _draw_axis_break(upper: plt.Axes, lower: plt.Axes) -> None:
    """The slanted tick pair marking the discontinuity on both spines."""
    upper.spines["bottom"].set_visible(False)
    lower.spines["top"].set_visible(False)
    upper.tick_params(axis="x", which="both", bottom=False, labelbottom=False)
    marks = dict(
        marker=[(-1, -0.5), (1, 0.5)], markersize=9, linestyle="none",
        color="k", mec="k", mew=1, clip_on=False,
    )
    upper.plot([0, 1], [0, 0], transform=upper.transAxes, **marks)
    lower.plot([0, 1], [1, 1], transform=lower.transAxes, **marks)


def manhattan(
    frame: pd.DataFrame,
    positions: dict[str, tuple[str, int]],
    cell_type: str,
    fdr: float = 0.05,
    label_top: int = 10,
    y_break: Optional[tuple[float, float]] = None,
    allow_y_break: bool = True,
) -> Optional[plt.Figure]:
    """
    Gene-level Manhattan plot, using each gene's cis-window midpoint in the LD
    reference as its x position.

    When a few associations tower over the rest the y-axis is broken so the
    signals near the significance threshold stay readable. Pass an explicit
    `(low, high)` as `y_break` to place the cut by hand, or `allow_y_break=False`
    for a plain continuous axis.
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

    lines = []
    threshold = data["bonferroni_threshold"].dropna()
    if not threshold.empty:
        lines.append(
            (-np.log10(float(threshold.iloc[0])), "firebrick", "--", "Bonferroni (0.05)")
        )
    significant = data[data.get("significant_fdr", False) == True]  # noqa: E712
    if not significant.empty:
        lines.append((
            float(_neg_log10(significant["pvalue"].to_numpy(dtype=float)).min()),
            "steelblue", ":", f"BH FDR {fdr:g}",
        ))

    # Keep the break clear of the significance lines, so the cut can never fall
    # through the band the plot is read in.
    floor = max([value for value, *_ in lines], default=float(np.nanmedian(y)))
    if y_break is None and allow_y_break:
        y_break = _find_y_break(y, floor)
    elif y_break is not None:
        hidden = int(((y > y_break[0]) & (y < y_break[1])).sum())
        if hidden:
            logging.warning(
                "The requested y-axis break %s falls across %d gene(s), which "
                "will not appear in either panel.", y_break, hidden,
            )

    if y_break is None:
        fig, lower = plt.subplots(figsize=(12, 5))
        upper = None
        axes = [lower]
    else:
        # Constrained rather than tight layout: it is the one that knows how to
        # place `supylabel` across a stack of axes.
        fig, (upper, lower) = plt.subplots(
            2, 1, figsize=(12, 5.6), sharex=True,
            height_ratios=[1, 3], layout="constrained",
        )
        fig.get_layout_engine().set(hspace=0.02)
        axes = [upper, lower]

    for ax in axes:
        for index, (_, block) in enumerate(data.groupby("chrom_key", sort=True)):
            ax.scatter(
                x[block.index],
                y[block.index],
                s=6,
                alpha=0.7,
                color="0.30" if index % 2 == 0 else "0.60",
            )
        for value, color, style, label in lines:
            ax.axhline(value, color=color, linestyle=style, linewidth=1, label=label)

    if y_break is None:
        lower.set_ylim(bottom=0)
    else:
        low, high = y_break
        # A little air on either side of the cut so no marker sits on a spine.
        lower.set_ylim(0, low * 1.06)
        upper.set_ylim(high - 0.06 * (float(np.nanmax(y)) - high + 1.0), float(np.nanmax(y)) * 1.04)
        upper.yaxis.set_major_locator(MaxNLocator(nbins=3, integer=True))
        _draw_axis_break(upper, lower)

    if label_top:
        top = data.nsmallest(label_top, "pvalue")
        for row_index, row in top.iterrows():
            # Annotate on whichever side of the break the gene actually landed.
            target = (
                upper if y_break is not None and y[row_index] >= y_break[1] else lower
            )
            target.annotate(
                row.get("gene_name") or row["gene"],
                (x[row_index], y[row_index]),
                fontsize=7,
                xytext=(0, 4),
                textcoords="offset points",
                ha="center",
            )

    lower.set_xticks(ticks)
    lower.set_xticklabels(tick_labels, fontsize=8)
    lower.set_xlabel("Chromosome")
    lower.set_xlim(-cumulative * 0.01, cumulative * 1.01)

    title = f"{cell_type} — S-PrediXcan gene-level association"
    if lower.get_legend_handles_labels()[0]:
        lower.legend(loc="upper right", fontsize=8)
    if y_break is None:
        lower.set_ylabel(r"$-\log_{10}$ p")
        lower.set_title(title)
        fig.tight_layout()
    else:
        fig.supylabel(r"$-\log_{10}$ p")
        upper.set_title(title)
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


def agreement_histogram(
    frame: pd.DataFrame,
    cell_type: str,
    agreement_column: str = "agreement_bonferroni",
) -> Optional[plt.Figure]:
    """
    How many member-bootstrap fits call each gene significant.

    Only genes at least one fit picks up are shown; the overwhelming majority
    that no fit ever calls significant would flatten everything else. A hit list
    worth trusting piles up on the right, near unanimity.
    """
    if agreement_column not in frame.columns or "n_draws" not in frame.columns:
        return None
    counts_column = agreement_column.replace("agreement_", "n_draws_significant_")
    if counts_column not in frame.columns:
        return None

    data = frame[frame[agreement_column].fillna(0.0) > 0.0]
    if data.empty:
        return None

    counts = data[counts_column].to_numpy(dtype=int)
    n_draws = int(frame["n_draws"].max())
    edges = np.arange(0.5, n_draws + 1.5, 1.0)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(counts, bins=edges, color="#3f6fb0", edgecolor="white")
    for threshold, style in ((0.5, ":"), (0.8, "--")):
        ax.axvline(
            threshold * n_draws, color="firebrick", linestyle=style, linewidth=1.2,
            label=f"{threshold:.0%} agreement",
        )
    ax.set_xlabel(f"Member-bootstrap fits calling the gene significant (of {n_draws})")
    ax.set_ylabel("Genes")
    criterion = agreement_column.replace("agreement_", "")
    ax.set_title(f"{cell_type} — MI model agreement ({criterion})")
    ax.text(
        0.03, 0.95,
        f"{len(data):,} gene(s) significant in >=1 fit\n"
        f"{int((data[agreement_column] >= 0.8).sum()):,} at >=80% agreement\n"
        f"{int((data[agreement_column] >= 1.0).sum()):,} unanimous",
        transform=ax.transAxes,
        va="top",
    )
    ax.legend(loc="upper center", fontsize=8)
    fig.tight_layout()
    return fig


def agreement_vs_strength(
    frame: pd.DataFrame,
    cell_type: str,
    agreement_column: str = "agreement_bonferroni",
) -> Optional[plt.Figure]:
    """
    What separates a gene few fits agree on from one they nearly all agree on.

    Left: pooled association strength against agreement — if the cloud rises
    steeply, low agreement simply means a weak gene. Right: the strength of the
    best GWAS variant in the cis window against agreement — if *that* is flat
    while agreement varies, the disagreement is the model reshuffling weight
    among correlated variants rather than a weak locus, which is the regime
    where requiring agreement does real fine-mapping work.
    """
    if agreement_column not in frame.columns:
        return None
    data = frame[frame[agreement_column].notna()].copy()
    data = data[data[agreement_column] > 0.0]
    if data.empty:
        return None

    agreement = data[agreement_column].to_numpy(dtype=float)
    absolute_z = np.abs(data["zscore"].to_numpy(dtype=float))

    has_gwas = "best_gwas_p" in data.columns and data["best_gwas_p"].notna().any()
    fig, axes = plt.subplots(1, 2 if has_gwas else 1, figsize=(12 if has_gwas else 6.5, 5))
    axes = np.atleast_1d(axes)

    sizes = None
    if "mean_n_snps_used" in data.columns:
        snps = data["mean_n_snps_used"].to_numpy(dtype=float)
        sizes = 8.0 + 40.0 * (snps - np.nanmin(snps)) / max(np.ptp(snps[np.isfinite(snps)]), 1e-9)

    axes[0].scatter(agreement, absolute_z, s=sizes if sizes is not None else 16,
                    alpha=0.6, color="#3f6fb0", edgecolor="none")
    axes[0].set_xlabel("Fraction of fits calling the gene significant")
    axes[0].set_ylabel(r"$|\mathrm{E}[z]|$")
    axes[0].set_title("Agreement vs pooled association strength")
    if sizes is not None:
        axes[0].text(0.03, 0.95, "marker size = mean SNPs used",
                     transform=axes[0].transAxes, va="top", fontsize=8)

    if has_gwas:
        gwas = _neg_log10(data["best_gwas_p"].to_numpy(dtype=float))
        axes[1].scatter(agreement, gwas, s=16, alpha=0.6, color="#b06a3f",
                        edgecolor="none")
        axes[1].set_xlabel("Fraction of fits calling the gene significant")
        axes[1].set_ylabel(r"$-\log_{10}$ best GWAS $p$ in the cis window")
        axes[1].set_title("Agreement vs strength of the underlying locus")

    fig.suptitle(f"{cell_type} — what drives MI model agreement")
    fig.tight_layout()
    return fig


def agreement_ld_block_curve(
    curve: pd.DataFrame, cell_type: str, n_blocks_total: Optional[int] = None
) -> Optional[plt.Figure]:
    """
    The fine-mapping argument in one figure.

    Genes and distinct LD blocks are plotted against how much model agreement is
    demanded. Genes falling away while the block count holds means the ensemble
    is pruning redundant genes inside loci rather than losing loci; the
    genes-per-block trace underneath is that same statement as a single number
    heading towards 1.
    """
    if curve is None or curve.empty:
        return None

    thresholds = curve["threshold"].to_numpy(dtype=float)
    fig, (top, bottom) = plt.subplots(
        2, 1, figsize=(8, 7), sharex=True, height_ratios=[2, 1]
    )

    top.plot(thresholds, curve["n_genes"], "o-", color="#3f6fb0", label="Significant genes")
    top.set_ylabel("Genes", color="#3f6fb0")
    top.tick_params(axis="y", labelcolor="#3f6fb0")

    twin = top.twinx()
    twin.plot(thresholds, curve["n_ld_blocks"], "s-", color="#b03f3f",
              label="Distinct LD blocks")
    twin.set_ylabel("LD blocks", color="#b03f3f")
    twin.tick_params(axis="y", labelcolor="#b03f3f")
    twin.set_ylim(bottom=0)
    top.set_ylim(bottom=0)

    total = n_blocks_total or int(curve["n_ld_blocks_total"].iloc[0])
    top.set_title(
        f"{cell_type} — significant genes and LD blocks vs required model agreement\n"
        f"(out of {total:,} pre-defined LD blocks)"
    )
    handles = top.get_legend_handles_labels()[0] + twin.get_legend_handles_labels()[0]
    labels = top.get_legend_handles_labels()[1] + twin.get_legend_handles_labels()[1]
    top.legend(handles, labels, loc="upper right", fontsize=8)

    bottom.plot(thresholds, curve["genes_per_ld_block"], "d-", color="0.3")
    bottom.axhline(1.0, color="grey", linestyle=":", linewidth=1,
                   label="one gene per block")
    bottom.set_ylabel("Genes per LD block")
    bottom.set_xlabel("Required fraction of fits calling the gene significant")
    bottom.set_ylim(bottom=0)
    bottom.legend(loc="upper right", fontsize=8)

    fig.tight_layout()
    return fig


def ld_block_gene_counts(
    frame: pd.DataFrame, cell_type: str, significance_column: str = "significant_bonferroni"
) -> Optional[plt.Figure]:
    """
    How many significant genes each implicated LD block contains.

    A long tail of blocks carrying many genes each is the signature of LD
    dragging neighbours along; a distribution concentrated at one gene per block
    is what a fine-mapped result looks like.
    """
    if "block_index" not in frame.columns or significance_column not in frame.columns:
        return None
    hits = frame[frame[significance_column].fillna(False) & (frame["block_index"] >= 0)]
    if hits.empty:
        return None

    counts = hits["block_index"].value_counts().to_numpy(dtype=int)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.hist(counts, bins=np.arange(0.5, counts.max() + 1.5, 1.0),
            color="#4f7f5f", edgecolor="white")
    ax.set_xlabel("Significant genes in the LD block")
    ax.set_ylabel("LD blocks")
    ax.set_title(f"{cell_type} — significant genes per implicated LD block")
    ax.text(
        0.97, 0.95,
        f"{counts.size:,} block(s), {counts.sum():,} gene(s)\n"
        f"{counts.mean():.2f} genes per block",
        transform=ax.transAxes,
        va="top",
        ha="right",
    )
    fig.tight_layout()
    return fig


def top_genes(frame: pd.DataFrame, n: int = 25) -> pd.DataFrame:
    """The strongest associations, for logging as a table."""
    columns = [
        c
        for c in (
            "gene", "gene_name", "zscore", "pvalue", "qvalue", "effect_size",
            "zscore_sd", "n_snps_used", "mean_n_snps_used", "n_draws",
            "n_draws_significant_bonferroni", "agreement_bonferroni",
            "best_gwas_p", "block",
        )
        if c in frame.columns
    ]
    return frame.nsmallest(n, "pvalue")[columns].reset_index(drop=True)


__all__ = [
    "agreement_histogram",
    "agreement_ld_block_curve",
    "agreement_vs_strength",
    "ld_block_gene_counts",
    "manhattan",
    "mi_draw_spread",
    "mi_draw_summary",
    "mi_stability",
    "qq",
    "top_genes",
    "volcano",
    "zscore_histogram",
]
