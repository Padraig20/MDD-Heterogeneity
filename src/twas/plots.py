from __future__ import annotations

import logging
from typing import Optional

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.cbook import boxplot_stats
from matplotlib.ticker import MaxNLocator
from matplotlib.transforms import blended_transform_factory

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


class _GenomeLayout:
    """Genes placed along one genome-wide x axis, with chromosome ticks."""

    def __init__(self, data: pd.DataFrame, x: np.ndarray, ticks: list[float],
                 tick_labels: list[str], span: float) -> None:
        self.data = data
        self.x = x
        self.ticks = ticks
        self.tick_labels = tick_labels
        self.span = span


def _genome_layout(
    frame: pd.DataFrame, positions: dict[str, tuple[str, int]]
) -> Optional[_GenomeLayout]:
    """
    Lay genes out along the genome by cis-window midpoint, or None if none of
    them can be placed.

    Shared by the two Manhattan plots so that a gene sits at the same x in
    both, and the chromosome ticks line up between them.
    """
    data = frame.dropna(subset=["pvalue"]).copy()
    data["chrom"] = data["gene"].map(lambda gene: positions.get(gene, (None, None))[0])
    data["bp"] = data["gene"].map(lambda gene: positions.get(gene, (None, None))[1])
    data = data.dropna(subset=["chrom", "bp"])
    if data.empty:
        return None

    data["chrom_key"] = data["chrom"].map(_chrom_key)
    data = data.sort_values(["chrom_key", "bp"]).reset_index(drop=True)

    ticks: list[float] = []
    tick_labels: list[str] = []
    cumulative = 0.0
    x = np.empty(len(data))
    for _, block in data.groupby("chrom_key", sort=True):
        span = float(block["bp"].max()) or 1.0
        x[block.index] = cumulative + block["bp"].to_numpy(dtype=float)
        ticks.append(cumulative + span / 2.0)
        tick_labels.append(str(block["chrom"].iloc[0]).removeprefix("chr"))
        cumulative += span * 1.02
    return _GenomeLayout(data, x, ticks, tick_labels, cumulative)


def _setup_manhattan_axes(
    y_for_break: np.ndarray,
    lines: list[tuple],
    y_break: Optional[tuple[float, float]],
    allow_y_break: bool,
    figsize: tuple[float, float],
) -> tuple[plt.Figure, Optional[plt.Axes], plt.Axes, list[plt.Axes], Optional[tuple[float, float]]]:
    """
    The shared Manhattan figure: one axis, or a broken pair, with the
    significance-line floor already applied to the break search.
    """
    floor = max([value for value, *_ in lines], default=float(np.nanmedian(y_for_break)))
    if y_break is None and allow_y_break:
        y_break = _find_y_break(y_for_break, floor)

    if y_break is None:
        fig, lower = plt.subplots(figsize=figsize)
        return fig, None, lower, [lower], None

    fig, (upper, lower) = plt.subplots(
        2, 1, figsize=(figsize[0], figsize[1] + 0.6), sharex=True,
        height_ratios=[1, 3], layout="constrained",
    )
    fig.get_layout_engine().set(hspace=0.02)
    return fig, upper, lower, [upper, lower], y_break


def _finish_manhattan(
    fig: plt.Figure,
    upper: Optional[plt.Axes],
    lower: plt.Axes,
    layout: _GenomeLayout,
    y_break: Optional[tuple[float, float]],
    ymax: float,
    title: str,
) -> plt.Figure:
    """Chromosome ticks, limits, labels and the optional axis break."""
    if y_break is None:
        lower.set_ylim(bottom=0)
    else:
        low, high = y_break
        lower.set_ylim(0, low * 1.06)
        upper.set_ylim(high - 0.06 * (ymax - high + 1.0), ymax * 1.04)
        upper.yaxis.set_major_locator(MaxNLocator(nbins=3, integer=True))
        _draw_axis_break(upper, lower)

    lower.set_xticks(layout.ticks)
    lower.set_xticklabels(layout.tick_labels, fontsize=8)
    lower.set_xlabel("Chromosome")
    lower.set_xlim(-layout.span * 0.01, layout.span * 1.01)
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


GENE_SETS = ("expectation", "all", "any")
GENE_SET_TITLES = {
    "expectation": "E[z] significant",
    "all": "significant in every MI fit",
    "any": "significant in at least one MI fit",
}


def _gene_set_mask(
    data: pd.DataFrame, criterion: str, gene_set: str
) -> Optional[np.ndarray]:
    """
    Which genes a Manhattan boxplot is drawn for.

    `expectation` is the call on E[z] -- what the ordinary TWAS table reports.
    `all` / `any` are the two ends of the MI agreement: every member-bootstrap
    fit independently significant, or at least one. Those two are only defined
    when the agreement columns are present, i.e. after an MI aggregation.
    """
    gene_set = gene_set.lower()
    if gene_set not in GENE_SETS:
        raise ValueError(
            f"Unknown gene set {gene_set!r}; expected one of {GENE_SETS}."
        )
    if gene_set == "expectation":
        flags = data.get(f"significant_{criterion}")
        if flags is None:
            return None
        return flags.fillna(False).to_numpy(dtype=bool)

    agreement = data.get(f"agreement_{criterion}")
    if agreement is not None:
        fraction = agreement.fillna(0.0).to_numpy(dtype=float)
    else:
        counts = data.get(f"n_draws_significant_{criterion}")
        n_draws = data.get("n_draws")
        if counts is None or n_draws is None:
            return None
        draws = n_draws.to_numpy(dtype=float)
        fraction = np.divide(
            counts.fillna(0).to_numpy(dtype=float),
            draws,
            out=np.zeros(len(data), dtype=float),
            where=draws > 0,
        )
    if gene_set == "all":
        return fraction >= (1.0 - 1e-12)
    return fraction > 0.0


def _gene_label(row) -> str:
    """Prefer the GTF symbol; fall back to a versionless Ensembl id."""
    name = row["gene_name"] if "gene_name" in row.index else None
    if name is not None and pd.notna(name):
        text = str(name).strip()
        if text and text.lower() not in {"nan", "none"} and not text.upper().startswith("ENSG"):
            return text
    return str(row["gene"]).split(".")[0]


def _ld_block_runs(block_ids: list[int]) -> list[tuple[int, int, int]]:
    """Inclusive `(start, end, block_id)` runs of the same assigned LD block."""
    runs: list[tuple[int, int, int]] = []
    if not block_ids:
        return runs
    start = 0
    for i in range(1, len(block_ids) + 1):
        if i == len(block_ids) or block_ids[i] != block_ids[start]:
            block_id = block_ids[start]
            if block_id is not None and int(block_id) >= 0:
                runs.append((start, i - 1, int(block_id)))
            start = i
    return runs


def _draw_ld_block_braces(
    ax: plt.Axes, box_x: np.ndarray, block_ids: list[int]
) -> bool:
    """
    Square under-brackets grouping neighbouring genes that share an LD block.

    Drawn in mixed data/axes coordinates just below the spine so the y-scale
    of the p-values is left alone. Single-gene blocks get a short tick;
    runs of two or more get a bracket spanning them.
    """
    runs = _ld_block_runs(block_ids)
    if not runs:
        return False
    transform = blended_transform_factory(ax.transData, ax.transAxes)
    y_top, y_bot = -0.10, -0.16
    for start, end, _block_id in runs:
        x0, x1 = float(box_x[start]), float(box_x[end])
        if start == end:
            ax.plot(
                [x0, x0], [y_top, y_bot],
                transform=transform, color="0.40", lw=0.8,
                clip_on=False, solid_capstyle="butt",
            )
        else:
            ax.plot(
                [x0, x0, x1, x1], [y_top, y_bot, y_bot, y_top],
                transform=transform, color="0.40", lw=0.9,
                clip_on=False, solid_capstyle="butt",
            )
    return True


def _significance_lines(data: pd.DataFrame, fdr: float) -> list[tuple]:
    """The Bonferroni and BH threshold lines, as `(y, colour, style, label)`."""
    lines = []
    if "bonferroni_threshold" in data.columns:
        threshold = data["bonferroni_threshold"].dropna()
        if not threshold.empty:
            lines.append((
                -np.log10(float(threshold.iloc[0])),
                "firebrick", "--", "Bonferroni (0.05)",
            ))
    significant = data[data.get("significant_fdr", False) == True]  # noqa: E712
    if not significant.empty:
        lines.append((
            float(_neg_log10(significant["pvalue"].to_numpy(dtype=float)).min()),
            "steelblue", ":", f"BH FDR {fdr:g}",
        ))
    return lines


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
    layout = _genome_layout(frame, positions)
    if layout is None:
        return None
    data, x = layout.data, layout.x

    y = _neg_log10(data["pvalue"].to_numpy(dtype=float))
    lines = _significance_lines(data, fdr)
    fig, upper, lower, axes, y_break = _setup_manhattan_axes(
        y, lines, y_break, allow_y_break, figsize=(12, 5)
    )

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

    if label_top:
        top = data.nsmallest(label_top, "pvalue")
        for row_index, row in top.iterrows():
            # Annotate on whichever side of the break the gene actually landed.
            target = (
                upper if y_break is not None and y[row_index] >= y_break[1] else lower
            )
            target.annotate(
                _gene_label(row),
                (x[row_index], y[row_index]),
                fontsize=7,
                xytext=(0, 4),
                textcoords="offset points",
                ha="center",
            )

    return _finish_manhattan(
        fig, upper, lower, layout, y_break, float(np.nanmax(y)),
        f"{cell_type} — S-PrediXcan gene-level association",
    )


def manhattan_boxplots(
    frame: pd.DataFrame,
    positions: dict[str, tuple[str, int]],
    cell_type: str,
    fdr: float = 0.05,
    criterion: str = "fdr",
    gene_set: str = "expectation",
    y_break: Optional[tuple[float, float]] = None,
    allow_y_break: bool = True,
) -> Optional[plt.Figure]:
    """
    The Manhattan layout, but each selected gene is folded into a chromosome
    boxplot of -log10 p instead of being drawn as its own point.

    `gene_set` chooses which genes go in the boxes: the E[z] call
    (`expectation`), genes every MI fit called significant (`all`), or genes
    at least one fit called (`any`). The last two are the two ends of the
    agreement spectrum and only exist after an MI aggregation.

    A cloud of hits at one locus is hard to read as dots -- they stack, and the
    only thing the eye takes away is that the chromosome is busy. The box shows
    the distribution: a tight box sitting well above the threshold is a
    chromosome of uniformly strong associations, a box whose whisker just
    clears it is a chromosome carried by one or two genes. Genes outside the
    selected set stay as faint dots so the rest of the genome is still in view.

    Positions and chromosome ticks come from the same layout as `manhattan`,
    so the figures overlay.
    """
    layout = _genome_layout(frame, positions)
    if layout is None:
        return None
    data, x = layout.data, layout.x

    significant = _gene_set_mask(data, criterion, gene_set)
    if significant is None:
        return None
    if not significant.any():
        logging.info(
            "No gene of '%s' is %s at %s, so the boxplot Manhattan was skipped.",
            cell_type, GENE_SET_TITLES[gene_set.lower()], criterion,
        )
        return None

    background = _neg_log10(data["pvalue"].to_numpy(dtype=float))
    boxes, box_x, box_colors = [], [], []
    for index, (_, block) in enumerate(data.groupby("chrom_key", sort=True)):
        hit = block.index[significant[block.index]]
        if not len(hit):
            continue
        boxes.append(background[hit])
        box_x.append(float(np.median(x[hit])))
        box_colors.append("0.30" if index % 2 == 0 else "0.60")
    if not boxes:
        return None

    stats = boxplot_stats(boxes)
    spread = np.concatenate(boxes)
    ymax = float(max(np.nanmax(spread), np.nanmax(background)))
    lines = _significance_lines(data, fdr)
    fig, upper, lower, axes, y_break = _setup_manhattan_axes(
        spread, lines, y_break, allow_y_break, figsize=(12, 5)
    )

    # Wide enough to read as a box, narrow enough that neighbouring
    # chromosomes do not collide. The floor is a handful of genes' worth of
    # the genome so a chromosome with a single hit is still visible.
    box_width = max(layout.span / 40.0, layout.span / max(len(boxes) * 3.0, 1.0))

    for ax in axes:
        ax.scatter(
            x[~significant], background[~significant],
            s=4, alpha=0.25, color="0.75", zorder=1,
        )
        for i, color in enumerate(box_colors):
            ax.bxp(
                [stats[i]],
                positions=[box_x[i]],
                widths=box_width,
                manage_ticks=False,
                patch_artist=True,
                showfliers=True,
                flierprops=dict(
                    marker="o", markersize=3.5, markerfacecolor=color,
                    markeredgecolor="0.20", alpha=0.85,
                ),
                boxprops=dict(facecolor=color, edgecolor="0.20", linewidth=0.8),
                medianprops=dict(color="firebrick", linewidth=1.2),
                whiskerprops=dict(color="0.20", linewidth=0.7),
                capprops=dict(color="0.20", linewidth=0.7),
                zorder=3,
            )
        for value, color, style, label in lines:
            ax.axhline(value, color=color, linestyle=style, linewidth=1,
                       label=label, zorder=2)

    return _finish_manhattan(
        fig, upper, lower, layout, y_break, ymax,
        f"{cell_type} — {len(spread):,} {criterion.upper()} gene(s) "
        f"{GENE_SET_TITLES[gene_set.lower()]} as per-chromosome -log10 p boxplots",
    )


def manhattan_draw_boxplots(
    long: pd.DataFrame,
    frame: pd.DataFrame,
    positions: dict[str, tuple[str, int]],
    cell_type: str,
    fdr: float = 0.05,
    criterion: str = "fdr",
    gene_set: str = "expectation",
    label_top: Optional[int] = None,
    y_break: Optional[tuple[float, float]] = None,
    allow_y_break: bool = True,
) -> Optional[plt.Figure]:
    """
    One boxplot per selected gene of its per-draw -log10 p, in genomic order.

    `gene_set` is the same choice as `manhattan_boxplots`: the E[z] call, the
    genes every fit agreed on, or the genes any fit called. The box is always
    the full set of draws, so a unanimous gene is a box sitting entirely above
    the threshold and an "any" gene that only one fit saw is a box that
    straddles it.

    Genes are spaced evenly rather than at their bp so neighbouring hits in
    the same LD block stay readable as separate boxes. Every selected gene is
    labelled with its GTF symbol. When the result table carries `block_index`
    (i.e. `--ld-blocks` was given), neighbouring genes that share a block
    are grouped with a square bracket under the chromosome ticks.
    """
    if long.empty or "pvalue" not in long.columns:
        return None

    layout = _genome_layout(frame, positions)
    if layout is None:
        return None
    data = layout.data

    selected = _gene_set_mask(data, criterion, gene_set)
    if selected is None:
        return None
    chosen = data.index[selected]
    if not len(chosen):
        logging.info(
            "No gene of '%s' is %s at %s, so the per-draw boxplot Manhattan "
            "was skipped.", cell_type, GENE_SET_TITLES[gene_set.lower()], criterion,
        )
        return None

    per_gene = {
        gene: _neg_log10(values.to_numpy(dtype=float))
        for gene, values in long.dropna(subset=["pvalue"]).groupby("gene")["pvalue"]
    }
    boxes, box_rows, chrom_keys = [], [], []
    for row in chosen:
        values = per_gene.get(data.at[row, "gene"])
        if values is None or values.size == 0:
            continue
        boxes.append(values)
        box_rows.append(row)
        chrom_keys.append(int(data.at[row, "chrom_key"]))
    if not boxes:
        return None

    # Even spacing: one slot per gene, in the genomic order `_genome_layout`
    # already sorted `data` into. Chromosome ticks sit at the midpoint of
    # each chromosome's run of boxes.
    box_x = np.arange(1, len(boxes) + 1, dtype=float)
    ticks, tick_labels = [], []
    chrom_keys_arr = np.asarray(chrom_keys)
    for key in pd.unique(chrom_keys_arr):
        slots = box_x[chrom_keys_arr == key]
        ticks.append(float(slots.mean()))
        label = str(data.at[box_rows[int(np.flatnonzero(chrom_keys_arr == key)[0])], "chrom"])
        tick_labels.append(label.removeprefix("chr"))

    stats = boxplot_stats(boxes)
    spread = np.concatenate(boxes)
    ymax = float(np.nanmax(spread))
    lines = _significance_lines(data, fdr)
    n_labels = len(boxes) if label_top is None else max(0, label_top)
    width_inches = float(np.clip(0.16 * len(boxes) + 8.0, 12.0, 48.0))
    height_inches = 6.6 if n_labels else 5.4
    fig, upper, lower, axes, y_break = _setup_manhattan_axes(
        spread, lines, y_break, allow_y_break,
        figsize=(width_inches, height_inches),
    )
    box_width = 0.65

    for ax in axes:
        for parity, facecolor in ((0, "0.45"), (1, "0.75")):
            indices = [i for i, key in enumerate(chrom_keys) if key % 2 == parity]
            if not indices:
                continue
            ax.bxp(
                [stats[i] for i in indices],
                positions=[box_x[i] for i in indices],
                widths=box_width,
                manage_ticks=False,
                patch_artist=True,
                showfliers=False,
                boxprops=dict(facecolor=facecolor, edgecolor="0.20", linewidth=0.6),
                medianprops=dict(color="firebrick", linewidth=1.1),
                whiskerprops=dict(color="0.20", linewidth=0.6),
                capprops=dict(color="0.20", linewidth=0.6),
                zorder=3,
            )
        for value, color, style, label in lines:
            ax.axhline(value, color=color, linestyle=style, linewidth=1,
                       label=label, zorder=2)

    block_ids = [
        int(data.at[row, "block_index"])
        if "block_index" in data.columns and pd.notna(data.at[row, "block_index"])
        else -1
        for row in box_rows
    ]
    for start, end, block_id in _ld_block_runs(block_ids):
        if end <= start:
            continue
        color = "#4c72b0" if block_id % 2 == 0 else "#dd8452"
        for ax in axes:
            ax.axvspan(
                box_x[start] - 0.45, box_x[end] + 0.45,
                color=color, alpha=0.07, zorder=0, lw=0,
            )

    if y_break is None:
        lower.set_ylim(bottom=0)
    else:
        low, high = y_break
        lower.set_ylim(0, low * 1.06)
        upper.set_ylim(high - 0.06 * (ymax - high + 1.0), ymax * 1.04)
        upper.yaxis.set_major_locator(MaxNLocator(nbins=3, integer=True))
        _draw_axis_break(upper, lower)

    if n_labels:
        # Every selected gene, or the strongest `label_top`, labelled with
        # the GTF symbol. Rotated so neighbouring names stay readable.
        order = list(range(len(boxes)))
        if label_top is not None:
            order = sorted(
                order, key=lambda i: -float(np.median(boxes[i]))
            )[:n_labels]
        fontsize = 5.5 if len(order) > 40 else 7
        for i in order:
            row = box_rows[i]
            top_of_box = float(stats[i]["whishi"])
            target = (
                upper if y_break is not None and top_of_box >= y_break[1] else lower
            )
            target.annotate(
                _gene_label(data.loc[row]),
                (box_x[i], top_of_box),
                fontsize=fontsize,
                rotation=90,
                xytext=(0, 5),
                textcoords="offset points",
                ha="center",
                va="bottom",
                clip_on=False,
            )

    lower.set_xticks(ticks)
    lower.set_xticklabels(tick_labels, fontsize=8)
    braced = _draw_ld_block_braces(lower, box_x, block_ids)
    if braced:
        lower.tick_params(axis="x", pad=18)
        lower.set_xlabel("Chromosome  (brackets mark LD blocks)")
    else:
        lower.set_xlabel("Chromosome")
    lower.set_xlim(0.2, len(boxes) + 0.8)
    n_draws = int(long["draw"].nunique()) if "draw" in long.columns else 0
    title = (
        f"{cell_type} — per-draw p-value spread of the {len(boxes):,} "
        f"{criterion.upper()} gene(s) {GENE_SET_TITLES[gene_set.lower()]} "
        f"over {n_draws} MI fit(s)"
    )
    if lower.get_legend_handles_labels()[0]:
        lower.legend(loc="upper right", fontsize=8)
    if y_break is None:
        lower.set_ylabel(r"$-\log_{10}$ p")
        lower.set_title(title)
        fig.tight_layout(rect=(0, 0.10 if braced else 0.0, 1, 1))
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
                _gene_label(row),
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


def qq_comparison(
    ours: pd.DataFrame,
    theirs: pd.DataFrame,
    cell_type: str,
    ours_label: str = "This study",
    theirs_label: str = "ctPred",
) -> Optional[plt.Figure]:
    """
    Quantile-quantile plot of the two methods' evidence, ours on the y-axis.

    Both sets of p-values are sorted independently and matched quantile against
    quantile, so this asks whether one method's *distribution* of evidence sits
    above the other's rather than whether the two agree gene by gene. Points
    above the diagonal mean we carry more signal at the same quantile; the
    companion `pvalue_scatter` answers the per-gene question.
    """
    from src.twas.compare import two_sample_quantiles

    x, y = two_sample_quantiles(
        _neg_log10(theirs["pvalue"].to_numpy(dtype=float)),
        _neg_log10(ours["pvalue"].to_numpy(dtype=float)),
    )
    if x.size == 0:
        return None

    limit = float(max(x.max(), y.max())) * 1.05 or 1.0
    fig, ax = plt.subplots(figsize=(6.5, 6.2))
    ax.plot([0, limit], [0, limit], "--", color="0.55", linewidth=1.6, label="y = x")
    ax.scatter(x, y, s=14, color="black", alpha=0.75, edgecolor="none")
    ax.set_xlim(0, limit)
    ax.set_ylim(0, limit)
    ax.set_aspect("equal")
    ax.set_xlabel(rf"{theirs_label}  $-\log_{{10}}$ p")
    ax.set_ylabel(rf"{ours_label}  $-\log_{{10}}$ p")
    ax.set_title(f"Quantile-quantile plot of TWAS $-\\log_{{10}}$p\n{cell_type}", fontsize=11)
    above = float(np.mean(y > x)) if y.size else float("nan")
    ax.text(
        0.03, 0.97,
        f"{above:.0%} of quantiles above y = x\n"
        f"{len(ours):,} vs {len(theirs):,} genes tested",
        transform=ax.transAxes,
        va="top",
        fontsize=8,
    )
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    return fig


def pvalue_scatter(
    matched: pd.DataFrame,
    cell_type: str,
    ours_label: str = "This study",
    theirs_label: str = "ctPred",
    suffixes: tuple[str, str] = ("_ours", "_ctpred"),
    label_top: int = 8,
) -> Optional[plt.Figure]:
    """
    Gene-matched evidence, coloured by which method calls the gene significant.

    Unlike the Q-Q plot this pairs genes, so it shows whether the two methods
    agree case by case. Genes far off the diagonal are the interesting ones:
    they are where the two teachers disagree about the same gene rather than
    merely about the transcriptome overall.
    """
    ours_suffix, theirs_suffix = suffixes
    x_column, y_column = f"pvalue{theirs_suffix}", f"pvalue{ours_suffix}"
    if x_column not in matched.columns or y_column not in matched.columns:
        return None
    data = matched.dropna(subset=[x_column, y_column]).reset_index(drop=True)
    if data.empty:
        return None

    x = _neg_log10(data[x_column].to_numpy(dtype=float))
    y = _neg_log10(data[y_column].to_numpy(dtype=float))
    mine = data.get(f"significant_bonferroni{ours_suffix}", pd.Series(False, index=data.index)).fillna(False).to_numpy(bool)
    yours = data.get(f"significant_bonferroni{theirs_suffix}", pd.Series(False, index=data.index)).fillna(False).to_numpy(bool)

    groups = [
        (~mine & ~yours, "0.75", "Neither", 8),
        (mine & yours, "#4f7f5f", "Both", 20),
        (mine & ~yours, "#3f6fb0", f"{ours_label} only", 20),
        (~mine & yours, "#b06a3f", f"{theirs_label} only", 20),
    ]
    limit = float(max(x.max(), y.max())) * 1.05 or 1.0

    fig, ax = plt.subplots(figsize=(6.8, 6.4))
    ax.plot([0, limit], [0, limit], "--", color="0.55", linewidth=1.4)
    for mask, color, label, size in groups:
        if mask.any():
            ax.scatter(x[mask], y[mask], s=size, color=color, alpha=0.75,
                       edgecolor="none", label=f"{label} ({int(mask.sum()):,})")
    ax.set_xlim(0, limit)
    ax.set_ylim(0, limit)
    ax.set_aspect("equal")
    ax.set_xlabel(rf"{theirs_label}  $-\log_{{10}}$ p")
    ax.set_ylabel(rf"{ours_label}  $-\log_{{10}}$ p")
    ax.set_title(f"{cell_type} — gene-matched association strength", fontsize=11)

    if label_top:
        # Label the genes the two methods disagree about most, but only among
        # those at least one of them actually calls: a large gap between two
        # null p-values is not a disagreement worth naming.
        called = np.flatnonzero(mine | yours)
        order = called[np.argsort(-np.abs(y[called] - x[called]))][:label_top]
        names = data.get(f"gene_name{ours_suffix}", data.get("gene_name", data["gene_key"]))
        for index in order:
            # Labels near the left edge would otherwise run off the axes.
            left = x[index] < limit / 2
            ax.annotate(
                str(names.iloc[index]) if names is not None else data["gene_key"].iloc[index],
                (x[index], y[index]),
                fontsize=7,
                xytext=(4 if left else -4, 3),
                textcoords="offset points",
                ha="left" if left else "right",
            )
    ax.legend(loc="lower right", fontsize=7)
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
    "manhattan_boxplots",
    "manhattan_draw_boxplots",
    "mi_draw_spread",
    "mi_draw_summary",
    "mi_stability",
    "pvalue_scatter",
    "qq",
    "qq_comparison",
    "top_genes",
    "volcano",
    "zscore_histogram",
]
