#!/usr/bin/env python3

from __future__ import annotations

import argparse
import logging
import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats


"""
evaluate_cell_type.py

Compare per-cell-type predictions against ground-truth expression for OneK1K.
Supports both a single prediction table and side-by-side comparisons of several
models.

Both input CSVs share a common layout with at least:

    gene,<individual 1>,<individual 2>,...

where `gene` is the ENSID. Extra metadata columns (e.g. chrom, tss) are allowed
and ignored for the comparison.

Evaluation pipeline:
  1. Drop training individuals from the ground truth (first N columns plus an
     explicit exclusion list).
  2. Optionally restrict to a gene subset given as a TSV with an ENSID column.
  3. Intersect ground truth and predictions on both genes and individuals, then
     mask gene/individual pairs that are missing in either matrix.
  4. Report global metrics (RMSE, MAE, Pearson, Spearman) and draw per-gene
     histograms for RMSE, MAE, Pearson, Spearman and the two correlation
     p-values.
"""


DEFAULT_METADATA_COLS = ("gene", "chrom", "tss")

# Individuals that were used for training and must never enter the evaluation.
DEFAULT_EXCLUDE_INDIVIDUALS = (
    "OneK1K_1001",
    "OneK1K_1002",
    "OneK1K_1003",
    "OneK1K_1004",
    "OneK1K_1005",
)

DEFAULT_N_TRAIN = 200


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate per-cell-type predictions against ground-truth expression "
            "with global and per-gene metrics."
        )
    )
    parser.add_argument(
        "-p",
        "--predictions",
        type=Path,
        default=None,
        help=(
            "Single-model predictions table (CSV or TSV, optionally "
            "gzip-compressed). Use this for the simple single-model workflow."
        ),
    )
    parser.add_argument(
        "-m",
        "--model",
        dest="models",
        action="append",
        nargs="+",
        metavar="MODEL_ARG",
        default=None,
        help=(
            "Model to include in a multi-model comparison, given as NAME "
            "PREDICTIONS [UNCERTAINTY]. May be passed multiple times, e.g. "
            "-m PrediXcan predixcan.csv predixcan_unc.csv -m Enformer enformer.csv."
        ),
    )
    parser.add_argument(
        "-g",
        "--ground-truth",
        type=Path,
        required=True,
        help="Ground-truth CSV. Must contain a gene column and individual columns.",
    )
    parser.add_argument(
        "-s",
        "--gene-subset",
        type=Path,
        default=None,
        help=(
            "Optional TSV with an ENSID column. When given, ground-truth genes "
            "not present in this subset are dropped."
        ),
    )
    parser.add_argument(
        "-o",
        "--output-plot",
        type=Path,
        default=None,
        help="Path to save the per-gene histogram figure (PNG). Default: do not save.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Directory for default multi-model outputs. When --model is used, "
            "this makes --pearson-plot and --pval-plot optional."
        ),
    )
    parser.add_argument(
        "--output-r2-rank-plot",
        type=Path,
        default=None,
        help=(
            "Path to save the sorted per-gene R^2 rank figure (PNG). "
            "Default: when --output-plot is set, write a companion "
            "*_r2_rank PNG next to it; otherwise do not save."
        ),
    )
    parser.add_argument(
        "--output-pred-vs-gt-plot",
        type=Path,
        default=None,
        help=(
            "Path to save per-gene prediction-vs-ground-truth scatter plots (PNG). "
            "Default: when --output-plot is set, write a companion "
            "*_pred_vs_gt PNG next to it; otherwise do not save."
        ),
    )
    parser.add_argument(
        "--pearson-plot",
        type=Path,
        default=None,
        help=(
            "Multi-model output path for the per-gene Pearson R boxplot. "
            "Default: <output-dir>/pearson_boxplot.png."
        ),
    )
    parser.add_argument(
        "--pval-plot",
        type=Path,
        default=None,
        help=(
            "Multi-model output path for the per-gene Pearson p-value boxplot. "
            "Default: <output-dir>/pearson_pval_boxplot.png."
        ),
    )
    parser.add_argument(
        "--uncertainty-plot",
        type=Path,
        default=None,
        help=(
            "Optional multi-model output path for squared-error boxplots across "
            "epistemic uncertainty quantile bins. Default with --output-dir: "
            "<output-dir>/uncertainty_error_boxplot.png when any model has uncertainty."
        ),
    )
    parser.add_argument(
        "--uncertainty-spearman-plot",
        type=Path,
        default=None,
        help=(
            "Optional multi-model output path for per-gene Spearman correlations "
            "between epistemic uncertainty and squared error. Default with "
            "--output-dir: <output-dir>/uncertainty_spearman_boxplot.png when "
            "any model has uncertainty."
        ),
    )
    parser.add_argument(
        "--top-genes-dir",
        type=Path,
        default=None,
        help=(
            "Directory for multi-model top-gene TSV outputs. Default: "
            "--output-dir if provided, otherwise the --pearson-plot directory."
        ),
    )
    parser.add_argument(
        "--top-genes-tsv",
        type=Path,
        default=None,
        help=(
            "Path to save a TSV with the top per-gene predictions ranked by "
            "Pearson correlation. Default: write a companion "
            "*_top_N_genes.tsv next to --output-plot when set, otherwise next "
            "to --predictions."
        ),
    )
    parser.add_argument(
        "--top-genes-count",
        type=int,
        default=1000,
        help="Number of top-ranked genes to write to --top-genes-tsv. Default: 1000.",
    )
    parser.add_argument(
        "--n-pred-vs-gt-genes",
        type=int,
        default=10,
        help=(
            "Number of genes to include in the per-gene prediction-vs-ground-"
            "truth plot. Half are selected from the worst per-gene R^2 values "
            "and half from the best. Default: 10."
        ),
    )
    parser.add_argument(
        "--n-ence-bins",
        type=int,
        default=15,
        help=(
            "Number of equal-count uncertainty bins for global ENCE in "
            "multi-model mode. Default: 15."
        ),
    )
    parser.add_argument(
        "--n-uncertainty-plot-bins",
        type=int,
        default=6,
        help=(
            "Number of equal-count uncertainty bins for --uncertainty-plot in "
            "multi-model mode. Default: 6."
        ),
    )
    parser.add_argument(
        "--gene-col",
        default="gene",
        help='Name of the gene/ENSID column in both CSVs. Default: "gene".',
    )
    parser.add_argument(
        "--ensid-col",
        default="ENSID",
        help='Name of the ENSID column in the gene-subset TSV. Default: "ENSID".',
    )
    parser.add_argument(
        "--metadata-cols",
        nargs="*",
        default=list(DEFAULT_METADATA_COLS),
        help=(
            "Columns that are NOT individuals and should be ignored when "
            f"intersecting. Default: {list(DEFAULT_METADATA_COLS)}."
        ),
    )
    parser.add_argument(
        "--n-train",
        type=int,
        default=DEFAULT_N_TRAIN,
        help=(
            "Number of leading ground-truth individual columns to drop as "
            f"training individuals. Default: {DEFAULT_N_TRAIN}."
        ),
    )
    parser.add_argument(
        "--exclude-individual",
        action="append",
        default=None,
        help=(
            "Additional individual to drop from the ground truth. May be passed "
            f"multiple times. Default: {list(DEFAULT_EXCLUDE_INDIVIDUALS)}."
        ),
    )
    parser.add_argument(
        "--cell-type",
        default=None,
        help="Optional cell-type label used only for plot titles.",
    )
    parser.add_argument(
        "--min-points",
        type=int,
        default=2,
        help=(
            "Minimum number of valid (non-masked) individuals a gene needs for "
            "per-gene metrics. Default: 2."
        ),
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="count",
        default=0,
        help="Increase verbosity.",
    )
    args = parser.parse_args()
    if args.predictions is None and not args.models:
        parser.error(
            "Provide either --predictions for one model or --model for one or more models."
        )
    if args.predictions is not None and args.models:
        parser.error("--predictions and --model are mutually exclusive.")
    if args.models and args.output_dir is None and (
        args.pearson_plot is None or args.pval_plot is None
    ):
        parser.error(
            "Multi-model mode requires --output-dir, or both --pearson-plot and --pval-plot."
        )
    return args


def setup_logging(verbosity: int) -> None:
    level = logging.DEBUG if verbosity >= 1 else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    # Matplotlib's font manager is extremely chatty at DEBUG level.
    logging.getLogger("matplotlib").setLevel(logging.WARNING)


def infer_separator(path: Path) -> str:
    """Pick the field separator from the file extension (gzip-aware)."""
    suffixes = [s.lower() for s in path.suffixes]
    if ".tsv" in suffixes or ".tab" in suffixes:
        return "\t"
    if ".csv" in suffixes:
        return ","
    # Fall back to whitespace/comma auto-detection for unknown extensions.
    return None


def load_expression_csv(
    path: Path,
    gene_col: str,
    metadata_cols: list[str],
) -> pd.DataFrame:
    """Read an expression table indexed by gene, keeping only individual columns.

    Supports comma- or tab-separated files, optionally gzip-compressed
    (e.g. ``.csv``, ``.tsv``, ``.tsv.gz``). Compression is inferred from the
    extension.
    """
    logging.info("Reading expression table from %s", path)
    sep = infer_separator(path)
    if sep is None:
        df = pd.read_csv(path, sep=None, engine="python", compression="infer")
    else:
        df = pd.read_csv(path, sep=sep, compression="infer")

    if gene_col not in df.columns:
        raise ValueError(
            f"Gene column {gene_col!r} not found in {path}. "
            f"Available columns: {df.columns.tolist()[:10]}..."
        )

    df[gene_col] = df[gene_col].astype(str)
    if df[gene_col].duplicated().any():
        examples = df.loc[df[gene_col].duplicated(), gene_col].unique().tolist()[:10]
        raise ValueError(
            f"Duplicate genes in {path}; cannot align unambiguously. "
            f"Examples: {examples}"
        )

    df = df.set_index(gene_col)

    drop_cols = [c for c in metadata_cols if c != gene_col and c in df.columns]
    individual_df = df.drop(columns=drop_cols)

    if individual_df.shape[1] == 0:
        raise ValueError(f"No individual columns left in {path} after dropping metadata.")

    individual_df = individual_df.apply(pd.to_numeric, errors="coerce")
    logging.info(
        "Loaded %d genes x %d individuals from %s",
        individual_df.shape[0],
        individual_df.shape[1],
        path.name,
    )
    return individual_df


def drop_training_individuals(
    ground_truth: pd.DataFrame,
    n_train: int,
    exclude_individuals: list[str],
) -> pd.DataFrame:
    """Remove leading training individuals plus an explicit exclusion list."""
    individuals = list(ground_truth.columns)

    if n_train > 0:
        if n_train >= len(individuals):
            raise ValueError(
                f"--n-train={n_train} removes all {len(individuals)} ground-truth "
                "individuals; nothing left to evaluate."
            )
        dropped_lead = individuals[:n_train]
        logging.info("Dropping first %d (training) individuals from ground truth", len(dropped_lead))
        ground_truth = ground_truth.iloc[:, n_train:]

    present_excludes = [ind for ind in exclude_individuals if ind in ground_truth.columns]
    missing_excludes = [ind for ind in exclude_individuals if ind not in ground_truth.columns]
    if missing_excludes:
        logging.warning(
            "%d excluded individuals were not in the (post-training) ground truth: %s",
            len(missing_excludes),
            missing_excludes,
        )
    if present_excludes:
        logging.info("Dropping %d explicitly excluded individuals: %s", len(present_excludes), present_excludes)
        ground_truth = ground_truth.drop(columns=present_excludes)

    if ground_truth.shape[1] == 0:
        raise ValueError("No ground-truth individuals remain after removing training individuals.")
    return ground_truth


def load_gene_subset(path: Path, ensid_col: str) -> set[str]:
    logging.info("Reading gene subset from %s", path)
    subset = pd.read_csv(path, sep="\t")
    if ensid_col not in subset.columns:
        raise ValueError(
            f"ENSID column {ensid_col!r} not found in {path}. "
            f"Available columns: {subset.columns.tolist()}"
        )
    genes = set(subset[ensid_col].dropna().astype(str).tolist())
    if not genes:
        raise ValueError(f"Gene subset {path} contains no genes.")
    logging.info("Gene subset contains %d genes", len(genes))
    return genes


def restrict_to_gene_subset(ground_truth: pd.DataFrame, subset: set[str]) -> pd.DataFrame:
    keep = ground_truth.index.isin(subset)
    n_kept = int(keep.sum())
    if n_kept == 0:
        raise ValueError("No ground-truth genes are present in the gene subset.")
    logging.info(
        "Restricting ground truth to gene subset: %d / %d genes kept",
        n_kept,
        ground_truth.shape[0],
    )
    return ground_truth.loc[keep]


def align(
    predictions: pd.DataFrame,
    ground_truth: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Intersect on genes and individuals; order both frames identically."""
    common_genes = ground_truth.index.intersection(predictions.index)
    common_individuals = ground_truth.columns.intersection(predictions.columns)

    if len(common_genes) == 0:
        raise ValueError("Predictions and ground truth share no genes.")
    if len(common_individuals) == 0:
        raise ValueError("Predictions and ground truth share no individuals.")

    logging.info(
        "Intersection: %d genes, %d individuals",
        len(common_genes),
        len(common_individuals),
    )

    gt = ground_truth.loc[common_genes, common_individuals]
    pred = predictions.loc[common_genes, common_individuals]
    return pred, gt


def global_metrics(pred: np.ndarray, gt: np.ndarray) -> dict[str, float]:
    """Metrics over every valid (non-masked) gene/individual pair."""
    mask = np.isfinite(pred) & np.isfinite(gt)
    n = int(mask.sum())
    if n < 2:
        raise ValueError("Fewer than 2 valid measurements; cannot compute global metrics.")

    p = pred[mask]
    g = gt[mask]
    diff = p - g

    rmse = float(np.sqrt(np.mean(diff ** 2)))
    mae = float(np.mean(np.abs(diff)))
    pearson = float(stats.pearsonr(p, g)[0]) if np.std(p) > 0 and np.std(g) > 0 else float("nan")
    spearman = float(stats.spearmanr(p, g)[0])

    return {
        "n_pairs": n,
        "RMSE": rmse,
        "MAE": mae,
        "Pearson": pearson,
        "Spearman": spearman,
    }


def per_gene_metrics(
    pred: pd.DataFrame,
    gt: pd.DataFrame,
    min_points: int,
) -> pd.DataFrame:
    """Compute per-gene metrics across individuals, masking missing pairs."""
    pred_values = pred.to_numpy(dtype=np.float64)
    gt_values = gt.to_numpy(dtype=np.float64)

    rows: list[dict[str, float]] = []
    for i, gene in enumerate(pred.index):
        p = pred_values[i]
        g = gt_values[i]
        mask = np.isfinite(p) & np.isfinite(g)
        n = int(mask.sum())
        if n < min_points:
            continue

        pv = p[mask]
        gv = g[mask]
        diff = pv - gv

        rmse = float(np.sqrt(np.mean(diff ** 2)))
        mae = float(np.mean(np.abs(diff)))

        if np.std(pv) > 0 and np.std(gv) > 0:
            pear_r, pear_p = stats.pearsonr(pv, gv)
            spear_r, spear_p = stats.spearmanr(pv, gv)
        else:
            pear_r = pear_p = spear_r = spear_p = float("nan")

        rows.append(
            {
                "gene": gene,
                "n": n,
                "RMSE": rmse,
                "MAE": mae,
                "Pearson": float(pear_r),
                "Spearman": float(spear_r),
                "Pearson_pval": float(pear_p),
                "Spearman_pval": float(spear_p),
            }
        )

    if not rows:
        raise ValueError(
            f"No gene had at least --min-points={min_points} valid measurements."
        )

    metrics = pd.DataFrame(rows).set_index("gene")
    logging.info("Computed per-gene metrics for %d genes", len(metrics))
    return metrics


def print_global_metrics(metrics: dict[str, float], cell_type: str | None) -> None:
    label = f" (cell_type={cell_type})" if cell_type else ""
    print(f"\n=== Global metrics{label} ===")
    print(f"  valid pairs : {metrics['n_pairs']}")
    print(f"  RMSE        : {metrics['RMSE']:.4f}")
    print(f"  MAE         : {metrics['MAE']:.4f}")
    print(f"  Pearson r   : {metrics['Pearson']:.4f}")
    print(f"  Spearman rho: {metrics['Spearman']:.4f}")


def print_per_gene_summary(metrics: pd.DataFrame) -> None:
    print(f"\n=== Per-gene metrics (n={len(metrics)} genes) ===")
    summary = metrics[
        ["RMSE", "MAE", "Pearson", "Spearman", "Pearson_pval", "Spearman_pval"]
    ].describe(percentiles=[0.25, 0.5, 0.75])
    with pd.option_context("display.float_format", lambda v: f"{v:.4f}"):
        print(summary)


def plot_per_gene_histograms(
    metrics: pd.DataFrame,
    output_path: Path,
    cell_type: str | None,
    n_bins: int = 20,
) -> None:
    # (column, title, is_pvalue). p-value panels get a [0, 1] range and a
    # uniform-null reference line, matching the reference figure.
    panels = [
        ("RMSE", "RMSE", False),
        ("MAE", "MAE", False),
        ("Pearson", "Pearson correlation", False),
        ("Spearman", "Spearman correlation", False),
        ("Pearson_pval", "Pearson correlation p-values", True),
        ("Spearman_pval", "Spearman correlation p-values", True),
    ]

    ct_label = f"cell_type={cell_type}" if cell_type else "all genes"

    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    axes = axes.ravel()

    for ax, (column, title, is_pvalue) in zip(axes, panels):
        values = metrics[column].to_numpy(dtype=np.float64)
        values = values[np.isfinite(values)]
        n = len(values)

        hist_range = (0.0, 1.0) if is_pvalue else None
        ax.hist(
            values,
            bins=n_bins,
            range=hist_range,
            color="gray",
            edgecolor="white",
        )

        if is_pvalue and n > 0:
            per_bin = n / n_bins
            ax.axhline(
                per_bin,
                color="red",
                linestyle="--",
                linewidth=1.2,
                label=f"uniform null ({per_bin:.0f}/bin)",
            )
            ax.legend(loc="upper right", fontsize=9)

        ax.set_title(f"{title}\n({ct_label}; n={n} genes)")
        ax.set_xlabel("p-values" if is_pvalue else title)
        ax.set_ylabel("Number of genes")

    fig.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    logging.info("Wrote per-gene histogram figure to %s", output_path)


def default_r2_rank_output_path(output_plot: Path) -> Path:
    suffix = output_plot.suffix or ".png"
    return output_plot.with_name(f"{output_plot.stem}_r2_rank{suffix}")


def default_pred_vs_gt_output_path(output_plot: Path) -> Path:
    suffix = output_plot.suffix or ".png"
    return output_plot.with_name(f"{output_plot.stem}_pred_vs_gt{suffix}")


def default_top_genes_output_path(
    output_plot: Path | None,
    predictions_path: Path,
    n_genes: int,
) -> Path:
    if output_plot is not None:
        return output_plot.with_name(f"{output_plot.stem}_top_{n_genes}_genes.tsv")
    return predictions_path.with_name(f"{predictions_path.stem}_top_{n_genes}_genes.tsv")


def write_ranked_gene_tsv(
    metrics: pd.DataFrame,
    output_path: Path,
    *,
    score_column: str,
    n_genes: int,
    ascending: bool = False,
) -> None:
    if n_genes <= 0:
        raise ValueError("--top-genes-count must be positive.")
    if score_column not in metrics.columns:
        raise ValueError(f"Metric column {score_column!r} not found.")

    values = metrics[score_column].to_numpy(dtype=np.float64)
    finite = np.isfinite(values)
    ranked = (
        metrics.loc[finite, [score_column]]
        .sort_values(score_column, ascending=ascending, kind="mergesort")
        .head(n_genes)
    )
    output = pd.DataFrame(
        {
            "ENSID": ranked.index.astype(str),
            "Gene": ranked[score_column].to_numpy(dtype=np.float64),
        }
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(output_path, sep="\t", index=False)
    logging.info(
        "Wrote top %d genes ranked by %s to %s",
        len(output),
        score_column,
        output_path,
    )


def plot_per_gene_r2_rank(
    metrics: pd.DataFrame,
    output_path: Path,
    cell_type: str | None,
) -> None:
    pearson = metrics["Pearson"].to_numpy(dtype=np.float64)
    r2 = pearson ** 2
    r2 = r2[np.isfinite(r2)]
    if r2.size == 0:
        logging.warning("No finite per-gene Pearson values; skipping R^2 rank figure.")
        return

    r2 = np.sort(r2)
    ranks = np.arange(r2.size)
    mean_r2 = float(np.mean(r2))
    median_r2 = float(np.median(r2))

    title = f"{cell_type} - Per-gene R$^2$" if cell_type else "Per-gene R$^2$"
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(ranks, r2, s=4, alpha=0.5)
    ax.set_title(title)
    ax.set_xlabel("Gene rank (sorted by R$^2$ ascending)")
    ax.set_ylabel("R$^2$")
    ax.set_ylim(-0.05, 1.05)
    ax.text(
        0.03,
        0.95,
        f"mean_R$^2$ = {mean_r2:.3f}\nmedian_R$^2$ = {median_r2:.3f}",
        transform=ax.transAxes,
        va="top",
    )
    fig.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    logging.info("Wrote per-gene R^2 rank figure to %s", output_path)


def select_extreme_genes_by_r2(
    metrics: pd.DataFrame,
    n_genes: int,
) -> list[tuple[str, str]]:
    if n_genes <= 0:
        raise ValueError("--n-pred-vs-gt-genes must be positive.")

    r2 = metrics["Pearson"].to_numpy(dtype=np.float64) ** 2
    finite = np.isfinite(r2)
    if not finite.any():
        logging.warning(
            "No finite per-gene Pearson values; cannot select genes by R^2."
        )
        return []

    genes = metrics.index.to_numpy()[finite]
    order = np.argsort(r2[finite])
    genes_sorted = genes[order]
    n_select = min(n_genes, genes_sorted.size)
    if n_select == 1:
        return [(str(genes_sorted[-1]), "best 1")]

    n_worst = n_select // 2
    n_best = n_select - n_worst

    worst_genes = [str(gene) for gene in genes_sorted[:n_worst]]
    worst_gene_set = set(worst_genes)
    best_genes = [
        str(gene)
        for gene in genes_sorted[::-1]
        if str(gene) not in worst_gene_set
    ][:n_best]

    selected = [
        (gene, f"worst {i}")
        for i, gene in enumerate(worst_genes, start=1)
    ]
    selected.extend(
        (gene, f"best {i}")
        for i, gene in enumerate(best_genes, start=1)
    )
    return selected


def plot_prediction_vs_ground_truth_by_gene(
    pred: pd.DataFrame,
    gt: pd.DataFrame,
    output_path: Path,
    cell_type: str | None,
    gene_metrics: pd.DataFrame,
    n_genes: int,
) -> None:
    selected_genes = select_extreme_genes_by_r2(gene_metrics, n_genes)
    if not selected_genes:
        logging.warning("No genes selected; skipping prediction-vs-ground-truth plot.")
        return

    n_cols = 5 if len(selected_genes) > 6 else min(3, len(selected_genes))
    n_rows = int(np.ceil(len(selected_genes) / n_cols))
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(3.2 * n_cols, 3.2 * n_rows),
        squeeze=False,
    )
    axes_flat = axes.ravel()

    for ax, (gene, performance_label) in zip(axes_flat, selected_genes):
        p = pred.loc[gene].to_numpy(dtype=np.float64)
        g = gt.loc[gene].to_numpy(dtype=np.float64)
        mask = np.isfinite(p) & np.isfinite(g)
        p = p[mask]
        g = g[mask]

        xy_min = float(min(np.min(g), np.min(p)))
        xy_max = float(max(np.max(g), np.max(p)))
        if xy_min == xy_max:
            xy_min -= 0.5
            xy_max += 0.5
        padding = 0.05 * (xy_max - xy_min)
        xy_min -= padding
        xy_max += padding

        ax.scatter(g, p, s=12, alpha=0.7, linewidths=0)
        ax.plot(
            [xy_min, xy_max],
            [xy_min, xy_max],
            color="red",
            linestyle="--",
            linewidth=1.0,
        )
        ax.set_xlim(xy_min, xy_max)
        ax.set_ylim(xy_min, xy_max)
        ax.set_aspect("equal", adjustable="box")

        row = gene_metrics.loc[gene]
        r2 = row["Pearson"] ** 2 if np.isfinite(row["Pearson"]) else float("nan")
        ax.set_title(
            f"{performance_label}: {gene}\nR$^2$={r2:.3f}, n={int(row['n'])}",
            fontsize=9,
        )
        ax.tick_params(labelsize=8)

    for ax in axes_flat[len(selected_genes):]:
        ax.axis("off")

    for ax in axes[-1, :]:
        if ax.has_data():
            ax.set_xlabel("Ground truth expression")
    for ax in axes[:, 0]:
        if ax.has_data():
            ax.set_ylabel("Predicted expression")

    ct_label = f" ({cell_type})" if cell_type else ""
    fig.suptitle(
        f"Per-gene predictions vs ground truth{ct_label}\n"
        "worst- and best-performing genes by per-gene R$^2$",
        fontsize=12,
    )
    fig.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    logging.info(
        "Wrote per-gene prediction-vs-ground-truth figure to %s",
        output_path,
    )


def validate_models(models: list[list[str]]) -> list[tuple[str, Path, Path | None]]:
    """Normalize --model arguments to (name, predictions, uncertainty)."""
    parsed_models: list[tuple[str, Path, Path | None]] = []
    for model_args in models:
        if len(model_args) not in (2, 3):
            raise ValueError(
                "--model expects NAME PREDICTIONS [UNCERTAINTY], got "
                f"{len(model_args)} values: {model_args}"
            )
        name = model_args[0]
        predictions_path = Path(model_args[1])
        uncertainty_path = Path(model_args[2]) if len(model_args) == 3 else None
        parsed_models.append((name, predictions_path, uncertainty_path))
    return parsed_models


def spearman_rank_correlation(a: np.ndarray, b: np.ndarray) -> float:
    """Spearman rank correlation with average ranks for ties."""
    mask = np.isfinite(a) & np.isfinite(b)
    if int(mask.sum()) < 2:
        return float("nan")

    a_rank = pd.Series(a[mask]).rank(method="average").to_numpy(dtype=np.float64)
    b_rank = pd.Series(b[mask]).rank(method="average").to_numpy(dtype=np.float64)
    a_rank = a_rank - np.mean(a_rank)
    b_rank = b_rank - np.mean(b_rank)
    denom = np.linalg.norm(a_rank) * np.linalg.norm(b_rank)
    if denom <= 0:
        return float("nan")
    return float(np.dot(a_rank, b_rank) / denom)


def expected_normalized_calibration_error(
    uncertainty: np.ndarray,
    squared_error: np.ndarray,
    n_bins: int,
) -> float:
    """Compute ENCE over equal-count bins sorted by predicted uncertainty."""
    if n_bins <= 0:
        raise ValueError("--n-ence-bins must be positive.")

    mask = np.isfinite(uncertainty) & np.isfinite(squared_error)
    uncertainty = uncertainty[mask]
    squared_error = squared_error[mask]
    n = uncertainty.size
    if n < n_bins:
        return float("nan")

    order = np.argsort(uncertainty)
    uncertainty_sorted = uncertainty[order]
    squared_error_sorted = squared_error[order]

    ence = 0.0
    valid_bins = 0
    for idx in np.array_split(np.arange(n), n_bins):
        if idx.size == 0:
            continue
        rmv = np.sqrt(max(float(np.mean(uncertainty_sorted[idx])), 1e-12))
        rmse = np.sqrt(max(float(np.mean(squared_error_sorted[idx])), 0.0))
        ence += abs(rmv - rmse) / max(rmv, 1e-12)
        valid_bins += 1
    return ence / max(valid_bins, 1)


def uncertainty_calibration_metrics(
    pred_values: np.ndarray,
    gt_values: np.ndarray,
    uncertainty_values: np.ndarray,
    n_ence_bins: int,
) -> dict[str, float]:
    """Compute global per-element uncertainty calibration metrics."""
    squared_error = (pred_values - gt_values) ** 2
    mask = (
        np.isfinite(pred_values)
        & np.isfinite(gt_values)
        & np.isfinite(uncertainty_values)
    )
    uncertainty = uncertainty_values[mask]
    squared_error = squared_error[mask]

    return {
        "n_pairs": float(uncertainty.size),
        "mean_uncertainty": (
            float(np.mean(uncertainty)) if uncertainty.size else float("nan")
        ),
        "mean_squared_error": (
            float(np.mean(squared_error)) if squared_error.size else float("nan")
        ),
        "rmse": (
            float(np.sqrt(np.mean(squared_error)))
            if squared_error.size
            else float("nan")
        ),
        "ence": expected_normalized_calibration_error(
            uncertainty, squared_error, n_ence_bins
        ),
        "uncertainty_error_spearman": spearman_rank_correlation(
            uncertainty, squared_error
        ),
    }


def boxplot_stats(values: np.ndarray) -> dict[str, object]:
    """Return matplotlib.bxp-compatible stats without storing fliers."""
    q1, median, q3 = np.percentile(values, [25.0, 50.0, 75.0])
    iqr = q3 - q1
    low = q1 - 1.5 * iqr
    high = q3 + 1.5 * iqr
    within = values[(values >= low) & (values <= high)]
    if within.size == 0:
        whislo = float(np.min(values))
        whishi = float(np.max(values))
    else:
        whislo = float(np.min(within))
        whishi = float(np.max(within))

    return {
        "med": float(median),
        "q1": float(q1),
        "q3": float(q3),
        "whislo": whislo,
        "whishi": whishi,
        "fliers": [],
    }


def uncertainty_error_boxplot_stats(
    pred_values: np.ndarray,
    gt_values: np.ndarray,
    uncertainty_values: np.ndarray,
    n_bins: int,
) -> list[dict[str, object]]:
    """Summarize squared error in equal-count epistemic-uncertainty bins."""
    if n_bins <= 0:
        raise ValueError("--n-uncertainty-plot-bins must be positive.")

    squared_error = (pred_values - gt_values) ** 2
    mask = (
        np.isfinite(pred_values)
        & np.isfinite(gt_values)
        & np.isfinite(uncertainty_values)
    )
    uncertainty = uncertainty_values[mask]
    squared_error = squared_error[mask]
    if uncertainty.size < n_bins:
        return []

    order = np.argsort(uncertainty)
    squared_error_sorted = squared_error[order]

    stats = []
    for i, idx in enumerate(np.array_split(np.arange(uncertainty.size), n_bins), start=1):
        if idx.size == 0:
            continue
        bin_stats = boxplot_stats(squared_error_sorted[idx])
        bin_stats["label"] = str(i)
        stats.append(bin_stats)
    return stats


def per_gene_uncertainty_error_spearman(
    pred_values: np.ndarray,
    gt_values: np.ndarray,
    uncertainty_values: np.ndarray,
    genes: pd.Index,
) -> dict[str, float]:
    """Rank association between epistemic uncertainty and squared error per gene."""
    squared_error = (pred_values - gt_values) ** 2
    values: dict[str, float] = {}
    for i, gene in enumerate(genes):
        values[str(gene)] = spearman_rank_correlation(
            uncertainty_values[i],
            squared_error[i],
        )
    return values


def per_gene_spearman_summary(values: pd.Series) -> dict[str, float]:
    """Summarize per-gene uncertainty-error Spearman values."""
    finite = values.to_numpy(dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    return {
        "per_gene_spearman_mean": (
            float(np.mean(finite)) if finite.size else float("nan")
        ),
        "per_gene_spearman_median": (
            float(np.median(finite)) if finite.size else float("nan")
        ),
        "frac_genes_spearman_gt_0": (
            float(np.mean(finite > 0.0)) if finite.size else float("nan")
        ),
    }


def add_mean_uncertainty(
    metrics: pd.DataFrame,
    pred: pd.DataFrame,
    gt: pd.DataFrame,
    uncertainty: pd.DataFrame,
    *,
    n_ence_bins: int,
    n_uncertainty_plot_bins: int,
) -> pd.DataFrame:
    """Attach uncertainty diagnostics to one model's per-gene metrics."""
    common_genes = pred.index.intersection(uncertainty.index)
    common_individuals = pred.columns.intersection(uncertainty.columns)

    if len(common_genes) == 0:
        raise ValueError("Predictions and uncertainties share no genes.")
    if len(common_individuals) == 0:
        raise ValueError("Predictions and uncertainties share no individuals.")

    missing_gene_count = len(pred.index.difference(uncertainty.index))
    missing_individual_count = len(pred.columns.difference(uncertainty.columns))
    if missing_gene_count or missing_individual_count:
        logging.warning(
            "Uncertainty table is missing %d evaluated genes and %d evaluated individuals; "
            "their points will use the fallback color.",
            missing_gene_count,
            missing_individual_count,
        )

    pred_values = pred.loc[common_genes, common_individuals].to_numpy(dtype=np.float64)
    gt_values = gt.loc[common_genes, common_individuals].to_numpy(dtype=np.float64)
    uncertainty_values = uncertainty.loc[common_genes, common_individuals].to_numpy(
        dtype=np.float64
    )

    mean_uncertainty: dict[str, float] = {}
    gene_uncertainty_error_spearman = per_gene_uncertainty_error_spearman(
        pred_values,
        gt_values,
        uncertainty_values,
        common_genes,
    )
    for i, gene in enumerate(common_genes):
        mask = (
            np.isfinite(pred_values[i])
            & np.isfinite(gt_values[i])
            & np.isfinite(uncertainty_values[i])
        )
        mean_uncertainty[str(gene)] = (
            float(np.mean(uncertainty_values[i][mask]))
            if int(mask.sum()) > 0
            else float("nan")
        )

    metrics = metrics.copy()
    metrics["uncertainty"] = metrics.index.map(mean_uncertainty).astype(float)
    metrics["uncertainty_error_spearman"] = (
        metrics.index.map(gene_uncertainty_error_spearman).astype(float)
    )
    metrics.attrs["uncertainty_calibration"] = uncertainty_calibration_metrics(
        pred_values,
        gt_values,
        uncertainty_values,
        n_ence_bins=n_ence_bins,
    )
    metrics.attrs["uncertainty_calibration"].update(
        per_gene_spearman_summary(metrics["uncertainty_error_spearman"])
    )
    metrics.attrs["uncertainty_error_boxplot_stats"] = uncertainty_error_boxplot_stats(
        pred_values,
        gt_values,
        uncertainty_values,
        n_bins=n_uncertainty_plot_bins,
    )
    return metrics


def restrict_to_common_individuals(
    ground_truth: pd.DataFrame,
    loaded_models: list[tuple[str, pd.DataFrame, Path | None]],
) -> pd.DataFrame:
    """Restrict ground truth to individuals shared by GT and every model."""
    common = ground_truth.columns
    for _name, predictions, _unc in loaded_models:
        common = common.intersection(predictions.columns)

    if len(common) == 0:
        raise ValueError(
            "Ground truth and the models share no evaluation individuals "
            "after dropping training individuals."
        )

    common_ordered = ground_truth.columns[ground_truth.columns.isin(common)]
    logging.info(
        "Harmonized evaluation set: %d individuals shared across ground truth "
        "and all %d models",
        len(common_ordered),
        len(loaded_models),
    )

    common_set = set(common_ordered)
    gt_dropped = len(ground_truth.columns) - len(common_ordered)
    if gt_dropped:
        logging.info(
            "Dropping %d ground-truth individuals not predicted by every model.",
            gt_dropped,
        )
    for name, predictions, _unc in loaded_models:
        extra = [c for c in predictions.columns if c not in common_set]
        if extra:
            logging.warning(
                "Model %r predicts %d individuals outside the shared evaluation "
                "set; they will be ignored.",
                name,
                len(extra),
            )

    return ground_truth.loc[:, common_ordered]


def evaluate_model(
    name: str,
    predictions: pd.DataFrame,
    uncertainty_path: Path | None,
    ground_truth: pd.DataFrame,
    *,
    gene_col: str,
    metadata_cols: list[str],
    min_points: int,
    n_ence_bins: int,
    n_uncertainty_plot_bins: int,
) -> pd.DataFrame:
    """Score one model's predictions against the prepared ground truth."""
    logging.info("Evaluating model %r", name)
    pred, gt = align(predictions, ground_truth)
    metrics = per_gene_metrics(pred, gt, min_points=min_points)
    if uncertainty_path is not None:
        logging.info("Loading uncertainty estimates for model %r from %s", name, uncertainty_path)
        uncertainty = load_expression_csv(
            uncertainty_path, gene_col=gene_col, metadata_cols=metadata_cols
        )
        metrics = add_mean_uncertainty(
            metrics,
            pred,
            gt,
            uncertainty,
            n_ence_bins=n_ence_bins,
            n_uncertainty_plot_bins=n_uncertainty_plot_bins,
        )
    logging.info("Model %r: per-gene metrics for %d genes", name, len(metrics))
    return metrics


def collect_values_with_uncertainty(
    model_metrics: list[tuple[str, pd.DataFrame]],
    column: str,
    point_value_column: str,
) -> tuple[list[str], list[np.ndarray], list[np.ndarray | None]]:
    """Pull finite metric values and aligned point-color values for every model."""
    names: list[str] = []
    values: list[np.ndarray] = []
    point_values: list[np.ndarray | None] = []
    for name, metrics in model_metrics:
        col = metrics[column].to_numpy(dtype=np.float64)
        finite = np.isfinite(col)
        names.append(name)
        values.append(col[finite])

        if point_value_column in metrics.columns:
            point_values.append(
                metrics[point_value_column].to_numpy(dtype=np.float64)[finite]
            )
        else:
            point_values.append(None)
    return names, values, point_values


def uncertainty_quantiles(
    uncertainty: np.ndarray,
    sorted_reference: np.ndarray,
) -> np.ma.MaskedArray:
    """Convert uncertainty values to empirical quantiles against all finite values."""
    quantiles = np.full(uncertainty.shape, np.nan, dtype=np.float64)
    finite = np.isfinite(uncertainty)
    if not finite.any():
        return np.ma.masked_invalid(quantiles)

    if sorted_reference.size == 1:
        quantiles[finite] = 0.5
        return np.ma.masked_invalid(quantiles)

    values = uncertainty[finite]
    left = np.searchsorted(sorted_reference, values, side="left")
    right = np.searchsorted(sorted_reference, values, side="right") - 1
    quantiles[finite] = (left + right) / (2 * (sorted_reference.size - 1))
    return np.ma.masked_invalid(quantiles)


def model_xtick_label(name: str, metrics: pd.DataFrame, n_values: int) -> str:
    """Build a compact x-axis label, including uncertainty ranking if available."""
    label = f"{name}\n(n={n_values})"
    calibration = metrics.attrs.get("uncertainty_calibration")
    if calibration is None:
        return label

    spearman = calibration.get("uncertainty_error_spearman", float("nan"))
    if np.isfinite(spearman):
        label += f"\nunc-err Spearman={spearman:.2f}"
    return label


def plot_model_boxplots(
    model_metrics: list[tuple[str, pd.DataFrame]],
    column: str,
    output_path: Path,
    *,
    ylabel: str,
    title: str,
    cell_type: str | None,
    ylim: tuple[float, float] | None = None,
    point_value_column: str = "uncertainty",
    point_value_label: str = "Mean prediction uncertainty quantile",
    point_value_cmap_name: str = "RdYlGn_r",
    point_value_norm: plt.Normalize | None = None,
    jitter_seed: int = 0,
) -> None:
    """Draw one boxplot per model with jittered per-gene points overlaid."""
    names, values, point_values = collect_values_with_uncertainty(
        model_metrics,
        column,
        point_value_column,
    )

    cmap = plt.get_cmap("tab10")
    colors = [cmap(i % cmap.N) for i in range(len(names))]
    point_value_arrays = [u[np.isfinite(u)] for u in point_values if u is not None]
    finite_point_values = (
        np.concatenate(point_value_arrays) if point_value_arrays else np.array([])
    )
    sorted_point_values = None
    point_value_cmap = plt.get_cmap(point_value_cmap_name).copy()
    point_value_cmap.set_bad("black", alpha=0.15)
    if finite_point_values.size:
        sorted_point_values = np.sort(finite_point_values)
    if point_value_norm is None:
        point_value_norm = plt.Normalize(vmin=0.0, vmax=1.0)

    positions = np.arange(1, len(names) + 1)
    rng = np.random.default_rng(jitter_seed)

    width = max(8.0, 1.6 * len(names))
    fig, ax = plt.subplots(figsize=(width, 6))

    bp = ax.boxplot(
        values,
        positions=positions,
        widths=0.6,
        patch_artist=True,
        showfliers=False,
        medianprops={"color": "black", "linewidth": 1.5},
        whiskerprops={"color": "black"},
        capprops={"color": "black"},
        boxprops={"edgecolor": "black"},
    )
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.85)

    scatter_for_colorbar = None
    for pos, vals, model_point_values in zip(positions, values, point_values):
        if vals.size == 0:
            continue
        scatter_kwargs = {
            "s": 6,
            "alpha": 0.35 if model_point_values is not None else 0.15,
            "linewidths": 0,
            "zorder": 3,
        }
        if model_point_values is not None and sorted_point_values is not None:
            quantiles = uncertainty_quantiles(model_point_values, sorted_point_values)
            quantile_values = quantiles.filled(0.5)
            jitter = rng.uniform(-0.025, 0.025, size=vals.size)
            x_values = pos + ((quantile_values - 0.5) * 0.46) + jitter
            scatter_kwargs.update(
                {
                    "c": (
                        np.ma.masked_invalid(model_point_values)
                        if point_value_column == "uncertainty_error_spearman"
                        else quantiles
                    ),
                    "cmap": point_value_cmap,
                    "norm": point_value_norm,
                }
            )
        else:
            x_values = np.full(vals.size, pos) + rng.uniform(-0.18, 0.18, size=vals.size)
            scatter_kwargs["color"] = "black"

        scatter = ax.scatter(x_values, vals, **scatter_kwargs)
        if model_point_values is not None and sorted_point_values is not None:
            scatter_for_colorbar = scatter

    ax.set_xticks(positions)
    ax.set_xticklabels(
        [
            model_xtick_label(name, metrics, vals.size)
            for (name, metrics), vals in zip(model_metrics, values)
        ]
    )
    ax.set_ylabel(ylabel)
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.axhline(0.0, color="black", linewidth=0.8)

    ct_label = f" (cell_type={cell_type})" if cell_type else ""
    ax.set_title(f"{title}{ct_label}")

    if scatter_for_colorbar is not None:
        colorbar = fig.colorbar(scatter_for_colorbar, ax=ax, pad=0.02)
        colorbar.set_label(point_value_label)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    logging.info("Wrote %s boxplot figure to %s", column, output_path)


def plot_uncertainty_error_boxplots(
    model_metrics: list[tuple[str, pd.DataFrame]],
    output_path: Path,
    *,
    cell_type: str | None,
) -> None:
    """Plot squared error across epistemic-uncertainty quantile bins."""
    models_with_stats = [
        (name, metrics)
        for name, metrics in model_metrics
        if metrics.attrs.get("uncertainty_error_boxplot_stats")
    ]
    if not models_with_stats:
        logging.info("No uncertainty boxplot stats available; skipping %s", output_path)
        return

    n_models = len(models_with_stats)
    fig, axes = plt.subplots(
        1,
        n_models,
        figsize=(max(5.0, 4.5 * n_models), 4.5),
        squeeze=False,
        sharey=True,
    )
    axes = axes.ravel()

    for ax, (name, metrics) in zip(axes, models_with_stats):
        stats = metrics.attrs["uncertainty_error_boxplot_stats"]
        ax.bxp(
            stats,
            showfliers=False,
            patch_artist=True,
            boxprops={"facecolor": "#b9d9e0", "edgecolor": "#5a7d8c"},
            medianprops={"color": "#5a7d8c", "linewidth": 1.5},
            whiskerprops={"color": "#5a7d8c"},
            capprops={"color": "#5a7d8c"},
        )

        calibration = metrics.attrs.get("uncertainty_calibration", {})
        spearman = calibration.get("uncertainty_error_spearman", float("nan"))
        ence = calibration.get("ence", float("nan"))
        subtitle_parts = []
        if np.isfinite(spearman):
            subtitle_parts.append(f"Spearman={spearman:.2f}")
        if np.isfinite(ence):
            subtitle_parts.append(f"ENCE={ence:.2f}")
        subtitle = f"\n{', '.join(subtitle_parts)}" if subtitle_parts else ""

        ax.set_title(f"{name}{subtitle}")
        ax.set_xlabel("Epistemic uncertainty quantile bin\n(low to high)")
        ax.grid(axis="y", alpha=0.25)

    axes[0].set_ylabel("Squared error")
    ct_label = f" (cell_type={cell_type})" if cell_type else ""
    fig.suptitle(f"Squared error by epistemic uncertainty quantile{ct_label}")
    fig.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    logging.info("Wrote uncertainty-error boxplot figure to %s", output_path)


def plot_per_gene_uncertainty_spearman(
    model_metrics: list[tuple[str, pd.DataFrame]],
    output_path: Path,
    *,
    cell_type: str | None,
    jitter_seed: int = 0,
) -> None:
    """Plot per-gene uncertainty-error Spearman distributions per model."""
    rows = []
    for name, metrics in model_metrics:
        if "uncertainty_error_spearman" not in metrics.columns:
            continue
        values = metrics["uncertainty_error_spearman"].to_numpy(dtype=np.float64)
        values = values[np.isfinite(values)]
        if values.size:
            rows.append((name, values))

    if not rows:
        logging.info("No per-gene uncertainty Spearman values available; skipping %s", output_path)
        return

    names = [name for name, _ in rows]
    values = [vals for _, vals in rows]
    positions = np.arange(1, len(names) + 1)
    rng = np.random.default_rng(jitter_seed)

    width = max(8.0, 1.6 * len(names))
    fig, ax = plt.subplots(figsize=(width, 6))

    bp = ax.boxplot(
        values,
        positions=positions,
        widths=0.55,
        patch_artist=True,
        showfliers=False,
        medianprops={"color": "black", "linewidth": 1.5},
        whiskerprops={"color": "black"},
        capprops={"color": "black"},
        boxprops={"edgecolor": "black"},
    )
    cmap = plt.get_cmap("tab10")
    for i, patch in enumerate(bp["boxes"]):
        patch.set_facecolor(cmap(i % cmap.N))
        patch.set_alpha(0.75)

    point_cmap = plt.get_cmap("RdYlGn")
    norm = plt.Normalize(vmin=-1.0, vmax=1.0)
    scatter_for_colorbar = None
    for pos, vals in zip(positions, values):
        jitter = rng.uniform(-0.16, 0.16, size=vals.size)
        scatter_for_colorbar = ax.scatter(
            np.full(vals.size, pos) + jitter,
            vals,
            c=vals,
            cmap=point_cmap,
            norm=norm,
            s=7,
            alpha=0.3,
            linewidths=0,
            zorder=3,
        )

    ax.axhline(0.0, color="black", linestyle="--", linewidth=1.0)
    ax.set_ylim(-1.0, 1.0)
    ax.set_xticks(positions)
    ax.set_xticklabels(
        [f"{name}\n(n={vals.size})" for name, vals in zip(names, values)]
    )
    ax.set_ylabel("Per-gene uncertainty-error Spearman")
    ct_label = f" (cell_type={cell_type})" if cell_type else ""
    ax.set_title(f"Within-gene uncertainty ranking{ct_label}")
    ax.grid(axis="y", alpha=0.25)

    if scatter_for_colorbar is not None:
        colorbar = fig.colorbar(scatter_for_colorbar, ax=ax, pad=0.02)
        colorbar.set_label("Per-gene Spearman")

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    logging.info("Wrote per-gene uncertainty Spearman figure to %s", output_path)


def print_model_summary(model_metrics: list[tuple[str, pd.DataFrame]]) -> None:
    print("\n=== Per-gene Pearson summary (one row per model) ===")
    rows = []
    for name, metrics in model_metrics:
        pearson = metrics["Pearson"].to_numpy(dtype=np.float64)
        pearson = pearson[np.isfinite(pearson)]
        pval = metrics["Pearson_pval"].to_numpy(dtype=np.float64)
        pval = pval[np.isfinite(pval)]
        rows.append(
            {
                "model": name,
                "n_genes": len(pearson),
                "Pearson_mean": float(np.mean(pearson)) if pearson.size else float("nan"),
                "Pearson_median": float(np.median(pearson)) if pearson.size else float("nan"),
                "Pval_median": float(np.median(pval)) if pval.size else float("nan"),
                "frac_p<0.05": float(np.mean(pval < 0.05)) if pval.size else float("nan"),
            }
        )
    summary = pd.DataFrame(rows).set_index("model")
    with pd.option_context("display.float_format", lambda v: f"{v:.4f}"):
        print(summary)


def print_uncertainty_calibration_summary(
    model_metrics: list[tuple[str, pd.DataFrame]]
) -> None:
    rows = []
    for name, metrics in model_metrics:
        calibration = metrics.attrs.get("uncertainty_calibration")
        if calibration is None:
            continue
        rows.append({"model": name, **calibration})

    if not rows:
        return

    print("\n=== Global uncertainty calibration summary (one row per model) ===")
    summary = pd.DataFrame(rows).set_index("model")
    if "n_pairs" in summary.columns:
        summary["n_pairs"] = summary["n_pairs"].astype("Int64")
    with pd.option_context("display.float_format", lambda v: f"{v:.4f}"):
        print(summary)


def safe_filename_component(value: str) -> str:
    component = re.sub(r"[^A-Za-z0-9._-]+", "_", value.strip())
    component = component.strip("._-")
    return component or "model"


def write_uncertainty_top_gene_tsvs(
    model_metrics: list[tuple[str, pd.DataFrame]],
    output_dir: Path,
    *,
    output_prefix: str,
    n_genes: int,
) -> None:
    if n_genes <= 0:
        raise ValueError("--top-genes-count must be positive.")

    safe_prefix = safe_filename_component(output_prefix)
    n_written_models = 0
    for name, metrics in model_metrics:
        if (
            "uncertainty_error_spearman" not in metrics.columns
            or "uncertainty" not in metrics.columns
        ):
            continue

        safe_name = safe_filename_component(name)
        spearman_output = (
            output_dir
            / f"{safe_prefix}_{safe_name}_top_{n_genes}_uncertainty_spearman.tsv"
        )
        lowest_uncertainty_output = (
            output_dir
            / f"{safe_prefix}_{safe_name}_lowest_{n_genes}_mean_uncertainty.tsv"
        )
        write_ranked_gene_tsv(
            metrics,
            spearman_output,
            score_column="uncertainty_error_spearman",
            n_genes=n_genes,
            ascending=False,
        )
        write_ranked_gene_tsv(
            metrics,
            lowest_uncertainty_output,
            score_column="uncertainty",
            n_genes=n_genes,
            ascending=True,
        )
        n_written_models += 1

    if n_written_models == 0:
        logging.info("No models with uncertainty metrics; skipping uncertainty top-gene TSVs.")


def resolve_multi_output_paths(args: argparse.Namespace) -> dict[str, Path | None]:
    output_dir = args.output_dir
    pearson_plot = args.pearson_plot or output_dir / "pearson_boxplot.png"
    pval_plot = args.pval_plot or output_dir / "pearson_pval_boxplot.png"
    uncertainty_plot = args.uncertainty_plot
    uncertainty_spearman_plot = args.uncertainty_spearman_plot
    if output_dir is not None:
        uncertainty_plot = uncertainty_plot or output_dir / "uncertainty_error_boxplot.png"
        uncertainty_spearman_plot = (
            uncertainty_spearman_plot
            or output_dir / "uncertainty_spearman_boxplot.png"
        )

    top_genes_dir = args.top_genes_dir or output_dir or pearson_plot.parent
    return {
        "pearson_plot": pearson_plot,
        "pval_plot": pval_plot,
        "uncertainty_plot": uncertainty_plot,
        "uncertainty_spearman_plot": uncertainty_spearman_plot,
        "top_genes_dir": top_genes_dir,
    }


def run_single_model(args: argparse.Namespace, exclude_individuals: list[str]) -> None:
    predictions = load_expression_csv(
        args.predictions, gene_col=args.gene_col, metadata_cols=args.metadata_cols
    )
    ground_truth = load_expression_csv(
        args.ground_truth, gene_col=args.gene_col, metadata_cols=args.metadata_cols
    )

    ground_truth = drop_training_individuals(
        ground_truth,
        n_train=args.n_train,
        exclude_individuals=exclude_individuals,
    )

    if args.gene_subset is not None:
        subset = load_gene_subset(args.gene_subset, ensid_col=args.ensid_col)
        ground_truth = restrict_to_gene_subset(ground_truth, subset)

    pred, gt = align(predictions, ground_truth)

    g_metrics = global_metrics(pred.to_numpy(dtype=np.float64), gt.to_numpy(dtype=np.float64))
    print_global_metrics(g_metrics, args.cell_type)

    gene_metrics = per_gene_metrics(pred, gt, min_points=args.min_points)
    print_per_gene_summary(gene_metrics)

    top_genes_output = (
        args.top_genes_tsv
        if args.top_genes_tsv is not None
        else default_top_genes_output_path(
            args.output_plot,
            args.predictions,
            args.top_genes_count,
        )
    )
    write_ranked_gene_tsv(
        gene_metrics,
        top_genes_output,
        score_column="Pearson",
        n_genes=args.top_genes_count,
        ascending=False,
    )

    wrote_figure = False
    if args.output_plot is not None:
        plot_per_gene_histograms(gene_metrics, args.output_plot, args.cell_type)
        wrote_figure = True

    if args.output_plot is not None or args.output_r2_rank_plot is not None:
        r2_rank_output = (
            args.output_r2_rank_plot
            if args.output_r2_rank_plot is not None
            else default_r2_rank_output_path(args.output_plot)
        )
        plot_per_gene_r2_rank(gene_metrics, r2_rank_output, args.cell_type)
        wrote_figure = True

    if args.output_plot is not None or args.output_pred_vs_gt_plot is not None:
        pred_vs_gt_output = (
            args.output_pred_vs_gt_plot
            if args.output_pred_vs_gt_plot is not None
            else default_pred_vs_gt_output_path(args.output_plot)
        )
        plot_prediction_vs_ground_truth_by_gene(
            pred,
            gt,
            pred_vs_gt_output,
            args.cell_type,
            gene_metrics,
            args.n_pred_vs_gt_genes,
        )
        wrote_figure = True

    if not wrote_figure:
        logging.info(
            "No plot output path given; skipping figures "
            "(per-gene summary printed above)."
        )


def run_multi_model(args: argparse.Namespace, exclude_individuals: list[str]) -> None:
    outputs = resolve_multi_output_paths(args)
    parsed_models = validate_models(args.models)
    has_uncertainty = any(uncertainty_path is not None for _, _, uncertainty_path in parsed_models)

    ground_truth = load_expression_csv(
        args.ground_truth, gene_col=args.gene_col, metadata_cols=args.metadata_cols
    )
    ground_truth = drop_training_individuals(
        ground_truth,
        n_train=args.n_train,
        exclude_individuals=exclude_individuals,
    )
    if args.gene_subset is not None:
        subset = load_gene_subset(args.gene_subset, ensid_col=args.ensid_col)
        ground_truth = restrict_to_gene_subset(ground_truth, subset)

    loaded_models: list[tuple[str, pd.DataFrame, Path | None]] = []
    for name, predictions_path, uncertainty_path in parsed_models:
        logging.info("Loading predictions for model %r from %s", name, predictions_path)
        predictions = load_expression_csv(
            predictions_path, gene_col=args.gene_col, metadata_cols=args.metadata_cols
        )
        loaded_models.append((name, predictions, uncertainty_path))

    ground_truth = restrict_to_common_individuals(ground_truth, loaded_models)

    model_metrics: list[tuple[str, pd.DataFrame]] = []
    for name, predictions, uncertainty_path in loaded_models:
        metrics = evaluate_model(
            name,
            predictions,
            uncertainty_path,
            ground_truth,
            gene_col=args.gene_col,
            metadata_cols=args.metadata_cols,
            min_points=args.min_points,
            n_ence_bins=args.n_ence_bins,
            n_uncertainty_plot_bins=args.n_uncertainty_plot_bins,
        )
        model_metrics.append((name, metrics))

    print_model_summary(model_metrics)
    print_uncertainty_calibration_summary(model_metrics)

    write_uncertainty_top_gene_tsvs(
        model_metrics,
        outputs["top_genes_dir"],
        output_prefix=outputs["pearson_plot"].stem,
        n_genes=args.top_genes_count,
    )

    plot_model_boxplots(
        model_metrics,
        column="Pearson",
        output_path=outputs["pearson_plot"],
        ylabel="Pearson R",
        title="Per-gene Pearson correlation",
        cell_type=args.cell_type,
        ylim=(-1.0, 1.0),
        point_value_column="uncertainty_error_spearman",
        point_value_label="Per-gene uncertainty-error Spearman",
        point_value_cmap_name="RdYlGn",
        point_value_norm=plt.Normalize(vmin=-1.0, vmax=1.0),
    )
    plot_model_boxplots(
        model_metrics,
        column="Pearson_pval",
        output_path=outputs["pval_plot"],
        ylabel="Pearson correlation p-value",
        title="Per-gene Pearson correlation p-values",
        cell_type=args.cell_type,
        ylim=(0.0, 1.0),
    )
    if has_uncertainty and outputs["uncertainty_plot"] is not None:
        plot_uncertainty_error_boxplots(
            model_metrics,
            outputs["uncertainty_plot"],
            cell_type=args.cell_type,
        )
    if has_uncertainty and outputs["uncertainty_spearman_plot"] is not None:
        plot_per_gene_uncertainty_spearman(
            model_metrics,
            outputs["uncertainty_spearman_plot"],
            cell_type=args.cell_type,
        )


def main() -> None:
    args = parse_args()
    setup_logging(args.verbose)
    logging.debug("Arguments: %s", args)

    exclude_individuals = (
        args.exclude_individual
        if args.exclude_individual is not None
        else list(DEFAULT_EXCLUDE_INDIVIDUALS)
    )

    try:
        if args.models:
            run_multi_model(args, exclude_individuals)
        else:
            run_single_model(args, exclude_individuals)

    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)

    logging.info("Done.")


if __name__ == "__main__":
    main()
