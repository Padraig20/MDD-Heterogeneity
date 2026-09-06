#!/usr/bin/env python3

from __future__ import annotations

import argparse
import logging
import math
import re
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import brentq


"""
Evaluate OneK1K expression predictions from several models and random seeds.

Ground truth is a directory containing one wide CSV per cell type:

    gene,chrom,tss,<individual 1>,...,<individual N>

The predictions directory contains one directory per model and, below each
model, one directory per seed. Deterministic seed directories contain the
cell-type CSVs directly. Probabilistic seed directories have this layout:

    <model>/<seed>/preds/*.csv
    <model>/<seed>/totvar/*.csv
    <model>/<seed>/aleatoric/*.csv
    <model>/<seed>/epistemic/*.csv

Only the individuals in INDIVIDUAL_IDS are evaluated. Individual headers are
canonicalized by their trailing numeric ID, so e.g. "OneK1K_1001" and "1001"
match. Duplicate-suffixed labels such as "OneK1K_847_2" are excluded rather
than mapped to their final numeric token. Genes are aligned by
(gene, chrom, tss). Ground-truth donor availability is resolved separately for
every cell type; prediction metrics use the cell-type-specific intersection
with the requested test IDs.

Outputs:

    per_model/<model>/pearson_pvalues.png
    comparisons/significant_genes.png
    comparisons/m1.png
    comparisons/survival_curve.png
    comparisons/survival_curve_by_cell_type.png
    comparisons/survival_curve_auc.png
    comparisons/survival_curve_auc_by_cell_type.png
    comparisons/spearman_survival_curve.png
    comparisons/spearman_survival_curve_by_cell_type.png
    comparisons/spearman_survival_curve_auc.png
    comparisons/spearman_survival_curve_auc_by_cell_type.png
    comparisons/survival_curve_auc_uncertainty_ablation_by_cell_type.png
    comparisons/uncertainty_ablation/<cell_type>.png
    comparisons/sign_accuracy.png
    comparisons/sign_accuracy_by_cell_type.png
    uncertainty/<model>/between_gene_uncertainty_vs_difficulty.png
    uncertainty/<model>/within_gene_uncertainty_vs_error.png
    tables/per_seed_metrics.csv
    tables/model_summary.csv
    tables/scpredixcan_performance.csv
    tables/survival_curve_pooled.csv
    tables/survival_curve_by_cell_type.csv
    tables/survival_curve_auc_pooled.csv
    tables/survival_curve_auc_by_cell_type.csv
    tables/spearman_survival_curve_pooled.csv
    tables/spearman_survival_curve_by_cell_type.csv
    tables/spearman_survival_curve_auc_pooled.csv
    tables/spearman_survival_curve_auc_by_cell_type.csv
    tables/survival_curve_auc_uncertainty_ablation.csv
    tables/sign_accuracy_pooled.csv
    tables/sign_accuracy_by_cell_type.csv
    tables/uncertainty_*.csv

The uncertainty outputs are optional and are generated with
--analyze-uncertainty. See the corresponding CLI help for interpretation.
Besides the binned diagnostics, --analyze-uncertainty writes
tables/uncertainty_per_gene_error_correlation.csv: one row per
model/seed/cell type/gene holding the Spearman correlation between that
gene's prediction errors and its total, aleatoric, and epistemic
uncertainty, taken across individuals. Positive values mean the model is
more uncertain exactly where it is more wrong. Genes whose correlation is
undefined (fewer than --min-individuals paired donors, or no spread in
error or uncertainty) are omitted. --uncertainty-only narrows a run to the
uncertainty analysis alone: it implies --analyze-uncertainty, skips every
other plot and table, and ignores deterministic models entirely.
The m1 columns and plot are generated only with --compute-m1. A reported
scPrediXcan reference can additionally be included with
--scpredixcan-performance.

m1 = (1 - pi0) * (genes tested) estimates the number of true-positive genes
via the q-value framework, mirroring scPrediXcan's use of m1 = pi1 * #genes.
pi0 (the null fraction) is estimated with qvalue::pi0est-equivalent logic;
--pi0-method selects "smoother" (a natural cubic smoothing spline over the
lambda grid, qvalue's default and the method tied to the Storey & Tibshirani
2003 paper scPrediXcan cites) or "bootstrap" (qvalue's closed-form bootstrap
alternative). See --pi0-method and --smooth-df for details.

As an alternative, threshold-free view of the same tested genes, the
survival curve plots, for a grid of thresholds t in [0, 1], the fraction of
tested genes with |Pearson r| >= t and, in parallel outputs, |Spearman rho|
>= t. The absolute value is used for the same sign-invariant reason as m1:
up- vs down-regulation is not distinguished here. This is computed both
pooled across all cell types (one curve per model) and faceted per cell type
(one subplot per cell type, models overlaid). In both cases the curve is
first computed per model/seed, then summarized as the mean +/- sample SD
across seeds. See --survival-thresholds for the threshold grid resolution.

To directly compare models with a single number (per cell type or pooled),
the survival_curve_auc plots show the trapezoidal area under each
model/seed's fraction-based survival curve, again summarized as the seed
mean +/- SD. The per-cell-type panel uses the same grouped horizontal-bar
layout as the m1 comparison. Since S(t) = P(|r| >= t) for t in [0, 1],
each AUC is approximately the corresponding mean absolute Pearson r or
Spearman rho across tested genes -- a single, gene-count-independent number
that rewards stronger typical correlations, directly comparable across
models and cell types.

Optionally, --uncertainty-ablation-percents re-evaluates that AUC after
dropping the top X% most-uncertain individuals, ranked within each gene
by a chosen uncertainty map (default: totvar). The 0% point is the
uncertainty-aligned baseline (only gene-individual pairs with a finite
uncertainty contribute), so later percentages are comparable. The result
is a per-cell-type line plot of AUC versus percent removed, with one
line per probabilistic model (seed mean +/- SD). Deterministic models
are skipped because they have no uncertainty maps.

The survival curve is magnitude-only: it says nothing about whether a gene
with strong |Pearson r| is actually correlated in the expected (positive)
direction, i.e. whether the model's predicted expression truly tracks
observed expression rather than anti-tracking it. The sign_accuracy plots
answer this with the conditional sign accuracy, P(r > 0 | |r| >= t), for
the same threshold grid t in [0, 1] and the same "tested" gene set: among
the genes that pass an absolute-correlation bar of t (the "predictable"
genes at that threshold), what fraction are positively rather than
negatively correlated. It is computed pooled across cell types and faceted
per cell type, in both cases as the seed mean +/- SD, exactly mirroring the
survival curve's layout. A dashed line at 0.5 marks chance-level sign
agreement. Because the conditioning set shrinks as t grows, the curve
becomes noisier (and eventually undefined, leaving a gap) at thresholds no
gene reaches; see tables/sign_accuracy_*.csv for the underlying values.
"""


INDIVIDUAL_IDS = [
    "1001", "1002", "1003", "1004", "1005",
    "3", "7", "8", "11", "19", "24", "37", "45", "56", "60",
    "65", "82", "84", "99", "107", "112", "117", "119", "123", "127",
    "128", "137", "139", "141", "152", "168", "169", "172", "182", "185",
    "187", "189", "191", "192", "223", "232", "244", "246", "247", "253",
    "256", "257", "260", "265", "267", "272", "284", "287", "291", "302",
    "303", "304", "306", "307", "309", "313", "316", "328", "332", "355",
    "356", "365", "377", "384", "394", "399", "404", "410", "417", "418",
    "419", "420", "423", "434", "437", "439", "458", "466", "472", "485",
    "490", "494", "495", "499", "503", "529", "534", "540", "543", "562",
    "567", "572", "579", "585", "586", "590", "593", "599", "616", "617",
    "621", "622", "628", "635", "637", "641", "652", "661", "668", "671",
    "672", "673", "675", "680", "682", "688", "693", "701", "708", "712",
    "716", "717", "719", "721", "731", "735", "738", "751", "753", "756",
    "763", "765", "772", "777", "780", "782", "784", "789", "794", "798",
    "805", "806", "810", "813", "831", "832", "847", "848", "849", "862",
    "870", "874", "881", "886", "893", "900", "905", "906", "907", "908",
    "924", "927", "931", "933", "938", "948", "956", "957", "958", "960",
    "970", "975", "979", "982", "986", "987", "990", "993", "995", "997",
    "1007", "1012", "1027", "1029", "1031", "1033", "1045", "1047", "1049",
    "1057", "1064", "1066", "1071", "1073", "1080"
]

KEY_COLUMNS = ("gene", "chrom", "tss")
UNCERTAINTY_KINDS = ("totvar", "aleatoric", "epistemic")
DEFAULT_PVALUE_BINS = 20
DEFAULT_UNCERTAINTY_BINS = 5
DEFAULT_MIN_INDIVIDUALS = 3
DEFAULT_SURVIVAL_THRESHOLDS = 101
DEFAULT_ABLATION_PERCENTS = (0.0, 5.0, 10.0, 15.0, 20.0, 25.0, 30.0, 35.0, 40.0, 45.0, 50.0)
DEFAULT_ABLATION_PERCENTS_TEXT = ",".join(
    str(int(percent)) if percent == int(percent) else str(percent)
    for percent in DEFAULT_ABLATION_PERCENTS
)

# Matches qvalue::pi0est's default lambda grid, seq(0.05, 0.95, 0.05).
PI0_LAMBDAS = np.linspace(0.05, 0.95, 19)
DEFAULT_SMOOTH_DF = 3.0


@dataclass(frozen=True)
class SeedLayout:
    """Resolved files for one model seed."""

    name: str
    path: Path
    probabilistic: bool
    predictions: dict[str, Path]
    uncertainties: dict[str, dict[str, Path]]


@dataclass(frozen=True)
class ModelLayout:
    """Resolved seed layouts for one model."""

    name: str
    path: Path
    probabilistic: bool
    seeds: tuple[SeedLayout, ...]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate per-gene Pearson and Spearman correlations for OneK1K "
            "predictions from multiple models and seeds."
        ),
    )
    parser.add_argument(
        "-g",
        "--ground-truth",
        type=Path,
        required=True,
        help="Directory containing one ground-truth CSV per cell type.",
    )
    parser.add_argument(
        "-p",
        "--predictions",
        type=Path,
        required=True,
        help=(
            "Root directory containing <model>/<seed>/... prediction outputs "
            "from get_student_data.py."
        ),
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        required=True,
        help="Directory in which plots and summary tables are written.",
    )
    parser.add_argument(
        "--compute-m1",
        action="store_true",
        help=(
            "Estimate q-value-framework m1 values and write the m1 comparison "
            "plot. Disabled by default because pi0 estimation is optional."
        ),
    )
    parser.add_argument(
        "--scpredixcan-performance",
        "--scpredixcan-performance-file",
        dest="scpredixcan_performance",
        type=Path,
        default=None,
        metavar="XLSX",
        help=(
            "Optional scPrediXcan Excel workbook. When supplied, reported "
            "OneK1K m1 values are added as a 'scPrediXcan' reference in the "
            "m1 comparison and the parsed metrics are saved below tables/. "
            "Requires --compute-m1. Omit this argument to disable the reference."
        ),
    )
    parser.add_argument(
        "--pvalue-threshold",
        type=float,
        default=0.05,
        help=(
            "Nominal two-sided Pearson p-value threshold used for the gene "
            "count comparison. Default: 0.05."
        ),
    )
    parser.add_argument(
        "--pi0-method",
        choices=("smoother", "bootstrap"),
        default="smoother",
        help=(
            "Method used to estimate the null fraction pi0 for the m1 "
            "q-value metric, mirroring qvalue::pi0est's pi0.method. "
            "'smoother' fits a natural cubic smoothing spline to "
            "pi0(lambda) across the lambda grid (qvalue's default, and the "
            "method tied to the Storey & Tibshirani 2003 paper scPrediXcan "
            "cites for its m1 metric); 'bootstrap' uses the closed-form "
            "minimum-MSE lambda selection from qvalue's bootstrap option. "
            "Default: smoother."
        ),
    )
    parser.add_argument(
        "--smooth-df",
        type=float,
        default=DEFAULT_SMOOTH_DF,
        help=(
            "Degrees of freedom for the natural cubic smoothing spline used "
            "by --pi0-method=smoother, mirroring qvalue::pi0est's "
            f"smooth.df. Default: {DEFAULT_SMOOTH_DF:g} (qvalue's default)."
        ),
    )
    parser.add_argument(
        "--pvalue-bins",
        type=int,
        default=DEFAULT_PVALUE_BINS,
        help=(
            "Number of equal-width bins in each p-value histogram. "
            f"Default: {DEFAULT_PVALUE_BINS}."
        ),
    )
    parser.add_argument(
        "--min-individuals",
        type=int,
        default=DEFAULT_MIN_INDIVIDUALS,
        help=(
            "Minimum number of finite paired test individuals required for a "
            "gene's Pearson correlation. Default: 3."
        ),
    )
    parser.add_argument(
        "--survival-thresholds",
        type=int,
        default=DEFAULT_SURVIVAL_THRESHOLDS,
        metavar="N",
        help=(
            "Number of equally spaced absolute-correlation thresholds in "
            "[0, 1] used for the Pearson and Spearman survival-curve plots "
            "(fraction of tested genes with "
            f"|r| >= threshold). Default: {DEFAULT_SURVIVAL_THRESHOLDS}."
        ),
    )
    parser.add_argument(
        "--uncertainty-ablation-percents",
        nargs="?",
        const=DEFAULT_ABLATION_PERCENTS_TEXT,
        default=None,
        metavar="PCTS",
        help=(
            "If given, drop the top X%% most-uncertain individuals (ranked "
            "within each gene) and recompute the survival-curve AUC at each "
            "percentage. Pass the flag alone to use the default grid "
            f"{DEFAULT_ABLATION_PERCENTS_TEXT}, or a comma-separated list "
            "such as 0,10,20,30. 0 is always included as the "
            "uncertainty-aligned baseline. Omit the flag to skip the "
            "ablation. Only probabilistic models are evaluated."
        ),
    )
    parser.add_argument(
        "--ablation-uncertainty",
        choices=UNCERTAINTY_KINDS,
        default="totvar",
        help=(
            "Uncertainty map used to rank individuals for "
            "--uncertainty-ablation-percents. Default: totvar."
        ),
    )
    parser.add_argument(
        "--analyze-uncertainty",
        action="store_true",
        help=(
            "For probabilistic models, also generate between-gene and "
            "within-gene uncertainty diagnostics for total, aleatoric, and "
            "epistemic variance, plus a per-gene table of the Spearman "
            "correlation between prediction error and each uncertainty."
        ),
    )
    parser.add_argument(
        "--uncertainty-only",
        action="store_true",
        help=(
            "Restrict the run to the uncertainty analysis: implies "
            "--analyze-uncertainty and skips the p-value, m1, survival-curve "
            "and sign-accuracy outputs along with their tables. Deterministic "
            "models are skipped entirely. --uncertainty-ablation-percents is "
            "still honoured when explicitly requested."
        ),
    )
    parser.add_argument(
        "--uncertainty-bins",
        type=int,
        default=DEFAULT_UNCERTAINTY_BINS,
        help=(
            "Number of equal-count uncertainty bins used by both uncertainty "
            f"diagnostics. Default: {DEFAULT_UNCERTAINTY_BINS}."
        ),
    )
    parser.add_argument(
        "--uncertainty-error-transform",
        choices=("none", "log1p"),
        default=None,
        help=(
            "Scale on which squared errors are computed for the within-gene "
            "uncertainty diagnostic. Use 'log1p' when get_student_data.py "
            "inverted log-normalized predicted means with expm1, because its "
            "uncertainties remain variances in log1p space; otherwise use "
            "'none'. This option must be specified with "
            "--analyze-uncertainty."
        ),
    )
    parser.add_argument(
        "--save-per-gene",
        action="store_true",
        help=(
            "Also save gzip-compressed per-gene Pearson r/p-value and "
            "Spearman rho tables below "
            "tables/per_gene/. Disabled by default because these files can be "
            "large."
        ),
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=180,
        help="PNG resolution. Default: 180.",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="count",
        default=0,
        help="Increase logging verbosity.",
    )

    args = parser.parse_args()
    if not 0.0 <= args.pvalue_threshold <= 1.0:
        parser.error("--pvalue-threshold must lie in [0, 1].")
    if (
        args.scpredixcan_performance is not None
        and not args.compute_m1
        and not args.uncertainty_only
    ):
        parser.error("--scpredixcan-performance requires --compute-m1.")
    if args.compute_m1 and not 2.0 < args.smooth_df < len(PI0_LAMBDAS):
        parser.error(
            f"--smooth-df must lie strictly between 2 and {len(PI0_LAMBDAS)} "
            "(the number of lambda grid points), matching the valid range "
            "of a natural cubic smoothing spline's effective degrees of "
            "freedom."
        )
    if args.pvalue_bins < 2:
        parser.error("--pvalue-bins must be at least 2.")
    if args.min_individuals < 3:
        parser.error("--min-individuals must be at least 3 for Pearson p-values.")
    if args.survival_thresholds < 2:
        parser.error("--survival-thresholds must be at least 2.")
    if args.uncertainty_ablation_percents is not None:
        try:
            args.uncertainty_ablation_percents = parse_ablation_percents(
                args.uncertainty_ablation_percents,
            )
        except ValueError as error:
            parser.error(str(error))
    if args.uncertainty_bins < 2:
        parser.error("--uncertainty-bins must be at least 2.")
    if args.uncertainty_only:
        args.analyze_uncertainty = True
    if (
        args.analyze_uncertainty
        and args.uncertainty_error_transform is None
    ):
        parser.error(
            "--analyze-uncertainty requires "
            "--uncertainty-error-transform {none,log1p}; the correct variance "
            "scale cannot be inferred from the CSV files."
        )
    if args.dpi <= 0:
        parser.error("--dpi must be positive.")
    return args


def setup_logging(verbosity: int) -> None:
    level = logging.DEBUG if verbosity else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    logging.getLogger("matplotlib").setLevel(logging.WARNING)


def parse_ablation_percents(raw: str) -> tuple[float, ...]:
    """Parse a comma-separated percent list in [0, 100), always including 0."""
    parts = [part.strip() for part in str(raw).split(",") if part.strip()]
    if not parts:
        raise ValueError(
            "--uncertainty-ablation-percents must contain at least one number."
        )
    percents: list[float] = []
    for part in parts:
        try:
            value = float(part)
        except ValueError as error:
            raise ValueError(
                f"Invalid ablation percent {part!r}; expected a number in "
                "[0, 100)."
            ) from error
        if not 0.0 <= value < 100.0:
            raise ValueError(
                f"Ablation percent {value:g} must lie in [0, 100)."
            )
        percents.append(value)
    if 0.0 not in percents:
        percents.append(0.0)
    return tuple(sorted(set(percents)))


def natural_sort_key(value: str) -> tuple[object, ...]:
    """Sort strings containing numbers in human order (2 before 10)."""
    return tuple(
        int(part) if part.isdigit() else part.casefold()
        for part in re.split(r"(\d+)", value)
    )


def csv_files(directory: Path) -> dict[str, Path]:
    """Map cell-type stems to CSV paths in one directory."""
    result: dict[str, Path] = {}
    for path in sorted(directory.glob("*.csv"), key=lambda p: natural_sort_key(p.name)):
        if path.stem in result:
            raise ValueError(
                f"Duplicate cell-type stem {path.stem!r} in {directory}."
            )
        result[path.stem] = path
    return result


def discover_seed(seed_dir: Path) -> SeedLayout:
    present_uncertainty_dirs = [
        kind for kind in ("preds", *UNCERTAINTY_KINDS)
        if (seed_dir / kind).is_dir()
    ]
    probabilistic = bool(present_uncertainty_dirs)

    if probabilistic:
        required = {"preds", *UNCERTAINTY_KINDS}
        missing = sorted(required - set(present_uncertainty_dirs))
        if missing:
            raise ValueError(
                f"Probabilistic-looking seed directory {seed_dir} is missing "
                f"required subdirectories: {missing}."
            )
        predictions = csv_files(seed_dir / "preds")
        uncertainties = {
            kind: csv_files(seed_dir / kind)
            for kind in UNCERTAINTY_KINDS
        }
        for kind, files in uncertainties.items():
            if set(files) != set(predictions):
                missing_files = sorted(set(predictions) - set(files))
                extra_files = sorted(set(files) - set(predictions))
                raise ValueError(
                    f"{seed_dir / kind} does not match {seed_dir / 'preds'}: "
                    f"missing={missing_files}, extra={extra_files}."
                )
    else:
        predictions = csv_files(seed_dir)
        uncertainties = {}

    if not predictions:
        expected = f"{seed_dir / 'preds'}/*.csv" if probabilistic else f"{seed_dir}/*.csv"
        raise ValueError(f"No prediction CSVs found at {expected}.")

    return SeedLayout(
        name=seed_dir.name,
        path=seed_dir,
        probabilistic=probabilistic,
        predictions=predictions,
        uncertainties=uncertainties,
    )


def discover_models(prediction_root: Path) -> tuple[ModelLayout, ...]:
    if not prediction_root.is_dir():
        raise ValueError(
            f"Predictions path is not a directory: {prediction_root}"
        )

    model_dirs = sorted(
        (
            path for path in prediction_root.iterdir()
            if path.is_dir() and not path.name.startswith(".")
        ),
        key=lambda path: natural_sort_key(path.name),
    )
    if not model_dirs:
        raise ValueError(f"No model directories found in {prediction_root}.")

    models: list[ModelLayout] = []
    for model_dir in model_dirs:
        seed_dirs = sorted(
            (
                path for path in model_dir.iterdir()
                if path.is_dir() and not path.name.startswith(".")
            ),
            key=lambda path: natural_sort_key(path.name),
        )
        if not seed_dirs:
            raise ValueError(f"Model {model_dir.name!r} has no seed directories.")

        seeds = tuple(discover_seed(seed_dir) for seed_dir in seed_dirs)
        layouts = {seed.probabilistic for seed in seeds}
        if len(layouts) != 1:
            raise ValueError(
                f"Model {model_dir.name!r} mixes deterministic and "
                "probabilistic seed layouts."
            )
        reference_cell_types = set(seeds[0].predictions)
        for seed in seeds[1:]:
            seed_cell_types = set(seed.predictions)
            if seed_cell_types != reference_cell_types:
                missing = sorted(
                    reference_cell_types - seed_cell_types,
                    key=natural_sort_key,
                )
                extra = sorted(
                    seed_cell_types - reference_cell_types,
                    key=natural_sort_key,
                )
                raise ValueError(
                    f"Model {model_dir.name!r} does not contain the same "
                    f"cell-type CSVs in every seed. Relative to seed "
                    f"{seeds[0].name!r}, seed {seed.name!r} has "
                    f"missing={missing}, extra={extra}."
                )
        models.append(
            ModelLayout(
                name=model_dir.name,
                path=model_dir,
                probabilistic=seeds[0].probabilistic,
                seeds=seeds,
            )
        )

    return tuple(models)


def discover_ground_truth(ground_truth_dir: Path) -> dict[str, Path]:
    if not ground_truth_dir.is_dir():
        raise ValueError(
            f"Ground-truth path is not a directory: {ground_truth_dir}"
        )
    files = csv_files(ground_truth_dir)
    if not files:
        raise ValueError(
            f"No ground-truth CSV files found in {ground_truth_dir}."
        )
    return files


def normalize_individual_id(value: object) -> str | None:
    """Canonicalize a donor ID, excluding ``OneK1K_<ID>_<copy>`` duplicates."""
    text = str(value).strip()
    if re.fullmatch(r"OneK1K_\d+_\d+", text, flags=re.IGNORECASE):
        return None
    match = re.search(r"(\d+)$", text)
    return match.group(1) if match else text


def normalize_chromosome(value: object) -> str:
    text = str(value).strip()
    text = re.sub(r"^chr", "", text, flags=re.IGNORECASE)
    if re.fullmatch(r"\d+\.0", text):
        text = text[:-2]
    return text.upper()


def build_gene_index(frame: pd.DataFrame, path: Path) -> pd.MultiIndex:
    missing = [column for column in KEY_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(
            f"{path} is missing gene-key columns {missing}; expected "
            f"{list(KEY_COLUMNS)}."
        )

    gene = frame["gene"].astype("string").str.strip()
    if gene.isna().any() or (gene == "").any():
        raise ValueError(f"{path} contains missing/empty gene identifiers.")

    if frame["chrom"].isna().any():
        raise ValueError(f"{path} contains missing chromosome values.")
    chrom = frame["chrom"].map(normalize_chromosome)
    if (chrom == "").any():
        raise ValueError(f"{path} contains empty chromosome values.")
    tss_float = pd.to_numeric(frame["tss"], errors="coerce").to_numpy(
        dtype=np.float64,
    )
    if not np.isfinite(tss_float).all():
        raise ValueError(f"{path} contains non-numeric or missing tss values.")
    if not np.all(tss_float == np.floor(tss_float)):
        raise ValueError(f"{path} contains non-integral tss values.")

    index = pd.MultiIndex.from_arrays(
        [
            gene.astype(str).to_numpy(),
            chrom.astype(str).to_numpy(),
            tss_float.astype(np.int64),
        ],
        names=list(KEY_COLUMNS),
    )
    if index.has_duplicates:
        duplicate_examples = index[index.duplicated()].unique().tolist()[:5]
        raise ValueError(
            f"{path} contains duplicate (gene, chrom, tss) keys. "
            f"Examples: {duplicate_examples}"
        )
    return index


def load_expression_table(
    path: Path,
    test_ids: set[str],
    *,
    require_all_test_ids: bool = True,
) -> pd.DataFrame:
    """Load only requested individual columns and index rows by the gene triple."""
    logging.debug("Reading %s", path)
    header = pd.read_csv(path, nrows=0)
    missing = [column for column in KEY_COLUMNS if column not in header.columns]
    if missing:
        raise ValueError(
            f"{path} is missing required columns {missing}; available columns "
            f"start with {header.columns.tolist()[:10]}."
        )

    original_by_id: dict[str, str] = {}
    ignored_duplicate_columns: list[str] = []
    for column in header.columns:
        if column in KEY_COLUMNS:
            continue
        canonical = normalize_individual_id(column)
        if canonical is None:
            ignored_duplicate_columns.append(str(column))
            continue
        if canonical not in test_ids:
            continue
        if canonical in original_by_id:
            raise ValueError(
                f"{path} has multiple columns that normalize to individual "
                f"{canonical!r}: {original_by_id[canonical]!r} and {column!r}."
            )
        original_by_id[canonical] = column

    #if ignored_duplicate_columns:
    #    logging.warning(
    #        "%s: excluding %d duplicate-suffixed donor column(s), including %s",
    #        path,
    #        len(ignored_duplicate_columns),
    #        ignored_duplicate_columns[:5],
    #    )

    if not original_by_id:
        raise ValueError(
            f"{path} contains none of the {len(test_ids)} requested test "
            "individuals."
        )
    missing_test_ids = sorted(
        test_ids - set(original_by_id),
        key=natural_sort_key,
    )
    if missing_test_ids and require_all_test_ids:
        preview = missing_test_ids[:20]
        suffix = "..." if len(missing_test_ids) > len(preview) else ""
        raise ValueError(
            f"{path} is missing {len(missing_test_ids)} requested test "
            f"individual columns: {preview}{suffix}. Every input table must "
            "contain the complete test-ID allowlist so seeds and models are "
            "compared on the same donors."
        )
    if missing_test_ids:
        preview = missing_test_ids[:20]
        suffix = "..." if len(missing_test_ids) > len(preview) else ""
        logging.warning(
            "%s is missing %d requested test individuals for this cell type; "
            "evaluation will use the available intersection: %s%s",
            path,
            len(missing_test_ids),
            preview,
            suffix,
        )

    usecols = [*KEY_COLUMNS, *original_by_id.values()]
    frame = pd.read_csv(path, usecols=usecols)
    index = build_gene_index(frame, path)

    values = frame[list(original_by_id.values())].apply(
        pd.to_numeric,
        errors="coerce",
    )
    values.columns = list(original_by_id)
    values.index = index
    return values


def align_expression_tables(
    *tables: pd.DataFrame,
) -> tuple[pd.DataFrame, ...]:
    """Intersect tables on gene keys and requested individuals."""
    if len(tables) < 2:
        raise ValueError("At least two tables are required for alignment.")

    first = tables[0]
    common_index = first.index
    for table in tables[1:]:
        common_index = common_index[
            common_index.isin(table.index)
        ]

    requested_order = [str(individual) for individual in INDIVIDUAL_IDS]
    common_columns = [
        individual for individual in requested_order
        if all(individual in table.columns for table in tables)
    ]

    if len(common_index) == 0:
        raise ValueError(
            "The expression tables share no (gene, chrom, tss) keys."
        )
    if len(common_columns) == 0:
        raise ValueError(
            "The expression tables share no requested test individuals."
        )

    return tuple(
        table.loc[common_index, common_columns]
        for table in tables
    )


def rowwise_pearson(
    predictions: np.ndarray,
    observations: np.ndarray,
    min_individuals: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Vectorized pairwise-complete Pearson r and two-sided p-values by row."""
    predictions = np.asarray(predictions, dtype=np.float64)
    observations = np.asarray(observations, dtype=np.float64)
    if predictions.shape != observations.shape or predictions.ndim != 2:
        raise ValueError(
            "Predictions and observations must be equally shaped 2D arrays."
        )

    mask = np.isfinite(predictions) & np.isfinite(observations)
    n = mask.sum(axis=1).astype(np.int64)
    n_safe = np.maximum(n, 1).astype(np.float64)
    mean_x = np.where(mask, predictions, 0.0).sum(axis=1) / n_safe
    mean_y = np.where(mask, observations, 0.0).sum(axis=1) / n_safe
    # Center explicitly instead of subtracting raw sufficient statistics. This
    # remains stable for genes with large means but small donor-to-donor changes.
    x = np.where(mask, predictions - mean_x[:, None], 0.0)
    y = np.where(mask, observations - mean_y[:, None], 0.0)
    ss_x = np.square(x).sum(axis=1)
    ss_y = np.square(y).sum(axis=1)
    cross = (x * y).sum(axis=1)
    denominator = np.sqrt(ss_x * ss_y)

    valid = (
        (n >= min_individuals)
        & np.isfinite(denominator)
        & (denominator > np.finfo(np.float64).eps)
    )
    r = np.full(predictions.shape[0], np.nan, dtype=np.float64)
    r[valid] = np.clip(cross[valid] / denominator[valid], -1.0, 1.0)

    pvalue = np.full_like(r, np.nan)
    nonperfect = valid & (np.abs(r) < 1.0)
    degrees_freedom = n[nonperfect] - 2
    with np.errstate(divide="ignore", invalid="ignore"):
        t_statistic = (
            r[nonperfect]
            * np.sqrt(degrees_freedom)
            / np.sqrt(1.0 - np.square(r[nonperfect]))
        )
    pvalue[nonperfect] = 2.0 * stats.t.sf(
        np.abs(t_statistic),
        degrees_freedom,
    )
    pvalue[valid & (np.abs(r) >= 1.0)] = 0.0
    pvalue = np.clip(pvalue, 0.0, 1.0)
    return r, pvalue, n


def rowwise_spearman(
    predictions: np.ndarray,
    observations: np.ndarray,
    min_individuals: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Pairwise-complete Spearman rho by row, with average ranks for ties."""
    predictions = np.asarray(predictions, dtype=np.float64)
    observations = np.asarray(observations, dtype=np.float64)
    if predictions.shape != observations.shape or predictions.ndim != 2:
        raise ValueError(
            "Predictions and observations must be equally shaped 2D arrays."
        )

    ranked_predictions = np.full_like(predictions, np.nan)
    ranked_observations = np.full_like(observations, np.nan)
    n = np.zeros(predictions.shape[0], dtype=np.int64)
    for index, (predicted, observed) in enumerate(zip(predictions, observations)):
        valid = np.isfinite(predicted) & np.isfinite(observed)
        n[index] = int(valid.sum())
        if n[index] < min_individuals:
            continue
        ranked_predictions[index, valid] = stats.rankdata(
            predicted[valid],
            method="average",
        )
        ranked_observations[index, valid] = stats.rankdata(
            observed[valid],
            method="average",
        )
    rho, _, _ = rowwise_pearson(
        ranked_predictions,
        ranked_observations,
        min_individuals,
    )
    return rho, n


def _pi0_lambda_curve(
    values: np.ndarray,
    lambdas: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Per-lambda null-fraction estimates pi0(lambda) = mean(p>=lambda)/(1-lambda).

    Shared by every pi0.method, matching qvalue::pi0est's common first step
    before either the smoother or bootstrap tuning-parameter selection.
    """
    if ((values < 0.0) | (values > 1.0)).any():
        raise ValueError("p-values must lie in [0, 1].")
    m = float(values.size)
    w = np.asarray(
        [(values >= threshold).sum() for threshold in lambdas],
        dtype=np.float64,
    )
    pi0_lambda = w / (m * (1.0 - lambdas))
    return pi0_lambda, w, m


def storey_bootstrap_pi0(
    pvalues: np.ndarray,
    lambdas: np.ndarray = PI0_LAMBDAS,
) -> float:
    """Estimate the null fraction with a robust Storey-bootstrap variant.

    This uses the lambda grid and closed-form minimum-MSE selection from the
    ``pi0.method="bootstrap"`` option in StoreyLab's qvalue package. Unlike the
    R function's defensive rejection of a p-value vector truncated below the
    largest lambda, this variant retains the boundary estimate pi0=0 so one
    unusual cell type cannot abort the entire evaluation. Such rows are
    explicitly flagged in the output table and logs.
    """
    values = np.asarray(pvalues, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return float("nan")

    pi0_lambda, w, m = _pi0_lambda_curve(values, lambdas)
    reference = float(np.quantile(pi0_lambda, 0.1))
    mse = (
        (w / (m * m * np.square(1.0 - lambdas))) * (1.0 - w / m)
        + np.square(pi0_lambda - reference)
    )
    best = pi0_lambda[mse == np.nanmin(mse)]
    pi0 = float(np.nanmin(best))
    return float(np.clip(pi0, 0.0, 1.0))


def _natural_cubic_spline_penalty(x: np.ndarray) -> np.ndarray:
    """Green & Silverman (1994) roughness-penalty matrix K.

    For knots ``x`` (strictly increasing, length >= 3) and function values
    ``g`` at those knots, ``g @ K @ g`` equals the integrated squared second
    derivative of the natural cubic spline interpolating ``(x, g)``. This is
    the standard closed-form building block of a natural cubic smoothing
    spline (the same fitting criterion underlying R's ``smooth.spline`` and
    qvalue's ``pi0.method="smoother"``).
    """
    h = np.diff(x)
    n = x.size
    m = n - 2
    Q = np.zeros((n, m), dtype=np.float64)
    R = np.zeros((m, m), dtype=np.float64)
    for j in range(m):
        Q[j, j] = 1.0 / h[j]
        Q[j + 1, j] = -1.0 / h[j] - 1.0 / h[j + 1]
        Q[j + 2, j] = 1.0 / h[j + 1]
        R[j, j] = (h[j] + h[j + 1]) / 3.0
        if j + 1 < m:
            R[j, j + 1] = h[j + 1] / 6.0
            R[j + 1, j] = h[j + 1] / 6.0
    return Q @ np.linalg.solve(R, Q.T)


_PI0_SPLINE_EIGVALS, _PI0_SPLINE_EIGVECS = np.linalg.eigh(
    _natural_cubic_spline_penalty(PI0_LAMBDAS),
)


def _smoothing_spline_effective_df(lam: float) -> float:
    """Effective degrees of freedom (smoother-matrix trace) at penalty ``lam``."""
    return float(np.sum(1.0 / (1.0 + lam * _PI0_SPLINE_EIGVALS)))


def _smoothing_spline_fit(values: np.ndarray, lam: float) -> np.ndarray:
    """Fitted natural cubic smoothing spline values at the knots ``PI0_LAMBDAS``."""
    shrinkage = 1.0 / (1.0 + lam * _PI0_SPLINE_EIGVALS)
    smoother = _PI0_SPLINE_EIGVECS @ (shrinkage[:, None] * _PI0_SPLINE_EIGVECS.T)
    return smoother @ values


def storey_smoother_pi0(
    pvalues: np.ndarray,
    lambdas: np.ndarray = PI0_LAMBDAS,
    smooth_df: float = DEFAULT_SMOOTH_DF,
) -> float:
    """Estimate the null fraction with qvalue's default smoother method.

    Fits a natural cubic smoothing spline (degrees of freedom ``smooth_df``,
    matching qvalue::pi0est's ``smooth.df=3`` default) to pi0(lambda) over
    the lambda grid and evaluates it at max(lambda), exactly mirroring
    ``pi0.method="smoother"`` (the qvalue package default, and the method
    tied to the Storey & Tibshirani (2003) reference scPrediXcan cites for
    its m1 metric). As with :func:`storey_bootstrap_pi0`, boundary estimates
    are retained (not reset to 1) rather than aborting; such rows are
    flagged via ``pi0_boundary_case`` in the output table and logs.
    """
    values = np.asarray(pvalues, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return float("nan")

    pi0_lambda, _, _ = _pi0_lambda_curve(values, lambdas)
    effective_df = _smoothing_spline_effective_df
    lo_df, hi_df = effective_df(1e10), effective_df(1e-10)
    if not lo_df < smooth_df < hi_df:
        raise ValueError(
            f"smooth_df={smooth_df:g} is outside the achievable effective "
            f"degrees of freedom range ({lo_df:.4f}, {hi_df:.4f}) for "
            f"{len(lambdas)} lambda knots."
        )
    log_lam = brentq(
        lambda log_lam: effective_df(math.exp(log_lam)) - smooth_df,
        math.log(1e-10),
        math.log(1e10),
        xtol=1e-10,
    )
    fitted = _smoothing_spline_fit(pi0_lambda, math.exp(log_lam))
    pi0 = float(np.clip(fitted[-1], 0.0, 1.0))
    return pi0


def estimate_pi0(
    pvalues: np.ndarray,
    method: str,
    smooth_df: float = DEFAULT_SMOOTH_DF,
) -> float:
    """Dispatch to the requested qvalue::pi0est-equivalent pi0 estimator."""
    if method == "smoother":
        return storey_smoother_pi0(pvalues, smooth_df=smooth_df)
    if method == "bootstrap":
        return storey_bootstrap_pi0(pvalues)
    raise ValueError(f"Unknown pi0 method: {method!r}")


def sample_std(values: np.ndarray, axis: int = 0) -> np.ndarray:
    """Sample SD across seeds, returning zero when only one seed is available."""
    array = np.asarray(values, dtype=np.float64)
    n = array.shape[axis]
    if n <= 1:
        shape = list(array.shape)
        del shape[axis]
        return np.zeros(shape, dtype=np.float64)
    return np.nanstd(array, axis=axis, ddof=1)


def equal_count_curve(
    x: np.ndarray,
    y: np.ndarray,
    n_bins: int,
) -> tuple[np.ndarray, int]:
    """Mean y in deterministic, equal-count bins after sorting by x."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    valid = np.isfinite(x) & np.isfinite(y)
    x = x[valid]
    y = y[valid]
    if x.size < n_bins or np.ptp(x) <= np.finfo(np.float64).eps:
        return np.full(n_bins, np.nan), int(x.size)

    order = np.argsort(x, kind="mergesort")
    groups = np.array_split(order, n_bins)
    curve = np.asarray(
        [np.mean(y[group]) for group in groups],
        dtype=np.float64,
    )
    return curve, int(x.size)


def compute_survival_curve(
    absolute_correlations: np.ndarray,
    thresholds: np.ndarray,
) -> tuple[np.ndarray, int]:
    """Fraction of tested genes with an absolute correlation >= each threshold.

    This is a threshold-free complement to the m1 significant-gene count:
    instead of picking one p-value cutoff, it summarizes the whole
    distribution of absolute correlations via its (empirical) survival
    function S(t) = P(|r| >= t).
    """
    values = np.asarray(absolute_correlations, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return np.full(thresholds.shape, np.nan, dtype=np.float64), 0
    counts = np.asarray(
        [(values >= threshold).sum() for threshold in thresholds],
        dtype=np.float64,
    )
    return counts / values.size, int(values.size)


def compute_sign_accuracy_curve(
    pearson_r: np.ndarray,
    thresholds: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """P(r > 0 | |r| >= t) for a grid of thresholds t, among tested genes.

    The survival curve above is magnitude-only and cannot distinguish a
    model whose strongest correlations are genuinely predictive (r > 0)
    from one whose strongest correlations are anti-correlated (r < 0). This
    conditional sign accuracy fills that gap: among the "predictable" genes
    that clear an absolute-correlation bar of t, what fraction have the
    expected positive sign. Returns both the accuracy curve and the number
    of genes the conditioning set contains at each threshold (0 when no
    gene reaches that threshold, in which case accuracy is NaN there).
    """
    r = np.asarray(pearson_r, dtype=np.float64)
    r = r[np.isfinite(r)]
    n_at_threshold = np.zeros(thresholds.shape, dtype=np.float64)
    accuracy = np.full(thresholds.shape, np.nan, dtype=np.float64)
    if r.size == 0:
        return accuracy, n_at_threshold

    abs_r = np.abs(r)
    positive = r > 0.0
    for index, threshold in enumerate(thresholds):
        mask = abs_r >= threshold
        count = int(mask.sum())
        n_at_threshold[index] = count
        if count > 0:
            accuracy[index] = float(positive[mask].sum()) / count
    return accuracy, n_at_threshold


def survival_curve_auc(curve: np.ndarray, thresholds: np.ndarray) -> float:
    """Trapezoidal area under a fraction-based survival curve.

    Since S(t) = P(|r| >= t) for t in [0, 1], this area is the expectation
    of a [0, 1]-clipped |r|, i.e. approximately mean(|r|) across tested
    genes. It gives a single, gene-count-independent number per model/seed
    that can be compared directly across models or cell types with
    different numbers of tested genes.
    """
    curve = np.asarray(curve, dtype=np.float64)
    thresholds = np.asarray(thresholds, dtype=np.float64)
    return float(np.sum(np.diff(thresholds) * (curve[:-1] + curve[1:]) / 2.0))


def drop_top_uncertain_individuals(
    predictions: np.ndarray,
    observations: np.ndarray,
    uncertainty: np.ndarray,
    drop_fraction: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Mask the top ``drop_fraction`` of individuals per gene by uncertainty.

    For each gene, individuals with a finite prediction, observation, and
    non-negative uncertainty are ranked by that gene's uncertainty. The
    highest ``floor(n_valid * drop_fraction)`` individuals are set to NaN
    so :func:`rowwise_pearson` ignores them. ``drop_fraction=0`` only
    drops pairs that already lack a finite uncertainty, keeping later
    percentages comparable to this baseline.

    Ranking is within-gene: an individual can be dropped for one gene and
    kept for another.
    """
    if not 0.0 <= drop_fraction < 1.0:
        raise ValueError(
            f"drop_fraction must lie in [0, 1), got {drop_fraction:g}."
        )

    pred = np.asarray(predictions, dtype=np.float64)
    obs = np.asarray(observations, dtype=np.float64)
    unc = np.asarray(uncertainty, dtype=np.float64)
    if pred.shape != obs.shape or pred.shape != unc.shape or pred.ndim != 2:
        raise ValueError(
            "Predictions, observations, and uncertainty must be equally "
            "shaped 2D arrays."
        )

    valid = (
        np.isfinite(pred)
        & np.isfinite(obs)
        & np.isfinite(unc)
        & (unc >= 0.0)
    )
    pred = np.where(valid, pred, np.nan)
    obs = np.where(valid, obs, np.nan)
    if drop_fraction == 0.0:
        return pred, obs

    n_individuals = pred.shape[1]
    n_valid = valid.sum(axis=1)
    n_drop = np.floor(n_valid * drop_fraction).astype(np.int64)
    # Invalid entries sort first via -inf, so the last n_drop columns of
    # the argsort order are the highest-uncertainty valid individuals.
    order = np.argsort(
        np.where(valid, unc, -np.inf),
        axis=1,
        kind="mergesort",
    )
    ranks = np.empty_like(order)
    np.put_along_axis(
        ranks,
        order,
        np.arange(n_individuals, dtype=order.dtype)[None, :],
        axis=1,
    )
    drop = ranks >= (n_individuals - n_drop[:, None])
    pred = np.where(drop, np.nan, pred)
    obs = np.where(drop, np.nan, obs)
    return pred, obs


def summarize_metric_over_seeds(
    per_seed: pd.DataFrame,
    group_columns: list[str],
    metric: str,
) -> pd.DataFrame:
    """Seed mean +/- sample SD of one per-seed metric, grouped by columns."""
    rows: list[dict[str, object]] = []
    for key, group in per_seed.groupby(group_columns, sort=False):
        key_tuple = key if isinstance(key, tuple) else (key,)
        row: dict[str, object] = dict(zip(group_columns, key_tuple))
        row["n_seeds"] = len(group)
        row["seeds"] = ",".join(
            sorted(group["seed"].astype(str), key=natural_sort_key),
        )
        values = group[metric].to_numpy(dtype=np.float64)
        row[f"{metric}_mean"] = float(np.nanmean(values))
        row[f"{metric}_std"] = float(sample_std(values, axis=0))
        rows.append(row)
    return pd.DataFrame(rows)


def spearman_or_nan(x: np.ndarray, y: np.ndarray) -> float:
    valid = np.isfinite(x) & np.isfinite(y)
    x = np.asarray(x, dtype=np.float64)[valid]
    y = np.asarray(y, dtype=np.float64)[valid]
    if (
        x.size < 3
        or np.ptp(x) <= np.finfo(np.float64).eps
        or np.ptp(y) <= np.finfo(np.float64).eps
    ):
        return float("nan")
    return float(stats.spearmanr(x, y).statistic)


def transform_for_uncertainty_error(
    predictions: np.ndarray,
    observations: np.ndarray,
    transform: str,
) -> tuple[np.ndarray, np.ndarray]:
    if transform == "none":
        return predictions, observations
    if transform != "log1p":
        raise ValueError(f"Unknown uncertainty error transform: {transform}")

    pred = np.asarray(predictions, dtype=np.float64).copy()
    obs = np.asarray(observations, dtype=np.float64).copy()
    pred[pred <= -1.0] = np.nan
    obs[obs <= -1.0] = np.nan
    with np.errstate(invalid="ignore"):
        return np.log1p(pred), np.log1p(obs)


def within_gene_uncertainty_curve(
    predictions: np.ndarray,
    observations: np.ndarray,
    uncertainty: np.ndarray,
    n_bins: int,
    min_individuals: int,
    error_transform: str,
) -> tuple[np.ndarray, int]:
    """Rank individuals within each gene and aggregate relative squared error.

    Squared errors are divided by the gene's own mean squared error before
    aggregation. Therefore the curve tests within-gene discrimination rather
    than being dominated by genes on large expression scales. Values above one
    indicate above-average error for that gene.
    """
    pred, obs = transform_for_uncertainty_error(
        predictions,
        observations,
        error_transform,
    )
    unc = np.asarray(uncertainty, dtype=np.float64)

    gene_curves: list[np.ndarray] = []
    minimum = max(min_individuals, n_bins)
    for row in range(pred.shape[0]):
        valid = (
            np.isfinite(pred[row])
            & np.isfinite(obs[row])
            & np.isfinite(unc[row])
            & (unc[row] >= 0.0)
        )
        if int(valid.sum()) < minimum:
            continue

        gene_uncertainty = unc[row, valid]
        if np.ptp(gene_uncertainty) <= np.finfo(np.float64).eps:
            continue

        squared_error = np.square(pred[row, valid] - obs[row, valid])
        mean_error = float(np.mean(squared_error))
        if not np.isfinite(mean_error) or mean_error <= np.finfo(np.float64).eps:
            continue
        relative_error = squared_error / mean_error

        order = np.argsort(gene_uncertainty, kind="mergesort")
        groups = np.array_split(order, n_bins)
        gene_curves.append(
            np.asarray(
                [np.mean(relative_error[group]) for group in groups],
                dtype=np.float64,
            ),
        )

    if not gene_curves:
        return np.full(n_bins, np.nan), 0
    return (
        np.nanmean(np.vstack(gene_curves), axis=0),
        len(gene_curves),
    )


def per_gene_error_uncertainty_spearman(
    predictions: np.ndarray,
    observations: np.ndarray,
    uncertainty: np.ndarray,
    min_individuals: int,
    error_transform: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Per-gene Spearman rho between prediction error and uncertainty.

    Each correlation runs across the individuals of a single gene, so it
    measures whether that gene's uncertainty ranks its own donors by error.
    The error is the squared residual; since Spearman only uses ranks, this is
    equivalent to using the absolute residual.
    """
    pred, obs = transform_for_uncertainty_error(
        predictions,
        observations,
        error_transform,
    )
    unc = np.asarray(uncertainty, dtype=np.float64)

    n_genes = pred.shape[0]
    rho = np.full(n_genes, np.nan, dtype=np.float64)
    n_used = np.zeros(n_genes, dtype=np.int64)
    for row in range(n_genes):
        valid = (
            np.isfinite(pred[row])
            & np.isfinite(obs[row])
            & np.isfinite(unc[row])
            & (unc[row] >= 0.0)
        )
        n_used[row] = int(valid.sum())
        if n_used[row] < min_individuals:
            continue
        squared_error = np.square(pred[row, valid] - obs[row, valid])
        rho[row] = spearman_or_nan(unc[row, valid], squared_error)
    return rho, n_used


def pivot_per_gene_uncertainty_spearman(
    frames: list[pd.DataFrame],
) -> pd.DataFrame:
    """Turn per-kind per-gene rows into one row per gene, one column per kind."""
    long = pd.concat(frames, ignore_index=True)
    keys = ["model", "seed", "cell_type", "error_transform", *KEY_COLUMNS]
    # Pivoting each metric separately keeps the donor counts integral.
    wide = pd.concat(
        [
            long.pivot(index=keys, columns="uncertainty", values=metric)
            .rename(columns=lambda kind: f"{prefix}{kind}")
            for metric, prefix in (
                ("n_individuals", "n_individuals_"),
                ("spearman_error_vs_uncertainty", "spearman_error_vs_"),
            )
        ],
        axis=1,
    ).reset_index()
    wide.columns.name = None

    ordered = list(keys)
    for kind in UNCERTAINTY_KINDS:
        for column in (f"n_individuals_{kind}", f"spearman_error_vs_{kind}"):
            if column in wide.columns:
                ordered.append(column)
    ordered.extend(column for column in wide.columns if column not in ordered)
    return wide[ordered]


def format_seed_stat(mean: float, std: float, n_seeds: int, digits: int = 1) -> str:
    if n_seeds == 1:
        return f"{mean:.{digits}f}"
    return f"{mean:.{digits}f} ± {std:.{digits}f}"


def plot_pvalue_histograms(
    model: ModelLayout,
    histogram_counts: dict[str, dict[str, np.ndarray]],
    summary: pd.DataFrame,
    bin_edges: np.ndarray,
    threshold: float,
    output_path: Path,
    dpi: int,
) -> None:
    cell_types = sorted(histogram_counts, key=natural_sort_key)
    if not cell_types:
        logging.warning("No p-value histograms available for model %s.", model.name)
        return

    n_columns = min(4, max(1, math.ceil(math.sqrt(len(cell_types)))))
    n_rows = math.ceil(len(cell_types) / n_columns)
    fig, axes = plt.subplots(
        n_rows,
        n_columns,
        figsize=(4.2 * n_columns, 3.3 * n_rows),
        squeeze=False,
    )
    axes_flat = axes.ravel()
    centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0
    widths = np.diff(bin_edges)

    for axis, cell_type in zip(axes_flat, cell_types):
        seed_counts = histogram_counts[cell_type]
        arrays = np.vstack(
            [seed_counts[seed] for seed in sorted(seed_counts, key=natural_sort_key)]
        )
        mean = np.mean(arrays, axis=0)
        std = sample_std(arrays, axis=0)
        axis.bar(
            bin_edges[:-1],
            mean,
            width=widths,
            align="edge",
            color="#a9a9a9",
            edgecolor="white",
            linewidth=0.4,
        )
        if arrays.shape[0] > 1:
            axis.fill_between(
                centers,
                np.maximum(0.0, mean - std),
                mean + std,
                color="#595959",
                alpha=0.18,
                step="mid",
                linewidth=0,
            )

        rows = summary[
            (summary["model"] == model.name)
            & (summary["cell_type"] == cell_type)
        ]
        n_seeds = len(rows)
        tested_mean = float(rows["n_genes_tested"].mean())
        tested_std = (
            float(rows["n_genes_tested"].std(ddof=1))
            if n_seeds > 1 else 0.0
        )
        sig_mean = float(rows["n_significant"].mean())
        sig_std = (
            float(rows["n_significant"].std(ddof=1))
            if n_seeds > 1 else 0.0
        )
        uniform_height = tested_mean / (len(bin_edges) - 1)
        axis.axhline(
            uniform_height,
            color="#b2182b",
            linestyle="--",
            linewidth=1.0,
            alpha=0.8,
        )

        annotation = (
            f"tested: {format_seed_stat(tested_mean, tested_std, n_seeds)}\n"
            f"p ≤ {threshold:g}: "
            f"{format_seed_stat(sig_mean, sig_std, n_seeds)}\n"
            f"seeds: {n_seeds}"
        )
        axis.text(
            0.97,
            0.95,
            annotation,
            transform=axis.transAxes,
            ha="right",
            va="top",
            fontsize=8,
            bbox={
                "boxstyle": "round,pad=0.25",
                "facecolor": "white",
                "edgecolor": "none",
                "alpha": 0.82,
            },
        )
        axis.set_title(cell_type, fontsize=10)
        axis.set_xlim(0.0, 1.0)
        axis.set_xlabel("Pearson p-value")
        axis.set_ylabel("Genes (seed mean)")

    for axis in axes_flat[len(cell_types):]:
        axis.set_visible(False)

    fig.suptitle(
        f"{model.name}: Pearson correlations across OneK1K test individuals",
        fontsize=14,
        y=1.0,
    )
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    logging.info("Wrote %s", output_path)


def normalize_excel_header(value: object) -> str:
    """Normalize a workbook header for schema detection."""
    return re.sub(r"[^a-z0-9]+", "_", str(value).strip().casefold()).strip("_")


def canonical_cell_type_name(value: object) -> str:
    """Canonicalize paper/workbook cell-type labels for matching."""
    tokens = re.sub(
        r"[^a-z0-9]+",
        " ",
        str(value).strip().casefold().replace("_", " "),
    ).split()
    if tokens and tokens[-1] == "human":
        tokens.pop()
    if tokens[:3] == ["mucosal", "associated", "invariant"]:
        tokens.pop(1)
    return " ".join(tokens)


def load_scpredixcan_performance(
    path: Path,
    evaluation_cell_types: list[str],
) -> pd.DataFrame:
    """Load and align reported OneK1K performance from a scPrediXcan workbook."""
    if not path.is_file():
        raise ValueError(
            f"scPrediXcan performance path is not a file: {path}"
        )

    required = ("dataset", "cell_type", "m1")
    optional = ("cell_number", "pearson_r")
    tables: list[pd.DataFrame] = []
    workbook = pd.ExcelFile(path)
    for sheet_name in workbook.sheet_names:
        raw = pd.read_excel(
            workbook,
            sheet_name=sheet_name,
            header=None,
        )
        for header_row in range(min(20, len(raw))):
            headers = [
                normalize_excel_header(value)
                for value in raw.iloc[header_row].tolist()
            ]
            positions = {
                header: index
                for index, header in enumerate(headers)
                if header
            }
            if not set(required).issubset(positions):
                continue

            selected = [
                column for column in (*required, *optional)
                if column in positions
            ]
            table = raw.iloc[
                header_row + 1:,
                [positions[column] for column in selected],
            ].copy()
            table.columns = selected
            table["source_sheet"] = sheet_name
            tables.append(table)
            break

    if not tables:
        raise ValueError(
            f"{path} contains no worksheet with the required columns "
            f"{list(required)} in its first 20 rows."
        )

    performance = pd.concat(tables, ignore_index=True)
    dataset_key = (
        performance["dataset"]
        .astype("string")
        .str.strip()
        .str.casefold()
    )
    performance = performance.loc[dataset_key == "onek1k"].copy()
    if performance.empty:
        raise ValueError(f"{path} contains no rows for dataset OneK1K.")

    performance["m1"] = pd.to_numeric(
        performance["m1"],
        errors="coerce",
    )
    for column in optional:
        if column not in performance:
            performance[column] = np.nan
        performance[column] = pd.to_numeric(
            performance[column],
            errors="coerce",
        )
    performance = performance.loc[
        performance["cell_type"].notna()
        & performance["m1"].notna()
    ].copy()
    if performance.empty:
        raise ValueError(
            f"{path} has no OneK1K rows with both cell_type and numeric m1."
        )

    evaluation_by_key: dict[str, str] = {}
    for cell_type in evaluation_cell_types:
        key = canonical_cell_type_name(cell_type)
        if not key:
            raise ValueError(f"Cannot normalize cell type {cell_type!r}.")
        if key in evaluation_by_key:
            raise ValueError(
                "Evaluation cell types normalize to the same label: "
                f"{evaluation_by_key[key]!r} and {cell_type!r}."
            )
        evaluation_by_key[key] = cell_type

    performance["cell_type_key"] = performance["cell_type"].map(
        canonical_cell_type_name,
    )
    duplicate_keys = performance.loc[
        performance["cell_type_key"].duplicated(keep=False),
        "cell_type_key",
    ].unique()
    if len(duplicate_keys):
        raise ValueError(
            f"{path} contains duplicate OneK1K cell types after label "
            f"normalization: {sorted(duplicate_keys)}."
        )

    performance["reported_cell_type"] = performance["cell_type"].astype(str)
    performance["cell_type"] = performance["cell_type_key"].map(
        evaluation_by_key,
    )
    unmatched_reported = performance.loc[
        performance["cell_type"].isna(),
        "reported_cell_type",
    ].tolist()
    if unmatched_reported:
        logging.warning(
            "Ignoring %d reported scPrediXcan cell types absent from the "
            "current evaluation: %s",
            len(unmatched_reported),
            unmatched_reported,
        )
    performance = performance.loc[performance["cell_type"].notna()].copy()
    if performance.empty:
        raise ValueError(
            f"No OneK1K cell type in {path} matches the current evaluation."
        )

    matched = set(performance["cell_type"])
    missing_reported = [
        cell_type for cell_type in evaluation_cell_types
        if cell_type not in matched
    ]
    if missing_reported:
        logging.warning(
            "No reported scPrediXcan m1 is available for %d evaluated cell "
            "types: %s",
            len(missing_reported),
            missing_reported,
        )

    return performance[
        [
            "dataset",
            "cell_type",
            "reported_cell_type",
            "cell_number",
            "pearson_r",
            "m1",
            "source_sheet",
        ]
    ].reset_index(drop=True)


def make_model_summary(per_seed: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        "n_genes_tested",
        "n_significant",
        "significant_fraction",
        "n_common_individuals",
    ]
    if "m1" in per_seed.columns:
        metrics.extend(["pi0", "m1"])
    rows: list[dict[str, object]] = []
    for (model, cell_type), group in per_seed.groupby(
        ["model", "cell_type"],
        sort=False,
    ):
        row: dict[str, object] = {
            "model": model,
            "cell_type": cell_type,
            "n_seeds": len(group),
            "seeds": ",".join(
                sorted(group["seed"].astype(str), key=natural_sort_key),
            ),
        }
        for metric in metrics:
            values = group[metric].to_numpy(dtype=np.float64)
            row[f"{metric}_mean"] = float(np.nanmean(values))
            row[f"{metric}_std"] = float(sample_std(values, axis=0))
        rows.append(row)
    return pd.DataFrame(rows)


def plot_model_comparison(
    model_summary: pd.DataFrame,
    mean_column: str,
    std_column: str,
    xlabel: str,
    title: str,
    output_path: Path,
    dpi: int,
) -> None:
    models = sorted(model_summary["model"].unique(), key=natural_sort_key)
    cell_types = list(model_summary["cell_type"].unique())
    order_score = {
        cell_type: model_summary.loc[
            model_summary["cell_type"] == cell_type,
            mean_column,
        ].max()
        for cell_type in cell_types
    }
    cell_types.sort(
        key=lambda cell_type: (
            -float(order_score[cell_type]),
            natural_sort_key(cell_type),
        ),
    )

    n_models = len(models)
    group_height = 0.82
    bar_height = group_height / max(n_models, 1)
    y_base = np.arange(len(cell_types), dtype=np.float64)
    colors = plt.get_cmap("tab10")(np.linspace(0.0, 1.0, max(n_models, 2)))

    fig, axis = plt.subplots(
        figsize=(12.0, max(5.0, 0.43 * len(cell_types) + 2.2)),
    )
    for model_index, model in enumerate(models):
        offset = (
            model_index - (n_models - 1) / 2.0
        ) * bar_height
        subset = model_summary.set_index(["model", "cell_type"])
        means = np.asarray(
            [
                subset.loc[(model, cell_type), mean_column]
                if (model, cell_type) in subset.index else np.nan
                for cell_type in cell_types
            ],
            dtype=np.float64,
        )
        stds = np.asarray(
            [
                subset.loc[(model, cell_type), std_column]
                if (model, cell_type) in subset.index else np.nan
                for cell_type in cell_types
            ],
            dtype=np.float64,
        )
        valid = np.isfinite(means)
        valid_stds = stds[valid]
        xerr = (
            np.where(np.isfinite(valid_stds), valid_stds, 0.0)
            if np.isfinite(valid_stds).any()
            else None
        )
        axis.barh(
            y_base[valid] + offset,
            means[valid],
            height=bar_height * 0.88,
            xerr=xerr,
            color=colors[model_index],
            alpha=0.88,
            capsize=2,
            error_kw={"linewidth": 0.8},
            label=model,
        )

    axis.set_yticks(y_base)
    axis.set_yticklabels(cell_types)
    axis.invert_yaxis()
    axis.set_xlabel(xlabel)
    axis.set_title(title)
    axis.grid(axis="x", alpha=0.2, linewidth=0.7)
    axis.set_axisbelow(True)
    axis.legend(title="Model", loc="best")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    logging.info("Wrote %s", output_path)


def model_color_map(models: list[str]) -> dict[str, np.ndarray]:
    """Stable model -> color mapping shared across survival-curve plots."""
    colors = plt.get_cmap("tab10")(np.linspace(0.0, 1.0, max(len(models), 2)))
    return {model: colors[index] for index, model in enumerate(models)}


def plot_survival_curve_pooled(
    pooled_curves: dict[str, dict[str, np.ndarray]],
    thresholds: np.ndarray,
    output_path: Path,
    dpi: int,
    correlation_name: str = "Pearson r",
) -> None:
    """One line per model: mean +/- SD survival curve pooled over cell types."""
    models = sorted(pooled_curves, key=natural_sort_key)
    if not models:
        logging.warning("No survival curves available for the pooled comparison plot.")
        return
    colors = model_color_map(models)

    fig, axis = plt.subplots(figsize=(8.0, 5.5))
    for model in models:
        seed_curves = pooled_curves[model]
        arrays = np.vstack(
            [seed_curves[seed] for seed in sorted(seed_curves, key=natural_sort_key)],
        )
        mean = np.nanmean(arrays, axis=0)
        std = sample_std(arrays, axis=0)
        axis.plot(
            thresholds,
            mean,
            linewidth=1.8,
            color=colors[model],
            label=model,
        )
        if arrays.shape[0] > 1:
            axis.fill_between(
                thresholds,
                np.clip(mean - std, 0.0, 1.0),
                np.clip(mean + std, 0.0, 1.0),
                color=colors[model],
                alpha=0.18,
                linewidth=0,
            )

    axis.set_xlim(0.0, 1.0)
    axis.set_ylim(0.0, 1.0)
    axis.set_xlabel(f"Absolute {correlation_name} threshold")
    axis.set_ylabel("Fraction of tested genes with |r| \u2265 threshold")
    axis.set_title(
        f"Survival curve of absolute {correlation_name} (all cell types pooled, "
        "seed mean \u00b1 SD)",
    )
    axis.grid(alpha=0.2, linewidth=0.7)
    axis.set_axisbelow(True)
    axis.legend(title="Model", loc="best")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    logging.info("Wrote %s", output_path)


def plot_survival_curve_by_cell_type(
    curves: dict[str, dict[str, dict[str, np.ndarray]]],
    thresholds: np.ndarray,
    output_path: Path,
    dpi: int,
    correlation_name: str = "Pearson r",
) -> None:
    """Grid of subplots, one per cell type, with one line per model."""
    cell_types = sorted(curves, key=natural_sort_key)
    if not cell_types:
        logging.warning(
            "No survival curves available for the per-cell-type comparison plot.",
        )
        return
    models = sorted(
        {model for by_model in curves.values() for model in by_model},
        key=natural_sort_key,
    )
    colors = model_color_map(models)

    n_columns = min(4, max(1, math.ceil(math.sqrt(len(cell_types)))))
    n_rows = math.ceil(len(cell_types) / n_columns)
    fig, axes = plt.subplots(
        n_rows,
        n_columns,
        figsize=(4.2 * n_columns, 3.6 * n_rows),
        squeeze=False,
    )
    axes_flat = axes.ravel()

    for axis, cell_type in zip(axes_flat, cell_types):
        for model in models:
            seed_curves = curves[cell_type].get(model, {})
            if not seed_curves:
                continue
            arrays = np.vstack(
                [
                    seed_curves[seed]
                    for seed in sorted(seed_curves, key=natural_sort_key)
                ],
            )
            mean = np.nanmean(arrays, axis=0)
            std = sample_std(arrays, axis=0)
            axis.plot(
                thresholds,
                mean,
                linewidth=1.6,
                color=colors[model],
                label=model,
            )
            if arrays.shape[0] > 1:
                axis.fill_between(
                    thresholds,
                    np.clip(mean - std, 0.0, 1.0),
                    np.clip(mean + std, 0.0, 1.0),
                    color=colors[model],
                    alpha=0.15,
                    linewidth=0,
                )

        axis.set_title(cell_type, fontsize=10)
        axis.set_xlim(0.0, 1.0)
        axis.set_ylim(0.0, 1.0)
        axis.set_xlabel(f"|{correlation_name}| threshold")
        axis.set_ylabel("Fraction \u2265 threshold")
        axis.grid(alpha=0.18, linewidth=0.7)
        axis.set_axisbelow(True)

    for axis in axes_flat[len(cell_types):]:
        axis.set_visible(False)

    handles = [
        plt.Line2D([0], [0], color=colors[model], linewidth=1.8, label=model)
        for model in models
    ]
    fig.legend(
        handles=handles,
        title="Model",
        loc="upper center",
        ncol=min(len(models), 6),
        frameon=False,
        bbox_to_anchor=(0.5, 1.0),
    )
    fig.suptitle(
        f"Survival curves of absolute {correlation_name} by cell type "
        "(seed mean \u00b1 SD)",
        fontsize=14,
        y=1.03,
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.95))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    logging.info("Wrote %s", output_path)


def plot_sign_accuracy_pooled(
    pooled_curves: dict[str, dict[str, np.ndarray]],
    thresholds: np.ndarray,
    output_path: Path,
    dpi: int,
) -> None:
    """One line per model: mean +/- SD sign accuracy pooled over cell types."""
    models = sorted(pooled_curves, key=natural_sort_key)
    if not models:
        logging.warning("No sign-accuracy curves available for the pooled plot.")
        return
    colors = model_color_map(models)

    fig, axis = plt.subplots(figsize=(8.0, 5.5))
    for model in models:
        seed_curves = pooled_curves[model]
        arrays = np.vstack(
            [seed_curves[seed] for seed in sorted(seed_curves, key=natural_sort_key)],
        )
        mean = np.nanmean(arrays, axis=0)
        std = sample_std(arrays, axis=0)
        axis.plot(
            thresholds,
            mean,
            linewidth=1.8,
            color=colors[model],
            label=model,
        )
        if arrays.shape[0] > 1:
            axis.fill_between(
                thresholds,
                np.clip(mean - std, 0.0, 1.0),
                np.clip(mean + std, 0.0, 1.0),
                color=colors[model],
                alpha=0.18,
                linewidth=0,
            )

    axis.axhline(0.5, color="#555555", linestyle="--", linewidth=1.0, alpha=0.8)
    axis.set_xlim(0.0, 1.0)
    axis.set_ylim(0.0, 1.0)
    axis.set_xlabel("Absolute Pearson r threshold t")
    axis.set_ylabel(r"P(r > 0 | |r| $\geq$ t)")
    axis.set_title(
        "Sign accuracy of predicted-vs-true correlation (all cell types "
        "pooled, seed mean \u00b1 SD)",
    )
    axis.grid(alpha=0.2, linewidth=0.7)
    axis.set_axisbelow(True)
    axis.legend(title="Model", loc="best")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    logging.info("Wrote %s", output_path)


def plot_sign_accuracy_by_cell_type(
    curves: dict[str, dict[str, dict[str, np.ndarray]]],
    thresholds: np.ndarray,
    output_path: Path,
    dpi: int,
) -> None:
    """Grid of subplots, one per cell type, with one sign-accuracy line per model."""
    cell_types = sorted(curves, key=natural_sort_key)
    if not cell_types:
        logging.warning(
            "No sign-accuracy curves available for the per-cell-type plot.",
        )
        return
    models = sorted(
        {model for by_model in curves.values() for model in by_model},
        key=natural_sort_key,
    )
    colors = model_color_map(models)

    n_columns = min(4, max(1, math.ceil(math.sqrt(len(cell_types)))))
    n_rows = math.ceil(len(cell_types) / n_columns)
    fig, axes = plt.subplots(
        n_rows,
        n_columns,
        figsize=(4.2 * n_columns, 3.6 * n_rows),
        squeeze=False,
    )
    axes_flat = axes.ravel()

    for axis, cell_type in zip(axes_flat, cell_types):
        for model in models:
            seed_curves = curves[cell_type].get(model, {})
            if not seed_curves:
                continue
            arrays = np.vstack(
                [
                    seed_curves[seed]
                    for seed in sorted(seed_curves, key=natural_sort_key)
                ],
            )
            mean = np.nanmean(arrays, axis=0)
            std = sample_std(arrays, axis=0)
            axis.plot(
                thresholds,
                mean,
                linewidth=1.6,
                color=colors[model],
                label=model,
            )
            if arrays.shape[0] > 1:
                axis.fill_between(
                    thresholds,
                    np.clip(mean - std, 0.0, 1.0),
                    np.clip(mean + std, 0.0, 1.0),
                    color=colors[model],
                    alpha=0.15,
                    linewidth=0,
                )

        axis.axhline(0.5, color="#555555", linestyle="--", linewidth=0.9, alpha=0.8)
        axis.set_title(cell_type, fontsize=10)
        axis.set_xlim(0.0, 1.0)
        axis.set_ylim(0.0, 1.0)
        axis.set_xlabel("|Pearson r| threshold t")
        axis.set_ylabel(r"P(r > 0 | |r| $\geq$ t)")
        axis.grid(alpha=0.18, linewidth=0.7)
        axis.set_axisbelow(True)

    for axis in axes_flat[len(cell_types):]:
        axis.set_visible(False)

    handles = [
        plt.Line2D([0], [0], color=colors[model], linewidth=1.8, label=model)
        for model in models
    ]
    fig.legend(
        handles=handles,
        title="Model",
        loc="upper center",
        ncol=min(len(models), 6),
        frameon=False,
        bbox_to_anchor=(0.5, 1.0),
    )
    fig.suptitle(
        "Sign accuracy of predicted-vs-true correlation by cell type "
        "(seed mean \u00b1 SD)",
        fontsize=14,
        y=1.03,
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.95))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    logging.info("Wrote %s", output_path)


def plot_model_scalar_comparison(
    summary: pd.DataFrame,
    mean_column: str,
    std_column: str,
    xlabel: str,
    title: str,
    output_path: Path,
    dpi: int,
) -> None:
    """Horizontal bar chart with a single mean +/- SD value per model."""
    if summary.empty:
        logging.warning("No data available for %s.", output_path)
        return
    ordered = summary.sort_values(mean_column, ascending=False, kind="mergesort")
    models = ordered["model"].tolist()
    means = ordered[mean_column].to_numpy(dtype=np.float64)
    stds = ordered[std_column].to_numpy(dtype=np.float64)
    colors_by_model = model_color_map(sorted(models, key=natural_sort_key))
    colors = [colors_by_model[model] for model in models]

    fig, axis = plt.subplots(
        figsize=(9.0, max(2.5, 0.5 * len(models) + 1.5)),
    )
    y = np.arange(len(models), dtype=np.float64)
    xerr = np.where(np.isfinite(stds), stds, 0.0)
    axis.barh(
        y,
        means,
        height=0.6,
        xerr=xerr,
        color=colors,
        alpha=0.88,
        capsize=3,
        error_kw={"linewidth": 0.9},
    )
    axis.set_yticks(y)
    axis.set_yticklabels(models)
    axis.invert_yaxis()
    axis.set_xlabel(xlabel)
    axis.set_title(title)
    axis.grid(axis="x", alpha=0.2, linewidth=0.7)
    axis.set_axisbelow(True)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    logging.info("Wrote %s", output_path)


def plot_uncertainty_ablation_by_cell_type(
    summary: pd.DataFrame,
    output_path: Path,
    dpi: int,
    uncertainty_kind: str,
    individual_dir: Path | None = None,
) -> None:
    """One subplot per cell type: survival-curve AUC vs percent removed."""
    required = {"cell_type", "model", "percent_removed", "auc_mean", "auc_std"}
    if summary.empty or not required.issubset(summary.columns):
        logging.warning(
            "No uncertainty-ablation AUC values available for %s.",
            output_path,
        )
        return

    cell_types = sorted(
        summary["cell_type"].astype(str).unique(),
        key=natural_sort_key,
    )
    models = sorted(summary["model"].astype(str).unique(), key=natural_sort_key)
    colors = model_color_map(models)

    n_columns = min(4, max(1, math.ceil(math.sqrt(len(cell_types)))))
    n_rows = math.ceil(len(cell_types) / n_columns)
    fig, axes = plt.subplots(
        n_rows,
        n_columns,
        figsize=(4.4 * n_columns, 3.7 * n_rows),
        squeeze=False,
    )
    axes_flat = axes.ravel()

    for axis, cell_type in zip(axes_flat, cell_types):
        _plot_uncertainty_ablation_axis(
            axis,
            summary,
            cell_type,
            models,
            colors,
        )

    for axis in axes_flat[len(cell_types):]:
        axis.set_visible(False)

    handles = [
        plt.Line2D(
            [0],
            [0],
            color=colors[model],
            marker="o",
            linewidth=1.8,
            label=model,
        )
        for model in models
    ]
    fig.legend(
        handles=handles,
        title="Model",
        loc="upper center",
        ncol=min(len(models), 6),
        frameon=False,
        bbox_to_anchor=(0.5, 1.0),
    )
    fig.suptitle(
        "Survival-curve AUC after dropping the top X% most-uncertain "
        f"individuals ({uncertainty_kind}, ranked within each gene; "
        "seed mean \u00b1 SD)",
        fontsize=14,
        y=1.03,
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.95))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    logging.info("Wrote %s", output_path)

    if individual_dir is None:
        return
    individual_dir.mkdir(parents=True, exist_ok=True)
    for cell_type in cell_types:
        fig, axis = plt.subplots(figsize=(6.5, 4.2))
        _plot_uncertainty_ablation_axis(
            axis,
            summary,
            cell_type,
            models,
            colors,
        )
        axis.legend(title="Model", loc="best")
        axis.set_title(
            f"{cell_type}: survival-curve AUC vs high-uncertainty dropout "
            f"({uncertainty_kind})",
        )
        cell_path = individual_dir / f"{cell_type}.png"
        fig.tight_layout()
        fig.savefig(cell_path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        logging.info("Wrote %s", cell_path)


def _plot_uncertainty_ablation_axis(
    axis: plt.Axes,
    summary: pd.DataFrame,
    cell_type: str,
    models: list[str],
    colors: dict[str, np.ndarray],
) -> None:
    subset = summary[summary["cell_type"].astype(str) == cell_type]
    for model in models:
        model_rows = subset[subset["model"].astype(str) == model].sort_values(
            "percent_removed",
            kind="mergesort",
        )
        if model_rows.empty:
            continue
        x = model_rows["percent_removed"].to_numpy(dtype=np.float64)
        mean = model_rows["auc_mean"].to_numpy(dtype=np.float64)
        std = model_rows["auc_std"].to_numpy(dtype=np.float64)
        axis.plot(
            x,
            mean,
            marker="o",
            linewidth=1.6,
            markersize=4.0,
            color=colors[model],
            label=model,
        )
        n_seeds = (
            model_rows["n_seeds"].to_numpy(dtype=np.float64)
            if "n_seeds" in model_rows.columns
            else np.ones(len(model_rows), dtype=np.float64)
        )
        if np.any(n_seeds > 1) and np.isfinite(std).any():
            axis.fill_between(
                x,
                mean - np.where(np.isfinite(std), std, 0.0),
                mean + np.where(np.isfinite(std), std, 0.0),
                color=colors[model],
                alpha=0.15,
                linewidth=0,
            )

    axis.set_title(cell_type, fontsize=10)
    axis.set_xlabel("% individuals removed")
    axis.set_ylabel("Survival-curve AUC")
    axis.grid(alpha=0.18, linewidth=0.7)
    axis.set_axisbelow(True)


def plot_uncertainty_curves(
    model_name: str,
    curves: dict[str, dict[str, dict[str, np.ndarray]]],
    n_bins: int,
    scope: str,
    output_path: Path,
    dpi: int,
) -> None:
    cell_types = sorted(curves, key=natural_sort_key)
    if not cell_types:
        logging.warning(
            "No %s uncertainty curves available for model %s.",
            scope,
            model_name,
        )
        return

    n_columns = min(3, max(1, math.ceil(math.sqrt(len(cell_types)))))
    n_rows = math.ceil(len(cell_types) / n_columns)
    fig, axes = plt.subplots(
        n_rows,
        n_columns,
        figsize=(5.0 * n_columns, 3.8 * n_rows),
        squeeze=False,
    )
    axes_flat = axes.ravel()
    x = np.arange(1, n_bins + 1, dtype=np.float64)
    colors = {
        "totvar": "#3b4cc0",
        "aleatoric": "#1b9e77",
        "epistemic": "#d95f02",
    }

    for axis, cell_type in zip(axes_flat, cell_types):
        for kind in UNCERTAINTY_KINDS:
            seed_curves = curves[cell_type].get(kind, {})
            if not seed_curves:
                continue
            arrays = np.vstack(
                [
                    seed_curves[seed]
                    for seed in sorted(seed_curves, key=natural_sort_key)
                ],
            )
            mean = np.nanmean(arrays, axis=0)
            std = sample_std(arrays, axis=0)
            axis.plot(
                x,
                mean,
                marker="o",
                linewidth=1.7,
                markersize=3.5,
                color=colors[kind],
                label=kind,
            )
            if arrays.shape[0] > 1:
                axis.fill_between(
                    x,
                    mean - std,
                    mean + std,
                    color=colors[kind],
                    alpha=0.15,
                    linewidth=0,
                )

        if scope == "between":
            axis.set_ylabel(
                r"Correlation difficulty ($1-|\mathrm{Pearson}\ r|$)",
            )
            axis.set_ylim(0.0, 1.0)
        elif scope == "within":
            axis.axhline(
                1.0,
                color="#555555",
                linestyle="--",
                linewidth=0.9,
                alpha=0.8,
            )
            axis.set_ylabel("Relative squared error")
        else:
            raise ValueError(f"Unknown uncertainty scope: {scope}")

        axis.set_title(cell_type, fontsize=10)
        axis.set_xlabel("Predicted uncertainty quantile (low → high)")
        axis.set_xticks(x)
        axis.grid(alpha=0.18, linewidth=0.7)
        axis.set_axisbelow(True)

    for axis in axes_flat[len(cell_types):]:
        axis.set_visible(False)

    handles = [
        plt.Line2D(
            [0],
            [0],
            color=colors[kind],
            marker="o",
            linewidth=1.7,
            label=kind,
        )
        for kind in UNCERTAINTY_KINDS
    ]
    fig.legend(
        handles=handles,
        loc="upper center",
        ncol=len(UNCERTAINTY_KINDS),
        frameon=False,
        bbox_to_anchor=(0.5, 1.0),
    )
    if scope == "between":
        title = (
            f"{model_name}: can average uncertainty rank gene-level "
            "correlation difficulty?"
        )
    else:
        title = (
            f"{model_name}: can within-gene uncertainty rank "
            "individual-level errors?"
        )
    fig.suptitle(title, fontsize=14, y=1.025)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.97))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    logging.info("Wrote %s", output_path)


def save_per_gene_metrics(
    output_dir: Path,
    model: str,
    seed: str,
    cell_type: str,
    index: pd.MultiIndex,
    n: np.ndarray,
    pearson_r: np.ndarray,
    pearson_pvalue: np.ndarray,
    spearman_rho: np.ndarray,
) -> None:
    output_path = (
        output_dir
        / "tables"
        / "per_gene"
        / model
        / seed
        / f"{cell_type}.csv.gz"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame = index.to_frame(index=False)
    frame["n_individuals"] = n
    frame["pearson_r"] = pearson_r
    frame["pearson_pvalue"] = pearson_pvalue
    frame["spearman_rho"] = spearman_rho
    frame.to_csv(output_path, index=False, compression="gzip")
    logging.debug("Wrote %s", output_path)


def evaluate_uncertainty_kind(
    model: ModelLayout,
    seed: SeedLayout,
    cell_type: str,
    kind: str,
    pred: pd.DataFrame,
    gt: pd.DataFrame,
    test_ids: set[str],
    min_individuals: int,
    n_bins: int,
    error_transform: str,
) -> tuple[
    np.ndarray,
    dict[str, object],
    np.ndarray,
    dict[str, object],
    pd.DataFrame,
]:
    uncertainty = load_expression_table(
        seed.uncertainties[kind][cell_type],
        test_ids,
    )
    pred_u, gt_u, uncertainty = align_expression_tables(pred, gt, uncertainty)
    pred_values = pred_u.to_numpy(dtype=np.float64)
    gt_values = gt_u.to_numpy(dtype=np.float64)
    uncertainty_values = uncertainty.to_numpy(dtype=np.float64)
    uncertainty_values[uncertainty_values < 0.0] = np.nan

    pearson_r, _, n = rowwise_pearson(
        np.where(np.isfinite(uncertainty_values), pred_values, np.nan),
        np.where(np.isfinite(uncertainty_values), gt_values, np.nan),
        min_individuals,
    )
    valid_pairs = (
        np.isfinite(pred_values)
        & np.isfinite(gt_values)
        & np.isfinite(uncertainty_values)
    )
    uncertainty_sum = np.where(
        valid_pairs,
        uncertainty_values,
        0.0,
    ).sum(axis=1)
    uncertainty_n = valid_pairs.sum(axis=1)
    mean_uncertainty = np.full(pred_values.shape[0], np.nan, dtype=np.float64)
    has_uncertainty = uncertainty_n > 0
    mean_uncertainty[has_uncertainty] = (
        uncertainty_sum[has_uncertainty]
        / uncertainty_n[has_uncertainty]
    )
    # scPrediXcan uses two-sided correlation p-values because Enformer can
    # reverse correlation signs. Keep the same sign-invariant notion here:
    # |r|=1 is easiest and |r|=0 is hardest.
    difficulty = 1.0 - np.abs(pearson_r)
    between_curve, n_between_genes = equal_count_curve(
        mean_uncertainty,
        difficulty,
        n_bins,
    )
    between_rho = spearman_or_nan(mean_uncertainty, difficulty)
    between_row: dict[str, object] = {
        "model": model.name,
        "seed": seed.name,
        "cell_type": cell_type,
        "uncertainty": kind,
        "n_genes": n_between_genes,
        "spearman_mean_uncertainty_vs_difficulty": between_rho,
        "mean_n_individuals": float(np.mean(n[np.isfinite(pearson_r)]))
        if np.isfinite(pearson_r).any() else float("nan"),
    }
    for bin_index, value in enumerate(between_curve, start=1):
        between_row[f"bin_{bin_index}"] = float(value)

    within_curve, n_within_genes = within_gene_uncertainty_curve(
        pred_values,
        gt_values,
        uncertainty_values,
        n_bins,
        min_individuals,
        error_transform,
    )
    within_row: dict[str, object] = {
        "model": model.name,
        "seed": seed.name,
        "cell_type": cell_type,
        "uncertainty": kind,
        "error_transform": error_transform,
        "n_genes": n_within_genes,
    }
    for bin_index, value in enumerate(within_curve, start=1):
        within_row[f"bin_{bin_index}"] = float(value)

    per_gene_rho, per_gene_n = per_gene_error_uncertainty_spearman(
        pred_values,
        gt_values,
        uncertainty_values,
        min_individuals,
        error_transform,
    )
    per_gene = pred_u.index.to_frame(index=False)
    per_gene["model"] = model.name
    per_gene["seed"] = seed.name
    per_gene["cell_type"] = cell_type
    per_gene["uncertainty"] = kind
    per_gene["error_transform"] = error_transform
    per_gene["n_individuals"] = per_gene_n
    per_gene["spearman_error_vs_uncertainty"] = per_gene_rho
    # Genes without a defined correlation (too few donors, or no spread in
    # error or uncertainty) carry no signal for this table.
    per_gene = per_gene[np.isfinite(per_gene_rho)].reset_index(drop=True)

    return between_curve, between_row, within_curve, within_row, per_gene


def evaluate_uncertainty_ablation(
    model: ModelLayout,
    seed: SeedLayout,
    cell_type: str,
    kind: str,
    pred: pd.DataFrame,
    gt: pd.DataFrame,
    test_ids: set[str],
    min_individuals: int,
    percents: tuple[float, ...],
    survival_thresholds: np.ndarray,
) -> list[dict[str, object]]:
    """Recompute survival-curve AUC after dropping high-uncertainty donors."""
    uncertainty_path = seed.uncertainties[kind].get(cell_type)
    if uncertainty_path is None:
        logging.warning(
            "%s/%s/%s: no %s uncertainty file; skipping ablation.",
            model.name,
            seed.name,
            cell_type,
            kind,
        )
        return []

    uncertainty = load_expression_table(uncertainty_path, test_ids)
    pred_a, gt_a, uncertainty = align_expression_tables(pred, gt, uncertainty)
    pred_values = pred_a.to_numpy(dtype=np.float64)
    gt_values = gt_a.to_numpy(dtype=np.float64)
    uncertainty_values = uncertainty.to_numpy(dtype=np.float64)
    uncertainty_values[uncertainty_values < 0.0] = np.nan

    rows: list[dict[str, object]] = []
    for percent in percents:
        pred_m, gt_m = drop_top_uncertain_individuals(
            pred_values,
            gt_values,
            uncertainty_values,
            percent / 100.0,
        )
        pearson_r, _, n = rowwise_pearson(pred_m, gt_m, min_individuals)
        abs_r = np.abs(pearson_r[np.isfinite(pearson_r)])
        curve, n_genes = compute_survival_curve(abs_r, survival_thresholds)
        finite_n = n[np.isfinite(pearson_r)]
        rows.append(
            {
                "model": model.name,
                "seed": seed.name,
                "cell_type": cell_type,
                "uncertainty": kind,
                "percent_removed": float(percent),
                "n_genes": n_genes,
                "mean_n_individuals": (
                    float(np.mean(finite_n)) if len(finite_n) else float("nan")
                ),
                "auc": survival_curve_auc(curve, survival_thresholds),
            },
        )
    return rows


def warn_about_cell_type_coverage(
    ground_truth_files: dict[str, Path],
    models: tuple[ModelLayout, ...],
) -> list[str]:
    ground_truth_cell_types = set(ground_truth_files)
    any_predictions: set[str] = set()
    for model in models:
        model_cell_types = set().union(
            *(set(seed.predictions) for seed in model.seeds)
        )
        any_predictions.update(model_cell_types)
        missing = sorted(
            ground_truth_cell_types - model_cell_types,
            key=natural_sort_key,
        )
        extra = sorted(
            model_cell_types - ground_truth_cell_types,
            key=natural_sort_key,
        )
        if missing:
            logging.warning(
                "Model %s has no predictions for %d ground-truth cell types: %s",
                model.name,
                len(missing),
                missing,
            )
        if extra:
            logging.warning(
                "Model %s has %d cell types absent from ground truth; they will "
                "be ignored: %s",
                model.name,
                len(extra),
                extra,
            )

    shared = sorted(
        ground_truth_cell_types & any_predictions,
        key=natural_sort_key,
    )
    if not shared:
        raise ValueError(
            "No cell-type CSV stem is shared by ground truth and predictions."
        )
    return shared


def main() -> int:
    args = parse_args()
    setup_logging(args.verbose)

    test_ids = {str(individual) for individual in INDIVIDUAL_IDS}
    if len(test_ids) != len(INDIVIDUAL_IDS):
        raise ValueError("INDIVIDUAL_IDS contains duplicate IDs.")

    ground_truth_files = discover_ground_truth(args.ground_truth)
    models = discover_models(args.predictions)
    cell_types = warn_about_cell_type_coverage(ground_truth_files, models)
    if args.uncertainty_only:
        if not any(model.probabilistic for model in models):
            raise ValueError(
                "--uncertainty-only was requested, but none of the discovered "
                "models has uncertainty maps, so there is nothing to evaluate."
            )
        logging.info(
            "--uncertainty-only: reporting the uncertainty analysis alone; "
            "p-value, m1, survival-curve and sign-accuracy outputs are "
            "skipped."
        )
        skipped = [
            flag for flag, requested in (
                ("--scpredixcan-performance", args.scpredixcan_performance),
                ("--compute-m1", args.compute_m1),
                ("--save-per-gene", args.save_per_gene),
            ) if requested
        ]
        if skipped:
            logging.warning(
                "--uncertainty-only ignores %s because it only affects the "
                "skipped outputs.",
                " and ".join(skipped),
            )
        args.scpredixcan_performance = None
        args.compute_m1 = False
        args.save_per_gene = False

    scpredixcan_performance: pd.DataFrame | None = None
    if args.scpredixcan_performance is not None:
        scpredixcan_performance = load_scpredixcan_performance(
            args.scpredixcan_performance,
            cell_types,
        )
        logging.info(
            "Loaded reported scPrediXcan metrics for %d OneK1K cell types "
            "from %s.",
            len(scpredixcan_performance),
            args.scpredixcan_performance,
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)

    logging.info(
        "Evaluating %d models, %d shared cell types, and %d requested test IDs.",
        len(models),
        len(cell_types),
        len(test_ids),
    )
    for model in models:
        logging.info(
            "Model %s: %d seeds (%s).",
            model.name,
            len(model.seeds),
            "probabilistic" if model.probabilistic else "deterministic",
        )

    bin_edges = np.linspace(0.0, 1.0, args.pvalue_bins + 1)
    histogram_counts: dict[
        str,
        dict[str, dict[str, np.ndarray]],
    ] = defaultdict(lambda: defaultdict(dict))
    per_seed_rows: list[dict[str, object]] = []

    survival_thresholds = np.linspace(0.0, 1.0, args.survival_thresholds)
    survival_curves_by_cell_type: dict[
        str,
        dict[str, dict[str, np.ndarray]],
    ] = defaultdict(lambda: defaultdict(dict))
    survival_rows: list[dict[str, object]] = []
    pooled_abs_pearson_r: dict[str, dict[str, list[np.ndarray]]] = defaultdict(
        lambda: defaultdict(list),
    )
    spearman_survival_curves_by_cell_type: dict[
        str,
        dict[str, dict[str, np.ndarray]],
    ] = defaultdict(lambda: defaultdict(dict))
    spearman_survival_rows: list[dict[str, object]] = []
    pooled_abs_spearman_rho: dict[str, dict[str, list[np.ndarray]]] = defaultdict(
        lambda: defaultdict(list),
    )

    sign_accuracy_curves_by_cell_type: dict[
        str,
        dict[str, dict[str, np.ndarray]],
    ] = defaultdict(lambda: defaultdict(dict))
    sign_accuracy_rows: list[dict[str, object]] = []
    pooled_pearson_r: dict[str, dict[str, list[np.ndarray]]] = defaultdict(
        lambda: defaultdict(list),
    )

    between_curves: dict[
        str,
        dict[str, dict[str, dict[str, np.ndarray]]],
    ] = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
    within_curves: dict[
        str,
        dict[str, dict[str, dict[str, np.ndarray]]],
    ] = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
    between_rows: list[dict[str, object]] = []
    within_rows: list[dict[str, object]] = []
    per_gene_uncertainty_frames: list[pd.DataFrame] = []
    ablation_rows: list[dict[str, object]] = []

    if args.uncertainty_ablation_percents is not None:
        deterministic_names = [
            model.name for model in models if not model.probabilistic
        ]
        if deterministic_names:
            logging.info(
                "Uncertainty ablation will skip deterministic models: %s",
                ", ".join(deterministic_names),
            )
        if not any(model.probabilistic for model in models):
            logging.warning(
                "--uncertainty-ablation-percents was requested, but no "
                "probabilistic model layout was found."
            )
        else:
            logging.info(
                "Uncertainty ablation: drop top %s %% of individuals by %s "
                "(ranked within each gene) and recompute survival-curve AUC.",
                ", ".join(
                    f"{percent:g}"
                    for percent in args.uncertainty_ablation_percents
                ),
                args.ablation_uncertainty,
            )

    for cell_type in cell_types:
        logging.info("Cell type: %s", cell_type)
        ground_truth = load_expression_table(
            ground_truth_files[cell_type],
            test_ids,
            require_all_test_ids=False,
        )

        for model in models:
            if args.uncertainty_only and not model.probabilistic:
                continue
            for seed in model.seeds:
                prediction_path = seed.predictions.get(cell_type)
                if prediction_path is None:
                    continue

                predictions = load_expression_table(prediction_path, test_ids)
                pred, gt = align_expression_tables(predictions, ground_truth)

                if args.analyze_uncertainty and model.probabilistic:
                    for kind in UNCERTAINTY_KINDS:
                        (
                            between_curve,
                            between_row,
                            within_curve,
                            within_row,
                            per_gene_uncertainty,
                        ) = evaluate_uncertainty_kind(
                            model=model,
                            seed=seed,
                            cell_type=cell_type,
                            kind=kind,
                            pred=pred,
                            gt=gt,
                            test_ids=test_ids,
                            min_individuals=args.min_individuals,
                            n_bins=args.uncertainty_bins,
                            error_transform=args.uncertainty_error_transform,
                        )
                        between_curves[model.name][cell_type][kind][
                            seed.name
                        ] = between_curve
                        within_curves[model.name][cell_type][kind][
                            seed.name
                        ] = within_curve
                        between_rows.append(between_row)
                        within_rows.append(within_row)
                        if not per_gene_uncertainty.empty:
                            per_gene_uncertainty_frames.append(
                                per_gene_uncertainty,
                            )

                if (
                    args.uncertainty_ablation_percents is not None
                    and model.probabilistic
                ):
                    ablation_rows.extend(
                        evaluate_uncertainty_ablation(
                            model=model,
                            seed=seed,
                            cell_type=cell_type,
                            kind=args.ablation_uncertainty,
                            pred=pred,
                            gt=gt,
                            test_ids=test_ids,
                            min_individuals=args.min_individuals,
                            percents=args.uncertainty_ablation_percents,
                            survival_thresholds=survival_thresholds,
                        )
                    )

                if args.uncertainty_only:
                    logging.info(
                        "  %s/%s: uncertainty analysed, genes aligned=%d, "
                        "individuals=%d",
                        model.name,
                        seed.name,
                        len(pred),
                        len(pred.columns),
                    )
                    continue

                pred_values = pred.to_numpy(dtype=np.float64)
                gt_values = gt.to_numpy(dtype=np.float64)
                pearson_r, pearson_pvalue, n = rowwise_pearson(
                    pred_values,
                    gt_values,
                    args.min_individuals,
                )
                spearman_rho, _ = rowwise_spearman(
                    pred_values,
                    gt_values,
                    args.min_individuals,
                )
                valid_pvalues = pearson_pvalue[np.isfinite(pearson_pvalue)]
                n_tested = int(len(valid_pvalues))
                if n_tested == 0:
                    logging.warning(
                        "%s/%s/%s: no genes have a defined Pearson p-value.",
                        model.name,
                        seed.name,
                        cell_type,
                    )
                    pi0 = float("nan")
                    m1 = float("nan")
                    pi0_boundary_case = False
                elif args.compute_m1:
                    pi0 = estimate_pi0(
                        valid_pvalues,
                        method=args.pi0_method,
                        smooth_df=args.smooth_df,
                    )
                    m1 = float((1.0 - pi0) * n_tested)
                    pi0_boundary_case = bool(
                        np.max(valid_pvalues) < 0.95 or pi0 <= 0.0
                    )
                    if pi0_boundary_case:
                        logging.warning(
                            "%s/%s/%s: the p-value distribution does not "
                            "support qvalue's usual upper-tail check "
                            "(max p < 0.95 or pi0=0). Retaining the modified "
                            "Storey boundary estimate; see "
                            "pi0_boundary_case in per_seed_metrics.csv.",
                            model.name,
                            seed.name,
                            cell_type,
                        )
                else:
                    pi0 = float("nan")
                    m1 = float("nan")
                    pi0_boundary_case = False

                n_significant = int(
                    np.sum(valid_pvalues <= args.pvalue_threshold)
                )
                histogram_counts[model.name][cell_type][seed.name] = (
                    np.histogram(valid_pvalues, bins=bin_edges)[0]
                )

                # Same "tested" gene set as m1/p-values (np.isfinite(pearson_r)
                # coincides with np.isfinite(pearson_pvalue); see rowwise_pearson).
                abs_pearson_r_tested = np.abs(pearson_r[np.isfinite(pearson_r)])
                survival_curve, n_survival_genes = compute_survival_curve(
                    abs_pearson_r_tested,
                    survival_thresholds,
                )
                survival_curves_by_cell_type[cell_type][model.name][seed.name] = (
                    survival_curve
                )
                survival_row: dict[str, object] = {
                    "model": model.name,
                    "seed": seed.name,
                    "cell_type": cell_type,
                    "n_genes": n_survival_genes,
                    "auc": survival_curve_auc(survival_curve, survival_thresholds),
                }
                for threshold, value in zip(survival_thresholds, survival_curve):
                    survival_row[f"threshold_{threshold:.3f}"] = float(value)
                survival_rows.append(survival_row)
                pooled_abs_pearson_r[model.name][seed.name].append(
                    abs_pearson_r_tested,
                )

                abs_spearman_rho_tested = np.abs(
                    spearman_rho[np.isfinite(spearman_rho)]
                )
                spearman_curve, n_spearman_genes = compute_survival_curve(
                    abs_spearman_rho_tested,
                    survival_thresholds,
                )
                spearman_survival_curves_by_cell_type[cell_type][model.name][
                    seed.name
                ] = spearman_curve
                spearman_row: dict[str, object] = {
                    "model": model.name,
                    "seed": seed.name,
                    "cell_type": cell_type,
                    "n_genes": n_spearman_genes,
                    "auc": survival_curve_auc(spearman_curve, survival_thresholds),
                }
                for threshold, value in zip(survival_thresholds, spearman_curve):
                    spearman_row[f"threshold_{threshold:.3f}"] = float(value)
                spearman_survival_rows.append(spearman_row)
                pooled_abs_spearman_rho[model.name][seed.name].append(
                    abs_spearman_rho_tested,
                )

                pearson_r_tested = pearson_r[np.isfinite(pearson_r)]
                sign_accuracy_curve, _ = compute_sign_accuracy_curve(
                    pearson_r_tested,
                    survival_thresholds,
                )
                sign_accuracy_curves_by_cell_type[cell_type][model.name][
                    seed.name
                ] = sign_accuracy_curve
                sign_accuracy_row: dict[str, object] = {
                    "model": model.name,
                    "seed": seed.name,
                    "cell_type": cell_type,
                    "n_genes": n_survival_genes,
                }
                for threshold, value in zip(
                    survival_thresholds,
                    sign_accuracy_curve,
                ):
                    sign_accuracy_row[f"threshold_{threshold:.3f}"] = float(value)
                sign_accuracy_rows.append(sign_accuracy_row)
                pooled_pearson_r[model.name][seed.name].append(pearson_r_tested)

                finite_n = n[np.isfinite(pearson_pvalue)]
                per_seed_row: dict[str, object] = {
                        "model": model.name,
                        "seed": seed.name,
                        "cell_type": cell_type,
                        "probabilistic": model.probabilistic,
                        "n_aligned_gene_keys": len(pred),
                        "n_genes_tested": n_tested,
                        "n_significant": n_significant,
                        "significant_fraction": (
                            n_significant / n_tested
                            if n_tested else float("nan")
                        ),
                        "pvalue_threshold": args.pvalue_threshold,
                        "n_common_individuals": len(pred.columns),
                        "min_finite_individuals_per_tested_gene": (
                            int(finite_n.min())
                            if len(finite_n) else 0
                        ),
                        "median_finite_individuals_per_tested_gene": (
                            float(np.median(finite_n))
                            if len(finite_n) else float("nan")
                        ),
                    }
                if args.compute_m1:
                    per_seed_row.update(
                        {
                            "pi0": pi0,
                            "m1": m1,
                            "pi0_method": (
                                "storey_"
                                f"{args.pi0_method}_closed_form_modified_boundary"
                            ),
                            "pi0_boundary_case": pi0_boundary_case,
                        }
                    )
                per_seed_rows.append(per_seed_row)
                logging.info(
                    "  %s/%s: tested=%d, p<=%g=%d, genes aligned=%d, "
                    "individuals=%d",
                    model.name,
                    seed.name,
                    n_tested,
                    args.pvalue_threshold,
                    n_significant,
                    len(pred),
                    len(pred.columns),
                )
                if args.compute_m1:
                    logging.info(
                        "  %s/%s: m1=%.1f (pi0=%.4f)",
                        model.name,
                        seed.name,
                        m1,
                        pi0,
                    )

                if args.save_per_gene:
                    save_per_gene_metrics(
                        args.output_dir,
                        model.name,
                        seed.name,
                        cell_type,
                        pred.index,
                        n,
                        pearson_r,
                        pearson_pvalue,
                        spearman_rho,
                    )

    if args.uncertainty_only and not between_rows:
        raise ValueError(
            "No model/seed/cell-type uncertainty evaluation was completed."
        )

    tables_dir = args.output_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    comparisons_dir = args.output_dir / "comparisons"

    if not args.uncertainty_only:
        report_core_outputs(
            args=args,
            models=models,
            per_seed_rows=per_seed_rows,
            histogram_counts=histogram_counts,
            bin_edges=bin_edges,
            scpredixcan_performance=scpredixcan_performance,
            survival_thresholds=survival_thresholds,
            survival_rows=survival_rows,
            survival_curves_by_cell_type=survival_curves_by_cell_type,
            pooled_abs_pearson_r=pooled_abs_pearson_r,
            spearman_survival_rows=spearman_survival_rows,
            spearman_survival_curves_by_cell_type=(
                spearman_survival_curves_by_cell_type
            ),
            pooled_abs_spearman_rho=pooled_abs_spearman_rho,
            sign_accuracy_rows=sign_accuracy_rows,
            sign_accuracy_curves_by_cell_type=sign_accuracy_curves_by_cell_type,
            pooled_pearson_r=pooled_pearson_r,
            tables_dir=tables_dir,
            comparisons_dir=comparisons_dir,
        )

    if args.uncertainty_ablation_percents is not None:
        report_uncertainty_ablation(
            args=args,
            ablation_rows=ablation_rows,
            tables_dir=tables_dir,
            comparisons_dir=comparisons_dir,
        )

    if args.analyze_uncertainty:
        report_uncertainty_outputs(
            args=args,
            models=models,
            between_rows=between_rows,
            within_rows=within_rows,
            per_gene_uncertainty_frames=per_gene_uncertainty_frames,
            between_curves=between_curves,
            within_curves=within_curves,
            tables_dir=tables_dir,
        )

    logging.info("Evaluation complete. Outputs are in %s", args.output_dir)
    return 0


def report_core_outputs(
    *,
    args: argparse.Namespace,
    models: tuple[ModelLayout, ...],
    per_seed_rows: list[dict[str, object]],
    histogram_counts: dict[str, dict[str, dict[str, np.ndarray]]],
    bin_edges: np.ndarray,
    scpredixcan_performance: pd.DataFrame | None,
    survival_thresholds: np.ndarray,
    survival_rows: list[dict[str, object]],
    survival_curves_by_cell_type: dict[str, dict[str, dict[str, np.ndarray]]],
    pooled_abs_pearson_r: dict[str, dict[str, list[np.ndarray]]],
    spearman_survival_rows: list[dict[str, object]],
    spearman_survival_curves_by_cell_type: dict[
        str,
        dict[str, dict[str, np.ndarray]],
    ],
    pooled_abs_spearman_rho: dict[str, dict[str, list[np.ndarray]]],
    sign_accuracy_rows: list[dict[str, object]],
    sign_accuracy_curves_by_cell_type: dict[
        str,
        dict[str, dict[str, np.ndarray]],
    ],
    pooled_pearson_r: dict[str, dict[str, list[np.ndarray]]],
    tables_dir: Path,
    comparisons_dir: Path,
) -> None:
    """Write the p-value, m1, survival-curve and sign-accuracy outputs."""
    per_seed = pd.DataFrame(per_seed_rows)
    if per_seed.empty:
        raise ValueError("No model/seed/cell-type evaluation was completed.")

    per_seed_path = tables_dir / "per_seed_metrics.csv"
    per_seed.to_csv(per_seed_path, index=False)
    logging.info("Wrote %s", per_seed_path)

    evaluated_model_summary = make_model_summary(per_seed)
    model_summary = evaluated_model_summary.copy()
    model_summary["source"] = "evaluated_predictions"
    model_summary["reported_cell_number"] = np.nan
    model_summary["reported_pearson_r"] = np.nan

    if scpredixcan_performance is not None:
        scpredixcan_path = tables_dir / "scpredixcan_performance.csv"
        scpredixcan_performance.to_csv(scpredixcan_path, index=False)
        logging.info("Wrote %s", scpredixcan_path)

        reported_rows: list[dict[str, object]] = []
        for reported in scpredixcan_performance.itertuples(index=False):
            row = {
                column: float("nan")
                for column in model_summary.columns
            }
            row.update(
                {
                    "model": "scPrediXcan (reported)",
                    "cell_type": reported.cell_type,
                    "n_seeds": 0,
                    "seeds": "",
                    "m1_mean": float(reported.m1),
                    # The workbook contains one reported value rather than
                    # seed replicates, so its seed SD is intentionally missing.
                    "m1_std": float("nan"),
                    "source": "reported_scpredixcan",
                    "reported_cell_number": reported.cell_number,
                    "reported_pearson_r": reported.pearson_r,
                },
            )
            reported_rows.append(row)
        model_summary = pd.concat(
            [model_summary, pd.DataFrame(reported_rows)],
            ignore_index=True,
        )

    model_summary_path = tables_dir / "model_summary.csv"
    model_summary.to_csv(model_summary_path, index=False)
    logging.info("Wrote %s", model_summary_path)

    for model in models:
        plot_pvalue_histograms(
            model=model,
            histogram_counts=histogram_counts.get(model.name, {}),
            summary=per_seed,
            bin_edges=bin_edges,
            threshold=args.pvalue_threshold,
            output_path=(
                args.output_dir
                / "per_model"
                / model.name
                / "pearson_pvalues.png"
            ),
            dpi=args.dpi,
        )

    plot_model_comparison(
        model_summary=evaluated_model_summary,
        mean_column="n_significant_mean",
        std_column="n_significant_std",
        xlabel=f"Genes with Pearson p ≤ {args.pvalue_threshold:g} (seed mean ± SD)",
        title="Nominally significant expression predictions by cell type",
        output_path=comparisons_dir / "significant_genes.png",
        dpi=args.dpi,
    )
    if args.compute_m1:
        m1_xlabel = (
            r"Estimated non-null genes, m1 = (1 − $\hat{\pi}_0$)m "
            "(evaluated models: seed mean ± SD)"
        )
        if scpredixcan_performance is not None:
            m1_xlabel += "; scPrediXcan: reported value"
        plot_model_comparison(
            model_summary=model_summary,
            mean_column="m1_mean",
            std_column="m1_std",
            xlabel=m1_xlabel,
            title="q-value-framework m1 comparison by cell type",
            output_path=comparisons_dir / "m1.png",
            dpi=args.dpi,
        )

    survival_by_cell_type_path = tables_dir / "survival_curve_by_cell_type.csv"
    pd.DataFrame(survival_rows).to_csv(survival_by_cell_type_path, index=False)
    logging.info("Wrote %s", survival_by_cell_type_path)

    pooled_survival_curves: dict[str, dict[str, np.ndarray]] = defaultdict(dict)
    pooled_survival_rows: list[dict[str, object]] = []
    for model_name, by_seed in pooled_abs_pearson_r.items():
        for seed_name, chunks in by_seed.items():
            combined = (
                np.concatenate(chunks) if chunks else np.array([], dtype=np.float64)
            )
            pooled_curve, n_pooled_genes = compute_survival_curve(
                combined,
                survival_thresholds,
            )
            pooled_survival_curves[model_name][seed_name] = pooled_curve
            pooled_row: dict[str, object] = {
                "model": model_name,
                "seed": seed_name,
                "n_genes": n_pooled_genes,
                "auc": survival_curve_auc(pooled_curve, survival_thresholds),
            }
            for threshold, value in zip(survival_thresholds, pooled_curve):
                pooled_row[f"threshold_{threshold:.3f}"] = float(value)
            pooled_survival_rows.append(pooled_row)

    survival_pooled_path = tables_dir / "survival_curve_pooled.csv"
    pd.DataFrame(pooled_survival_rows).to_csv(survival_pooled_path, index=False)
    logging.info("Wrote %s", survival_pooled_path)

    plot_survival_curve_pooled(
        pooled_curves=pooled_survival_curves,
        thresholds=survival_thresholds,
        output_path=comparisons_dir / "survival_curve.png",
        dpi=args.dpi,
    )
    plot_survival_curve_by_cell_type(
        curves=survival_curves_by_cell_type,
        thresholds=survival_thresholds,
        output_path=comparisons_dir / "survival_curve_by_cell_type.png",
        dpi=args.dpi,
    )

    survival_auc_by_cell_type = summarize_metric_over_seeds(
        pd.DataFrame(survival_rows),
        ["model", "cell_type"],
        "auc",
    )
    survival_auc_pooled = summarize_metric_over_seeds(
        pd.DataFrame(pooled_survival_rows),
        ["model"],
        "auc",
    )
    survival_auc_by_cell_type_path = (
        tables_dir / "survival_curve_auc_by_cell_type.csv"
    )
    survival_auc_pooled_path = tables_dir / "survival_curve_auc_pooled.csv"
    survival_auc_by_cell_type.to_csv(survival_auc_by_cell_type_path, index=False)
    survival_auc_pooled.to_csv(survival_auc_pooled_path, index=False)
    logging.info("Wrote %s", survival_auc_by_cell_type_path)
    logging.info("Wrote %s", survival_auc_pooled_path)

    auc_xlabel = (
        "AUC of absolute Pearson correlation survival curve "
        "(seed mean \u00b1 SD)"
    )
    plot_model_comparison(
        model_summary=survival_auc_by_cell_type,
        mean_column="auc_mean",
        std_column="auc_std",
        xlabel=auc_xlabel,
        title="Survival-curve AUC comparison by cell type",
        output_path=comparisons_dir / "survival_curve_auc_by_cell_type.png",
        dpi=args.dpi,
    )
    plot_model_scalar_comparison(
        summary=survival_auc_pooled,
        mean_column="auc_mean",
        std_column="auc_std",
        xlabel=auc_xlabel,
        title="Survival-curve AUC comparison (all cell types pooled)",
        output_path=comparisons_dir / "survival_curve_auc.png",
        dpi=args.dpi,
    )

    spearman_by_cell_type_path = (
        tables_dir / "spearman_survival_curve_by_cell_type.csv"
    )
    pd.DataFrame(spearman_survival_rows).to_csv(
        spearman_by_cell_type_path,
        index=False,
    )
    logging.info("Wrote %s", spearman_by_cell_type_path)

    pooled_spearman_curves: dict[str, dict[str, np.ndarray]] = defaultdict(dict)
    pooled_spearman_rows: list[dict[str, object]] = []
    for model_name, by_seed in pooled_abs_spearman_rho.items():
        for seed_name, chunks in by_seed.items():
            combined = (
                np.concatenate(chunks) if chunks else np.array([], dtype=np.float64)
            )
            pooled_curve, n_pooled_genes = compute_survival_curve(
                combined,
                survival_thresholds,
            )
            pooled_spearman_curves[model_name][seed_name] = pooled_curve
            pooled_row: dict[str, object] = {
                "model": model_name,
                "seed": seed_name,
                "n_genes": n_pooled_genes,
                "auc": survival_curve_auc(pooled_curve, survival_thresholds),
            }
            for threshold, value in zip(survival_thresholds, pooled_curve):
                pooled_row[f"threshold_{threshold:.3f}"] = float(value)
            pooled_spearman_rows.append(pooled_row)

    spearman_pooled_path = tables_dir / "spearman_survival_curve_pooled.csv"
    pd.DataFrame(pooled_spearman_rows).to_csv(spearman_pooled_path, index=False)
    logging.info("Wrote %s", spearman_pooled_path)

    plot_survival_curve_pooled(
        pooled_curves=pooled_spearman_curves,
        thresholds=survival_thresholds,
        output_path=comparisons_dir / "spearman_survival_curve.png",
        dpi=args.dpi,
        correlation_name="Spearman rho",
    )
    plot_survival_curve_by_cell_type(
        curves=spearman_survival_curves_by_cell_type,
        thresholds=survival_thresholds,
        output_path=(
            comparisons_dir / "spearman_survival_curve_by_cell_type.png"
        ),
        dpi=args.dpi,
        correlation_name="Spearman rho",
    )

    spearman_auc_by_cell_type = summarize_metric_over_seeds(
        pd.DataFrame(spearman_survival_rows),
        ["model", "cell_type"],
        "auc",
    )
    spearman_auc_pooled = summarize_metric_over_seeds(
        pd.DataFrame(pooled_spearman_rows),
        ["model"],
        "auc",
    )
    spearman_auc_by_cell_type_path = (
        tables_dir / "spearman_survival_curve_auc_by_cell_type.csv"
    )
    spearman_auc_pooled_path = (
        tables_dir / "spearman_survival_curve_auc_pooled.csv"
    )
    spearman_auc_by_cell_type.to_csv(
        spearman_auc_by_cell_type_path,
        index=False,
    )
    spearman_auc_pooled.to_csv(spearman_auc_pooled_path, index=False)
    logging.info("Wrote %s", spearman_auc_by_cell_type_path)
    logging.info("Wrote %s", spearman_auc_pooled_path)

    spearman_auc_xlabel = (
        "AUC of absolute Spearman correlation survival curve (seed mean ± SD)"
    )
    plot_model_comparison(
        model_summary=spearman_auc_by_cell_type,
        mean_column="auc_mean",
        std_column="auc_std",
        xlabel=spearman_auc_xlabel,
        title="Spearman survival-curve AUC comparison by cell type",
        output_path=(
            comparisons_dir / "spearman_survival_curve_auc_by_cell_type.png"
        ),
        dpi=args.dpi,
    )
    plot_model_scalar_comparison(
        summary=spearman_auc_pooled,
        mean_column="auc_mean",
        std_column="auc_std",
        xlabel=spearman_auc_xlabel,
        title=(
            "Spearman survival-curve AUC comparison (all cell types pooled)"
        ),
        output_path=comparisons_dir / "spearman_survival_curve_auc.png",
        dpi=args.dpi,
    )

    pooled_sign_accuracy_curves: dict[str, dict[str, np.ndarray]] = defaultdict(dict)
    pooled_sign_accuracy_rows: list[dict[str, object]] = []
    for model_name, by_seed in pooled_pearson_r.items():
        for seed_name, chunks in by_seed.items():
            combined = (
                np.concatenate(chunks) if chunks else np.array([], dtype=np.float64)
            )
            pooled_curve, _ = compute_sign_accuracy_curve(
                combined,
                survival_thresholds,
            )
            pooled_sign_accuracy_curves[model_name][seed_name] = pooled_curve
            pooled_row = {
                "model": model_name,
                "seed": seed_name,
                "n_genes": int(np.isfinite(combined).sum()),
            }
            for threshold, value in zip(survival_thresholds, pooled_curve):
                pooled_row[f"threshold_{threshold:.3f}"] = float(value)
            pooled_sign_accuracy_rows.append(pooled_row)

    sign_accuracy_by_cell_type_path = tables_dir / "sign_accuracy_by_cell_type.csv"
    sign_accuracy_pooled_path = tables_dir / "sign_accuracy_pooled.csv"
    pd.DataFrame(sign_accuracy_rows).to_csv(
        sign_accuracy_by_cell_type_path,
        index=False,
    )
    pd.DataFrame(pooled_sign_accuracy_rows).to_csv(
        sign_accuracy_pooled_path,
        index=False,
    )
    logging.info("Wrote %s", sign_accuracy_by_cell_type_path)
    logging.info("Wrote %s", sign_accuracy_pooled_path)

    plot_sign_accuracy_pooled(
        pooled_curves=pooled_sign_accuracy_curves,
        thresholds=survival_thresholds,
        output_path=comparisons_dir / "sign_accuracy.png",
        dpi=args.dpi,
    )
    plot_sign_accuracy_by_cell_type(
        curves=sign_accuracy_curves_by_cell_type,
        thresholds=survival_thresholds,
        output_path=comparisons_dir / "sign_accuracy_by_cell_type.png",
        dpi=args.dpi,
    )


def report_uncertainty_ablation(
    *,
    args: argparse.Namespace,
    ablation_rows: list[dict[str, object]],
    tables_dir: Path,
    comparisons_dir: Path,
) -> None:
    """Write the survival-curve AUC of the high-uncertainty donor ablation."""
    ablation_table = pd.DataFrame(ablation_rows)
    ablation_path = tables_dir / "survival_curve_auc_uncertainty_ablation.csv"
    ablation_table.to_csv(ablation_path, index=False)
    logging.info("Wrote %s", ablation_path)
    if ablation_table.empty:
        logging.warning(
            "Uncertainty ablation produced no rows; skipping the "
            "per-cell-type AUC plot."
        )
        return

    ablation_summary = summarize_metric_over_seeds(
        ablation_table,
        ["model", "cell_type", "percent_removed"],
        "auc",
    )
    plot_uncertainty_ablation_by_cell_type(
        summary=ablation_summary,
        output_path=(
            comparisons_dir
            / "survival_curve_auc_uncertainty_ablation_by_cell_type.png"
        ),
        dpi=args.dpi,
        uncertainty_kind=args.ablation_uncertainty,
        individual_dir=comparisons_dir / "uncertainty_ablation",
    )
    max_percent = max(args.uncertainty_ablation_percents)
    for model_name in sorted(
        ablation_summary["model"].astype(str).unique(),
        key=natural_sort_key,
    ):
        model_rows = ablation_summary[
            ablation_summary["model"] == model_name
        ]
        baseline = model_rows[model_rows["percent_removed"] == 0.0]
        dropped = model_rows[
            model_rows["percent_removed"] == max_percent
        ]
        merged = baseline.merge(
            dropped,
            on="cell_type",
            suffixes=("_0", "_end"),
        )
        if merged.empty:
            continue
        deltas = (
            merged["auc_mean_end"].to_numpy(dtype=np.float64)
            - merged["auc_mean_0"].to_numpy(dtype=np.float64)
        )
        logging.info(
            "Uncertainty ablation (%s, %g%% vs 0%%) for %s: mean AUC "
            "change across %d cell types = %+.4f (min %+.4f, max %+.4f)",
            args.ablation_uncertainty,
            max_percent,
            model_name,
            int(len(deltas)),
            float(np.nanmean(deltas)),
            float(np.nanmin(deltas)),
            float(np.nanmax(deltas)),
        )


def report_uncertainty_outputs(
    *,
    args: argparse.Namespace,
    models: tuple[ModelLayout, ...],
    between_rows: list[dict[str, object]],
    within_rows: list[dict[str, object]],
    per_gene_uncertainty_frames: list[pd.DataFrame],
    between_curves: dict[str, dict[str, dict[str, dict[str, np.ndarray]]]],
    within_curves: dict[str, dict[str, dict[str, dict[str, np.ndarray]]]],
    tables_dir: Path,
) -> None:
    """Write the between-gene, within-gene and per-gene uncertainty outputs."""
    probabilistic_models = [
        model for model in models if model.probabilistic
    ]
    if not probabilistic_models:
        logging.warning(
            "--analyze-uncertainty was requested, but no probabilistic "
            "model layout was found."
        )
        return

    between_table = pd.DataFrame(between_rows)
    within_table = pd.DataFrame(within_rows)
    between_path = tables_dir / "uncertainty_between_seed.csv"
    within_path = tables_dir / "uncertainty_within_seed.csv"
    between_table.to_csv(between_path, index=False)
    within_table.to_csv(within_path, index=False)
    logging.info("Wrote %s", between_path)
    logging.info("Wrote %s", within_path)

    if per_gene_uncertainty_frames:
        per_gene_uncertainty_table = pivot_per_gene_uncertainty_spearman(
            per_gene_uncertainty_frames,
        )
        per_gene_uncertainty_path = (
            tables_dir / "uncertainty_per_gene_error_correlation.csv"
        )
        per_gene_uncertainty_table.to_csv(
            per_gene_uncertainty_path,
            index=False,
        )
        logging.info(
            "Wrote %s (%d gene rows)",
            per_gene_uncertainty_path,
            len(per_gene_uncertainty_table),
        )
    else:
        logging.warning(
            "No gene had a defined error-versus-uncertainty Spearman "
            "correlation; skipping "
            "tables/uncertainty_per_gene_error_correlation.csv."
        )

    for model in probabilistic_models:
        uncertainty_dir = args.output_dir / "uncertainty" / model.name
        plot_uncertainty_curves(
            model_name=model.name,
            curves=between_curves.get(model.name, {}),
            n_bins=args.uncertainty_bins,
            scope="between",
            output_path=(
                uncertainty_dir
                / "between_gene_uncertainty_vs_difficulty.png"
            ),
            dpi=args.dpi,
        )
        plot_uncertainty_curves(
            model_name=model.name,
            curves=within_curves.get(model.name, {}),
            n_bins=args.uncertainty_bins,
            scope="within",
            output_path=(
                uncertainty_dir
                / "within_gene_uncertainty_vs_error.png"
            ),
            dpi=args.dpi,
        )


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, pd.errors.ParserError) as error:
        logging.error("%s", error)
        raise SystemExit(2) from error
