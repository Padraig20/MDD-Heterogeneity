"""Make an UpSet plot from the per-cell-type tables written by ``twas.run``.

The input is the directory passed to ``python -m src.twas.run --output-dir``.
For every cell type, :mod:`src.twas.run` writes
``<cell type>/<arm>/results.csv``.  This script reads the significant genes
from those tables and shows both their total count in each cell type and the
size of every *exact* intersection retained in the plot.

Figures and sharing statistics are logged to a new WandB run named
``General`` in ``--wandb-project``, namespaced under
``General/<arm>/<scope>/...`` so this-study and ctPred stay distinct.
Optional ``--output`` still writes a local copy of each figure.

    For example::

    python -m src.twas.analysis.generate_upset_plot \
        --input-dir results/mdd \
        --wandb-project mdd-twas \
        --criterion fdr \
        --cell-types Memory_B_cell Naive_B_cell "CD4-positive alpha-beta T cell"

    python -m src.twas.analysis.generate_upset_plot \
        --input-dir results/mdd \
        --wandb-project mdd-twas \
        --criterion fdr \
        --combination expectation

    python -m src.twas.analysis.generate_upset_plot \
        --input-dir results/mdd \
        --wandb-project mdd-twas \
        --group-onek1k \
        --cell-types "B cell" "T cell" "NK cell"

No UpSet-specific package is required; the figure is drawn directly with
Matplotlib so it works with the dependencies already used by the TWAS code.
"""

from __future__ import annotations

import argparse
import gzip
import logging
import re
from collections import Counter
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from matplotlib.ticker import MaxNLocator

from src.twas.aggregate import acat_rows, benjamini_hochberg
from src.twas.compare import normalize_cell_type
from src.twas.wandb_logger import TwasWandBLogger


ARMS = ("this-study", "ctPred")
CRITERIA = ("fdr", "bonferroni")
COMBINATIONS = ("expectation", "acat")
SORT_INTERSECTIONS = ("degree", "cardinality")
SORT_SETS = ("size", "name", "input")
GENE_SCOPES = ("all", "in-mhc", "outside-mhc")
GENE_SCOPE_ALIASES = {
    "in-hmc": "in-mhc",
    "outside-hmc": "outside-mhc",
}
GENE_SCOPE_CHOICES = GENE_SCOPES + tuple(GENE_SCOPE_ALIASES)
DEFAULT_MHC_REGION = "6:25000000-34000000"
DEFAULT_GTF = Path("data/hg38/Homo_sapiens.GRCh38.115.gtf")
GENERAL_RUN_NAME = "General"

_GTF_GENE_ID = re.compile(r'gene_id\s+"([^"]+)"')

# The 29 OneK1K types used by scPrediXcan, collapsed to the 12 rows in its SLE
# UpSet plot (Zhou et al., Cell Genomics 2025, doi:10.1016/j.xgen.2025.100875).
# Multi-member groups are combined at the p-value level with ACAT, as described
# in the paper, rather than taking the union of significant hits.
# Double-negative thymocytes remain their own row and are therefore deliberately
# not part of the T-cell group.
ONEK1K_GROUPS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("Mononuclear cell", ("Peripheral_blood_mononuclear_cell",)),
    ("Lymphoid-cell", ("Innate_lymphoid_cell",)),
    ("Platelet", ("Platelet",)),
    ("Plasmablast", ("Plasmablast",)),
    ("Erythrocyte", ("Erythrocyte",)),
    ("Thymocyte", ("Double_negative_thymocyte",)),
    ("Hematopoietic cell", ("Hematopoietic_precursor_cell",)),
    (
        "Monocyte",
        (
            "CD14-low_CD16-positive_monocyte",
            "CD14-positive_monocyte",
        ),
    ),
    (
        "B cell",
        (
            "Memory_B_cell",
            "Naive_B_cell",
            "Transitional_stage_B_cell",
        ),
    ),
    (
        "NK cell",
        (
            "CD16-negative_CD56-bright_natural_killer_cell",
            "Natural_killer_cell",
        ),
    ),
    (
        "Dendritic cell",
        (
            "Conventional_dendritic_cell",
            "Dendritic_cell",
            "Plasmacytoid_dendritic_cell",
        ),
    ),
    (
        "T cell",
        (
            "CD4-positive_alpha-beta_cytotoxic_T_cell",
            "CD4-positive_alpha-beta_T_cell",
            "CD8-positive_alpha-beta_T_cell",
            "Central_memory_CD4-positive_alpha-beta_T_cell",
            "Central_memory_CD8-positive_alpha-beta_T_cell",
            "Effector_memory_CD4-positive_alpha-beta_T_cell",
            "Effector_memory_CD8-positive_alpha-beta_T_cell",
            "Gamma-delta_T_cell",
            "Mucosal_invariant_T_cell",
            "Naive_thymus-derived_CD4-positive_alpha-beta_T_cell",
            "Naive_thymus-derived_CD8-positive_alpha-beta_T_cell",
            "Regulatory_T_cell",
        ),
    ),
)

# Equivalent Cell Ontology labels found in the current OneK1K outputs. Keep the
# grouping table above in the spelling used by scPrediXcan's supplementary
# table, and canonicalize these dataset-specific variants during matching.
ONEK1K_NAME_ALIASES = {
    normalize_cell_type(
        "CD16-negative,_CD56-bright_natural_killer_cell,_human"
    ): normalize_cell_type("CD16-negative_CD56-bright_natural_killer_cell"),
    normalize_cell_type(
        "mucosal-associated_invariant_T_cell"
    ): normalize_cell_type("Mucosal_invariant_T_cell"),
}


def _onek1k_key(name: str) -> str:
    """Normalize a OneK1K name and resolve known ontology-label aliases."""
    key = normalize_cell_type(name)
    return ONEK1K_NAME_ALIASES.get(key, key)


def _normalize_chromosome(value: str) -> str:
    """Return a chromosome label without a leading ``chr`` prefix."""
    label = str(value).strip()
    if label.lower().startswith("chr"):
        label = label[3:]
    return label.upper()


def parse_genomic_region(value: str) -> tuple[str, int, int]:
    """Parse a one-based inclusive ``chromosome:start-end`` region."""
    match = re.fullmatch(
        r"\s*([^:]+):\s*([0-9][0-9,]*)\s*-\s*([0-9][0-9,]*)\s*",
        value,
    )
    if match is None:
        raise ValueError(
            f"Invalid genomic region {value!r}; expected chromosome:start-end."
        )
    chromosome = _normalize_chromosome(match.group(1))
    start = int(match.group(2).replace(",", ""))
    end = int(match.group(3).replace(",", ""))
    if start < 1 or end < start:
        raise ValueError(
            f"Invalid genomic region {value!r}; require 1 <= start <= end."
        )
    return chromosome, start, end


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate an UpSet plot of significant genes across the cell-type "
            "results written by src/twas/run.py."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "-i",
        "--input-dir",
        type=Path,
        required=True,
        help="The --output-dir of a completed src/twas/run.py invocation.",
    )
    parser.add_argument(
        "--wandb-project",
        type=str,
        required=True,
        help=(
            "WandB project. Figures and sharing statistics are logged to a new "
            "run named 'General' under General/<arm>/<gene-scope>/..."
        ),
    )
    parser.add_argument("--wandb-entity", type=str, default=None)
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help=(
            "Optional local figure (.png, .pdf, .svg, or another Matplotlib "
            "format). Omit to log only to WandB."
        ),
    )
    parser.add_argument(
        "--arm",
        nargs="+",
        choices=ARMS,
        default=list(ARMS),
        metavar="ARM",
        help="Which run.py model arm(s) to plot. Defaults to both.",
    )
    parser.add_argument(
        "--criterion",
        choices=CRITERIA,
        default="fdr",
        help="Use the corresponding significant_<criterion> column.",
    )
    parser.add_argument(
        "--combination",
        choices=COMBINATIONS,
        default="acat",
        help=(
            "Pooled significance track when --min-agreement is omitted. "
            "'acat' (the default) uses the Cauchy-combined per-draw p-values "
            "from an MI TWAS (significant_<criterion>); 'expectation' uses "
            "the E[z] flags (significant_<criterion>_expectation)."
        ),
    )
    parser.add_argument(
        "--min-agreement",
        type=float,
        default=None,
        metavar="PERCENT",
        help=(
            "For an MI run, define candidates by agreement_<criterion>: 0 means "
            "significant in at least one draw, and positive values require at "
            "least that percentage of draws. Omit to use pooled significance."
        ),
    )
    parser.add_argument(
        "--group-onek1k",
        "--onek1k-groups",
        action="store_true",
        help=(
            "Collapse the 29 fine OneK1K cell types to the 12 scPrediXcan "
            "groups. Multi-subtype p-values are combined per gene with ACAT."
        ),
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.05,
        help=(
            "FDR or family-wise alpha used to call ACAT-combined OneK1K groups. "
            "It does not alter the per-cell-type calls stored by run.py."
        ),
    )
    parser.add_argument(
        "--gene-scope",
        nargs="+",
        choices=GENE_SCOPE_CHOICES,
        default=["all"],
        help=(
            "Generate plots from all candidates, only genes overlapping the "
            "MHC interval, or candidates outside it. Multiple scopes create "
            "one suffixed output file per scope; hmc spellings are aliases."
        ),
    )
    parser.add_argument(
        "--gtf",
        type=Path,
        default=DEFAULT_GTF,
        help=(
            "Gene annotation used for in-mhc/outside-mhc. Its genome build "
            "must match --mhc-region; plain GTF and .gtf.gz are supported."
        ),
    )
    parser.add_argument(
        "--mhc-region",
        default=DEFAULT_MHC_REGION,
        metavar="CHR:START-END",
        help=(
            "One-based inclusive MHC interval. The default is a broad extended "
            "MHC interval on GRCh38; override it for another definition/build."
        ),
    )
    parser.add_argument(
        "--cell-types",
        nargs="+",
        default=None,
        metavar="CELL_TYPE",
        help=(
            "Plot only these cell types. Names may use spaces or underscores; "
            "the default is every cell type with a result for that arm. "
            "With --group-onek1k, specify the broad group names."
        ),
    )
    parser.add_argument(
        "--min-intersection-size",
        type=int,
        default=1,
        metavar="N",
        help="Omit exact intersections containing fewer than N genes.",
    )
    parser.add_argument(
        "--max-intersections",
        type=int,
        default=40,
        metavar="N",
        help="Show at most the N largest intersections; 0 shows all.",
    )
    parser.add_argument(
        "--sort-intersections",
        choices=SORT_INTERSECTIONS,
        default="degree",
        help=(
            "Order columns by membership count then pattern, or by descending "
            "intersection size."
        ),
    )
    parser.add_argument(
        "--sort-sets",
        choices=SORT_SETS,
        default="size",
        help="Order cell-type rows by total size, name, or discovery order.",
    )
    parser.add_argument(
        "--title",
        default=None,
        help="Optional title above the intersection-size bars.",
    )
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args(argv)

    if args.min_intersection_size < 1:
        parser.error("--min-intersection-size must be at least 1.")
    if args.min_agreement is not None and not 0 <= args.min_agreement <= 100:
        parser.error("--min-agreement must be between 0 and 100.")
    if not 0 < args.alpha < 1:
        parser.error("--alpha must be strictly between 0 and 1.")
    if args.max_intersections < 0:
        parser.error("--max-intersections cannot be negative.")
    if args.dpi < 1:
        parser.error("--dpi must be at least 1.")
    args.arm = list(dict.fromkeys(args.arm))
    args.gene_scope = [
        GENE_SCOPE_ALIASES.get(scope, scope) for scope in args.gene_scope
    ]
    if any(scope != "all" for scope in args.gene_scope):
        try:
            parse_genomic_region(args.mhc_region)
        except ValueError as error:
            parser.error(str(error))
    return args


def discover_result_files(
    input_dir: Path,
    arm: str = "this-study",
    cell_types: Sequence[str] | None = None,
) -> list[tuple[str, Path]]:
    """Find ``(cell_type, results.csv)`` pairs in a run.py output directory."""
    input_dir = Path(input_dir)
    if not input_dir.is_dir():
        raise NotADirectoryError(f"Not a run.py output directory: {input_dir}")

    discovered = [
        (path.parent.parent.name, path)
        for path in input_dir.glob(f"*/{arm}/results.csv")
        if path.is_file()
    ]
    discovered.sort(key=lambda item: item[0].casefold())
    if not discovered:
        raise FileNotFoundError(
            f"No */{arm}/results.csv files found under {input_dir}. "
            f"Choose the other --arm if this was a comparison run."
        )

    if cell_types is None:
        return discovered

    by_key: dict[str, tuple[str, Path]] = {}
    for item in discovered:
        key = normalize_cell_type(item[0])
        if key in by_key:
            raise ValueError(
                f"Ambiguous cell-type directories {by_key[key][0]!r} and "
                f"{item[0]!r} under {input_dir}."
            )
        by_key[key] = item

    selected: list[tuple[str, Path]] = []
    missing: list[str] = []
    seen: set[str] = set()
    for requested in cell_types:
        key = normalize_cell_type(requested)
        if key in seen:
            continue
        seen.add(key)
        item = by_key.get(key)
        if item is None:
            missing.append(requested)
        else:
            selected.append(item)
    if missing:
        available = ", ".join(name for name, _ in discovered)
        raise FileNotFoundError(
            f"No {arm} results for cell type(s) {missing}. Available: {available}"
        )
    return selected


def _boolean_mask(values: pd.Series, path: Path, column: str) -> pd.Series:
    """Read booleans robustly from CSV bool, numeric, or string columns."""
    if pd.api.types.is_bool_dtype(values.dtype):
        return values.fillna(False).astype(bool)
    if pd.api.types.is_numeric_dtype(values.dtype):
        return values.fillna(0).ne(0)

    normalized = values.astype("string").str.strip().str.lower()
    valid = normalized.isna() | normalized.isin(
        {"true", "false", "t", "f", "yes", "no", "1", "0"}
    )
    if not bool(valid.all()):
        invalid = sorted(normalized.loc[~valid].dropna().unique().tolist())
        raise ValueError(
            f"{path}: {column!r} contains values that are not boolean: {invalid[:5]}"
        )
    return normalized.isin({"true", "t", "yes", "1"})


def load_gene_sets(
    result_files: Iterable[tuple[str, Path]],
    criterion: str = "fdr",
    min_agreement: float | None = None,
    combination: str = "acat",
) -> dict[str, set[str]]:
    """Load candidate Ensembl gene IDs for every cell type.

    Without ``min_agreement``, candidates use the pooled significance flag
    (ACAT or E[z], per ``combination``). With it, agreement itself defines
    the candidates: zero has the established run.py meaning of significant in
    at least one draw, while positive cutoffs are inclusive. Agreement is
    always the per-draw call, independent of ``combination``.
    """
    if min_agreement is not None and not 0 <= min_agreement <= 100:
        raise ValueError("min_agreement must be between 0 and 100")
    if combination not in COMBINATIONS:
        raise ValueError(
            f"Unknown combination {combination!r}; expected one of {COMBINATIONS}."
        )
    suffix = _combination_suffix(combination) if min_agreement is None else ""
    significant_column = f"significant_{criterion}{suffix}"
    agreement_column = f"agreement_{criterion}"
    gene_sets: dict[str, set[str]] = {}
    for cell_type, path in result_files:
        candidate_column = (
            significant_column if min_agreement is None else agreement_column
        )
        wanted = {"gene", candidate_column}
        frame = pd.read_csv(path, usecols=lambda name: name in wanted)
        missing = wanted - set(frame.columns)
        if missing:
            agreement_hint = (
                " --min-agreement is only available for MI results."
                if agreement_column in missing
                else ""
            )
            combination_hint = (
                " Pass --combination acat (the default) for ACAT flags, or "
                "re-run src/twas/run.py with --model-kind mi to write E[z] "
                "columns as significant_<criterion>_expectation."
                if combination == "expectation" and significant_column in missing
                else ""
            )
            raise ValueError(
                f"{path} is missing required column(s) {sorted(missing)}. "
                f"Pass --criterion matching the output of run.py."
                f"{agreement_hint}{combination_hint}"
            )
        if min_agreement is None:
            mask = _boolean_mask(
                frame[significant_column], path, significant_column
            )
        else:
            agreement = _agreement_values(
                frame[agreement_column], path, agreement_column
            )
            mask = _agreement_mask(agreement, min_agreement)
        genes = frame.loc[mask, "gene"].dropna().astype(str).str.strip()
        gene_sets[cell_type] = set(genes.loc[genes.ne("")])
    return gene_sets


def _combination_suffix(combination: str) -> str:
    return "_expectation" if combination == "expectation" else ""


def _agreement_values(values: pd.Series, path: Path, column: str) -> pd.Series:
    """Parse and validate run.py's fractional MI agreement column."""
    numeric = pd.to_numeric(values, errors="coerce")
    invalid = values.notna() & numeric.isna()
    if bool(invalid.any()):
        examples = values.loc[invalid].astype(str).unique().tolist()[:5]
        raise ValueError(f"{path}: {column!r} is not numeric: {examples}")
    outside = numeric.notna() & ~numeric.between(0.0, 1.0)
    if bool(outside.any()):
        examples = numeric.loc[outside].unique().tolist()[:5]
        raise ValueError(
            f"{path}: {column!r} must contain fractions from 0 to 1, got "
            f"{examples}."
        )
    return numeric


def _agreement_mask(agreement: pd.Series, percent: float) -> pd.Series:
    """Apply run.py's special zero-is-any, otherwise-inclusive convention."""
    threshold = percent / 100.0
    return agreement.gt(0.0) if threshold == 0.0 else agreement.ge(threshold)


def _pvalue_series(
    path: Path,
    pvalue_column: str = "pvalue",
) -> pd.Series:
    """Read one gene-indexed pooled p-value series for grouped ACAT."""
    wanted = {"gene", pvalue_column}
    frame = pd.read_csv(path, usecols=lambda name: name in wanted)
    missing = wanted - set(frame.columns)
    if missing:
        raise ValueError(
            f"{path} is missing required column(s) {sorted(missing)}."
        )

    genes = frame["gene"].astype("string").str.strip()
    valid_gene = genes.notna() & genes.ne("")
    if bool(genes.loc[valid_gene].duplicated().any()):
        duplicated = genes.loc[valid_gene & genes.duplicated(keep=False)].iloc[0]
        raise ValueError(f"{path}: duplicate gene identifier {duplicated!r}.")

    pvalues = pd.to_numeric(frame[pvalue_column], errors="coerce")
    invalid = frame[pvalue_column].notna() & pvalues.isna()
    if bool(invalid.any()):
        examples = frame.loc[invalid, pvalue_column].astype(str).unique().tolist()[:5]
        raise ValueError(f"{path}: {pvalue_column!r} is not numeric: {examples}")
    outside = pvalues.notna() & ~pvalues.between(0.0, 1.0)
    if bool(outside.any()):
        examples = pvalues.loc[outside].unique().tolist()[:5]
        raise ValueError(f"{path}: p-values must be between 0 and 1: {examples}")

    pvalues = pvalues.loc[valid_gene].copy()
    pvalues.index = genes.loc[valid_gene].astype(str)
    return pvalues


def _call_combined_genes(
    pvalues: pd.Series, criterion: str, alpha: float
) -> set[str]:
    """Apply the selected correction to one group's ACAT p-values."""
    significant = _significant_pvalue_mask(pvalues, criterion, alpha)
    return set(pvalues.index[significant].astype(str))


def _significant_pvalue_mask(
    pvalues: pd.Series, criterion: str, alpha: float
) -> np.ndarray:
    """Correct one complete TWAS p-value vector and return its calls."""
    values = pvalues.to_numpy(dtype=float)
    finite = np.isfinite(values)
    n_tested = int(finite.sum())
    if not n_tested:
        return np.zeros(len(pvalues), dtype=bool)
    if criterion == "fdr":
        significant = benjamini_hochberg(values) < alpha
    elif criterion == "bonferroni":
        significant = values < alpha / n_tested
    else:
        raise ValueError(f"Unknown significance criterion: {criterion}")
    return np.asarray(significant) & finite


def _per_draw_pvalue_series(results_path: Path) -> pd.Series:
    """Read ``(gene, draw) -> pvalue`` beside one MI results table."""
    path = results_path.parent / "per_draw_zscores.csv"
    if not path.is_file():
        raise FileNotFoundError(
            f"{path} is required to compute agreement after OneK1K grouping. "
            "Run src/twas/run.py with --model-kind mi."
        )
    wanted = {"gene", "draw", "pvalue"}
    frame = pd.read_csv(path, usecols=lambda name: name in wanted)
    missing = wanted - set(frame.columns)
    if missing:
        raise ValueError(f"{path} is missing required column(s) {sorted(missing)}.")

    genes = frame["gene"].astype("string").str.strip()
    draws = frame["draw"].astype("string").str.strip()
    valid_key = genes.notna() & genes.ne("") & draws.notna() & draws.ne("")
    frame = frame.loc[valid_key].copy()
    frame["gene"] = genes.loc[valid_key].astype(str)
    frame["draw"] = draws.loc[valid_key].astype(str)
    if bool(frame.duplicated(["gene", "draw"]).any()):
        duplicate = frame.loc[
            frame.duplicated(["gene", "draw"], keep=False), ["gene", "draw"]
        ].iloc[0]
        raise ValueError(
            f"{path}: duplicate gene/draw pair "
            f"({duplicate['gene']!r}, {duplicate['draw']!r})."
        )

    pvalues = pd.to_numeric(frame["pvalue"], errors="coerce")
    invalid = frame["pvalue"].notna() & pvalues.isna()
    if bool(invalid.any()):
        examples = frame.loc[invalid, "pvalue"].astype(str).unique().tolist()[:5]
        raise ValueError(f"{path}: 'pvalue' is not numeric: {examples}")
    outside = pvalues.notna() & ~pvalues.between(0.0, 1.0)
    if bool(outside.any()):
        examples = pvalues.loc[outside].unique().tolist()[:5]
        raise ValueError(f"{path}: p-values must be between 0 and 1: {examples}")
    pvalues.index = pd.MultiIndex.from_frame(frame[["gene", "draw"]])
    return pvalues


def _grouped_mi_gene_set(
    members: Sequence[tuple[str, Path]],
    criterion: str,
    min_agreement: float,
    alpha: float,
) -> set[str]:
    """ACAT-combine each MI draw, then threshold group-level agreement."""
    columns = {
        cell_type: _per_draw_pvalue_series(path)
        for cell_type, path in members
    }
    combined = acat_rows(pd.concat(columns, axis=1)).dropna()
    calls: list[pd.DataFrame] = []
    for _, draw_values in combined.groupby(level="draw", sort=False):
        significant = _significant_pvalue_mask(draw_values, criterion, alpha)
        calls.append(pd.DataFrame({
            "gene": draw_values.index.get_level_values("gene"),
            "significant": significant,
        }))
    if not calls:
        return set()
    agreement = (
        pd.concat(calls, ignore_index=True)
        .groupby("gene", sort=False)["significant"]
        .mean()
    )
    return set(agreement.index[_agreement_mask(agreement, min_agreement)])


def group_onek1k_gene_sets(
    result_files: Sequence[tuple[str, Path]],
    *,
    criterion: str = "fdr",
    min_agreement: float | None = None,
    alpha: float = 0.05,
    combination: str = "acat",
) -> dict[str, set[str]]:
    """Collapse fine OneK1K results to scPrediXcan's 12 ontology groups.

    Without an agreement cutoff, multi-subtype pooled p-values (ACAT or E[z],
    per ``combination``) are combined by ACAT and significance is recalculated
    at ``alpha``. With a cutoff, ACAT and multiple-testing correction are
    performed separately within each MI draw; candidates are then defined by
    the resulting group-level agreement. A single-member group reads the
    equivalent calls already written by run.py.
    """
    if min_agreement is not None and not 0 <= min_agreement <= 100:
        raise ValueError("min_agreement must be between 0 and 100")
    if not 0 < alpha < 1:
        raise ValueError("alpha must be strictly between 0 and 1")
    if combination not in COMBINATIONS:
        raise ValueError(
            f"Unknown combination {combination!r}; expected one of {COMBINATIONS}."
        )
    by_key: dict[str, tuple[str, Path]] = {}
    for item in result_files:
        key = _onek1k_key(item[0])
        if key in by_key:
            raise ValueError(
                f"Ambiguous cell-type directories {by_key[key][0]!r} and "
                f"{item[0]!r}."
            )
        by_key[key] = item

    recognized = {
        _onek1k_key(member)
        for _, members in ONEK1K_GROUPS
        for member in members
    }
    unknown = [
        name
        for name, _ in result_files
        if _onek1k_key(name) not in recognized
    ]
    if unknown:
        raise ValueError(
            "--group-onek1k received cell types outside the scPrediXcan "
            f"OneK1K mapping: {unknown}."
        )

    grouped: dict[str, set[str]] = {}
    for group_name, expected_members in ONEK1K_GROUPS:
        members = [
            by_key[key]
            for member in expected_members
            if (key := _onek1k_key(member)) in by_key
        ]
        if not members:
            continue
        missing = len(expected_members) - len(members)
        if missing and len(expected_members) > 1:
            logging.warning(
                "%s: combining %d of %d expected OneK1K subtypes.",
                group_name,
                len(members),
                len(expected_members),
            )
        if len(expected_members) == 1:
            grouped[group_name] = load_gene_sets(
                members,
                criterion=criterion,
                min_agreement=min_agreement,
                combination=combination,
            )[members[0][0]]
            logging.info(
                "%s: %d candidate genes from %s.",
                group_name,
                len(grouped[group_name]),
                members[0][0],
            )
            continue

        if min_agreement is None:
            pvalue_column = (
                "pvalue_expectation" if combination == "expectation" else "pvalue"
            )
            columns = {
                cell_type: _pvalue_series(path, pvalue_column=pvalue_column)
                for cell_type, path in members
            }
            combined = acat_rows(pd.concat(columns, axis=1))
            grouped[group_name] = _call_combined_genes(
                combined, criterion, alpha
            )
            detail = "pooled ACAT"
        else:
            grouped[group_name] = _grouped_mi_gene_set(
                members, criterion, min_agreement, alpha
            )
            detail = "per-draw ACAT and MI agreement"
        logging.info(
            "%s: %d candidate genes after %s across %d subtype(s).",
            group_name,
            len(grouped[group_name]),
            detail,
            len(members),
        )
    if not grouped:
        raise ValueError("No recognized OneK1K cell types were available to group.")
    return grouped


def select_gene_sets(
    gene_sets: Mapping[str, set[str]], requested: Sequence[str] | None
) -> dict[str, set[str]]:
    """Select sets by normalized display name while retaining request order."""
    if requested is None:
        return dict(gene_sets)
    by_key = {
        normalize_cell_type(name): (name, genes)
        for name, genes in gene_sets.items()
    }
    selected: dict[str, set[str]] = {}
    missing: list[str] = []
    for request in requested:
        item = by_key.get(normalize_cell_type(request))
        if item is None:
            missing.append(request)
        else:
            selected.setdefault(item[0], item[1])
    if missing:
        raise ValueError(
            f"Unknown cell type(s) {missing}. Available: {list(gene_sets)}"
        )
    return selected


def _gene_identifier_key(value: str) -> str:
    """Canonicalize an Ensembl identifier for matching against a GTF."""
    return str(value).split(".", 1)[0].strip().upper()


def load_mhc_gene_ids(
    gtf_path: Path,
    region: str = DEFAULT_MHC_REGION,
) -> tuple[set[str], set[str]]:
    """Return all annotated IDs and IDs whose gene interval overlaps MHC."""
    gtf_path = Path(gtf_path)
    if not gtf_path.is_file():
        raise FileNotFoundError(
            f"GTF required for the MHC split was not found: {gtf_path}"
        )
    chromosome, region_start, region_end = parse_genomic_region(region)
    opener = gzip.open if gtf_path.suffix == ".gz" else open
    annotated: set[str] = set()
    in_mhc: set[str] = set()
    with opener(gtf_path, "rt") as handle:
        for line in handle:
            if not line or line.startswith("#"):
                continue
            fields = line.rstrip("\n").split("\t")
            if len(fields) < 9 or fields[2] != "gene":
                continue
            identifier_match = _GTF_GENE_ID.search(fields[8])
            if identifier_match is None:
                continue
            identifier = _gene_identifier_key(identifier_match.group(1))
            if not identifier:
                continue
            annotated.add(identifier)
            if _normalize_chromosome(fields[0]) != chromosome:
                continue
            try:
                gene_start = int(fields[3])
                gene_end = int(fields[4])
            except ValueError as error:
                raise ValueError(
                    f"{gtf_path}: non-integer gene coordinates in line "
                    f"{line.rstrip()!r}."
                ) from error
            if gene_end >= region_start and gene_start <= region_end:
                in_mhc.add(identifier)
    if not annotated:
        raise ValueError(
            f"{gtf_path} contains no GTF gene features with gene_id attributes."
        )
    logging.info(
        "MHC definition: %s:%d-%d contains %d of %d annotated genes from %s.",
        chromosome,
        region_start,
        region_end,
        len(in_mhc),
        len(annotated),
        gtf_path,
    )
    return annotated, in_mhc


def filter_gene_sets_by_scope(
    gene_sets: Mapping[str, set[str]],
    scope: str,
    mhc_gene_ids: set[str] | None = None,
) -> dict[str, set[str]]:
    """Restrict candidate sets to all, inside-MHC, or outside-MHC genes."""
    if scope not in GENE_SCOPES:
        raise ValueError(f"Unknown gene scope: {scope}")
    if scope == "all":
        return {name: set(genes) for name, genes in gene_sets.items()}
    if mhc_gene_ids is None:
        raise ValueError(f"MHC gene IDs are required for scope {scope!r}.")
    keep_inside = scope == "in-mhc"
    return {
        name: {
            gene
            for gene in genes
            if (_gene_identifier_key(gene) in mhc_gene_ids) == keep_inside
        }
        for name, genes in gene_sets.items()
    }


def output_for_variant(
    output: Path,
    *,
    arm: str,
    scope: str,
    multiple_arms: bool,
    multiple_scopes: bool,
) -> Path:
    """Add arm/scope suffixes when one invocation writes several plots."""
    output = Path(output)
    parts: list[str] = []
    if multiple_arms:
        parts.append(arm)
    if multiple_scopes:
        parts.append(scope)
    if not parts:
        return output
    return output.with_name(f"{output.stem}_{'_'.join(parts)}{output.suffix}")


def order_gene_sets(
    gene_sets: Mapping[str, set[str]], sort_by: str = "size"
) -> dict[str, set[str]]:
    """Return a regular dict in the requested top-to-bottom row order."""
    items = list(gene_sets.items())
    if sort_by == "size":
        items.sort(key=lambda item: (len(item[1]), item[0].casefold()))
    elif sort_by == "name":
        items.sort(key=lambda item: item[0].casefold())
    elif sort_by != "input":
        raise ValueError(f"Unknown set ordering: {sort_by}")
    return dict(items)


def compute_intersections(
    gene_sets: Mapping[str, set[str]],
    *,
    min_size: int = 1,
    max_intersections: int = 40,
    sort_by: str = "degree",
) -> pd.DataFrame:
    """Count and order exact gene-membership intersections.

    A gene present in A and B is counted in the ``{A, B}`` column only if it is
    absent from every other displayed cell type. This is the standard UpSet
    definition and prevents a gene from being counted in several top bars.
    """
    if min_size < 1:
        raise ValueError("min_size must be at least 1")
    if max_intersections < 0:
        raise ValueError("max_intersections cannot be negative")
    if not gene_sets:
        raise ValueError("At least one gene set is required")

    memberships: Counter[frozenset[str]] = Counter()
    all_genes: set[str] = set().union(*gene_sets.values())
    for gene in all_genes:
        membership = frozenset(
            cell_type for cell_type, genes in gene_sets.items() if gene in genes
        )
        if membership:
            memberships[membership] += 1

    rows = [
        {"members": members, "size": size, "degree": len(members)}
        for members, size in memberships.items()
        if size >= min_size
    ]
    names = list(gene_sets)
    position = {name: index for index, name in enumerate(names)}

    def pattern(row: dict) -> tuple[int, ...]:
        return tuple(
            position[name] for name in names if name in row["members"]
        )

    if sort_by not in SORT_INTERSECTIONS:
        raise ValueError(f"Unknown intersection ordering: {sort_by}")

    n_available = len(rows)
    if max_intersections and n_available > max_intersections:
        # Limiting by display order would bias degree ordering toward low-degree
        # columns and could hide a much larger, widely shared intersection.
        rows = sorted(
            rows,
            key=lambda row: (-row["size"], row["degree"], pattern(row)),
        )[:max_intersections]

    if sort_by == "degree":
        rows.sort(key=lambda row: (row["degree"], pattern(row)))
    else:
        rows.sort(key=lambda row: (-row["size"], row["degree"], pattern(row)))

    result = pd.DataFrame(rows, columns=["members", "size", "degree"])
    result.attrs["n_available"] = n_available
    return result


def sharing_summary(gene_sets: Mapping[str, set[str]]) -> dict:
    """Count unique hits by how many displayed cell types they appear in."""
    n_cell_types = len(gene_sets)
    all_genes: set[str] = set().union(*gene_sets.values()) if gene_sets else set()
    n_hits = len(all_genes)
    n_shared_all = n_shared_at_least_two = n_single = 0
    for gene in all_genes:
        degree = sum(gene in genes for genes in gene_sets.values())
        if degree >= 2:
            n_shared_at_least_two += 1
        if n_cell_types and degree == n_cell_types:
            n_shared_all += 1
        if degree == 1:
            n_single += 1

    def percent(count: int) -> float:
        return 100.0 * count / n_hits if n_hits else 0.0

    return {
        "n_hits": n_hits,
        "n_cell_types": n_cell_types,
        "n_shared_all": n_shared_all,
        "pct_shared_all": percent(n_shared_all),
        "n_shared_at_least_two": n_shared_at_least_two,
        "pct_shared_at_least_two": percent(n_shared_at_least_two),
        "n_single": n_single,
        "pct_single": percent(n_single),
    }


def format_sharing_summary(
    summary: Mapping[str, float],
    *,
    hit_kind: str = "TWAS hits",
) -> str:
    """The scPrediXcan-style sharing sentence for one UpSet gene universe."""
    noun = "cell type" if summary["n_cell_types"] == 1 else "cell types"
    return (
        f"Among {int(summary['n_hits']):,} {hit_kind} from the "
        f"{int(summary['n_cell_types']):,} {noun}, "
        f"{int(summary['n_shared_all']):,} ({summary['pct_shared_all']:.1f}%) "
        f"genes are shared in all cell types, "
        f"{int(summary['n_shared_at_least_two']):,} "
        f"({summary['pct_shared_at_least_two']:.1f}%) genes are shared in at "
        f"least two cell types, and {int(summary['n_single']):,} "
        f"({summary['pct_single']:.1f}%) genes yield significance in a single "
        f"cell type."
    )


def _log_key(arm: str, scope: str, name: str) -> str:
    """Namespace one metric, figure, or table under ``General/<arm>/<scope>/``."""
    return f"{GENERAL_RUN_NAME}/{arm}/{scope}/{name}"


def _scope_statistics(
    summary: Mapping[str, float],
    intersections: pd.DataFrame,
) -> dict[str, int | float]:
    """Scalar sharing and intersection counts for one gene-scope plot."""
    n_available = intersections.attrs.get("n_available", len(intersections))
    return {
        "n_hits": int(summary["n_hits"]),
        "n_cell_types": int(summary["n_cell_types"]),
        "n_shared_all": int(summary["n_shared_all"]),
        "pct_shared_all": float(summary["pct_shared_all"]),
        "n_shared_at_least_two": int(summary["n_shared_at_least_two"]),
        "pct_shared_at_least_two": float(summary["pct_shared_at_least_two"]),
        "n_single": int(summary["n_single"]),
        "pct_single": float(summary["pct_single"]),
        "n_intersections_shown": int(len(intersections)),
        "n_intersections_available": int(n_available),
    }


def _set_size_table(gene_sets: Mapping[str, set[str]], arm: str) -> pd.DataFrame:
    """One row per displayed cell type with its candidate-gene count."""
    return pd.DataFrame(
        [
            {"arm": arm, "cell_type": name, "n_genes": len(genes)}
            for name, genes in gene_sets.items()
        ]
    )


def _intersection_table(intersections: pd.DataFrame, arm: str) -> pd.DataFrame:
    """Serialize exact intersections for a WandB table."""
    rows = []
    for _, row in intersections.iterrows():
        members = sorted(row["members"], key=str.casefold)
        rows.append(
            {
                "arm": arm,
                "members": " ∩ ".join(members),
                "n_members": int(row["degree"]),
                "n_genes": int(row["size"]),
            }
        )
    return pd.DataFrame(rows, columns=["arm", "members", "n_members", "n_genes"])


def _wandb_config(args: argparse.Namespace) -> dict:
    """Record the analysis settings on the General run."""
    return {
        "arms": list(args.arm),
        "criterion": args.criterion,
        "combination": args.combination,
        "min_agreement": args.min_agreement,
        "group_onek1k": args.group_onek1k,
        "alpha": args.alpha,
        "gene_scope": list(args.gene_scope),
        "gtf": str(args.gtf),
        "mhc_region": args.mhc_region,
        "cell_types": list(args.cell_types) if args.cell_types else None,
        "min_intersection_size": args.min_intersection_size,
        "max_intersections": args.max_intersections,
        "sort_intersections": args.sort_intersections,
        "sort_sets": args.sort_sets,
        "input_dir": str(args.input_dir),
    }


def _arm_min_agreement(args: argparse.Namespace, arm: str) -> float | None:
    """Agreement cutoffs are an MI this-study feature; ctPred has no such columns."""
    if arm != "this-study":
        return None
    return args.min_agreement


def load_arm_gene_sets(
    args: argparse.Namespace, arm: str
) -> dict[str, set[str]]:
    """Load (and optionally OneK1K-group) candidate sets for one model arm."""
    min_agreement = _arm_min_agreement(args, arm)
    if args.min_agreement is not None and min_agreement is None:
        logging.info(
            "Arm %s has no MI agreement columns; using pooled %s significance.",
            arm,
            args.combination,
        )
    if args.group_onek1k:
        files = discover_result_files(args.input_dir, arm)
        gene_sets = group_onek1k_gene_sets(
            files,
            criterion=args.criterion,
            min_agreement=min_agreement,
            alpha=args.alpha,
            combination=args.combination,
        )
        return select_gene_sets(gene_sets, args.cell_types)
    files = discover_result_files(args.input_dir, arm, args.cell_types)
    return load_gene_sets(
        files,
        criterion=args.criterion,
        min_agreement=min_agreement,
        combination=args.combination,
    )


def _figure_title(title: str | None, arm: str) -> str:
    """Keep the arm visible on the figure itself, not only in the WandB key."""
    return f"{title} ({arm})" if title else arm


def _display_name(name: str) -> str:
    return name.replace("_", " ")


def plot_upset(
    gene_sets: Mapping[str, set[str]],
    intersections: pd.DataFrame,
    *,
    title: str | None = None,
) -> Figure:
    """Draw the set-size bars, intersection bars, and membership matrix."""
    names = list(gene_sets)
    n_sets = len(names)
    n_intersections = len(intersections)
    # Size grows with the two dimensions but remains usable for small inputs.
    figure_width = max(8.0, 4.8 + 0.42 * max(n_intersections, 1))
    figure_height = max(5.0, 2.5 + 0.42 * n_sets)
    figure = plt.figure(
        figsize=(figure_width, figure_height), constrained_layout=False
    )
    grid = figure.add_gridspec(
        2,
        3,
        height_ratios=(2.3, max(2.3, 0.42 * n_sets)),
        width_ratios=(2.0, 3.8, max(3.5, 0.42 * max(n_intersections, 1))),
        hspace=0.08,
        wspace=0.03,
    )
    top = figure.add_subplot(grid[0, 2])
    set_sizes = figure.add_subplot(grid[1, 0])
    labels = figure.add_subplot(grid[1, 1], sharey=set_sizes)
    matrix = figure.add_subplot(grid[1, 2], sharey=set_sizes)

    y = np.arange(n_sets)
    totals = np.asarray([len(gene_sets[name]) for name in names], dtype=int)
    max_total = max(int(totals.max(initial=0)), 1)
    set_sizes.barh(y, totals, height=0.56, color="#171780", edgecolor="black")
    set_sizes.set_xlim(-0.24 * max_total, 1.05 * max_total)
    for row, total in zip(y, totals):
        set_sizes.text(
            -0.04 * max_total,
            row,
            f"{total:,}",
            ha="right",
            va="center",
            fontsize=10,
            fontweight="bold",
        )
    set_sizes.set_title("Total candidate\ncausal genes", fontsize=11, fontweight="bold")
    set_sizes.set_yticks([])
    set_sizes.set_xticks([])
    for side in ("top", "right", "bottom", "left"):
        set_sizes.spines[side].set_visible(False)

    labels.set_xlim(0, 1)
    labels.set_xticks([])
    labels.set_yticks([])
    for row, name in zip(y, names):
        labels.text(
            0.98,
            row,
            _display_name(name),
            ha="right",
            va="center",
            fontsize=11,
            fontweight="bold",
        )
    for spine in labels.spines.values():
        spine.set_visible(False)

    # Alternating bands make it much easier to track wide intersection rows.
    for row in y:
        if row % 2:
            labels.axhspan(row - 0.5, row + 0.5, color="#f0f0f0", zorder=0)
            matrix.axhspan(row - 0.5, row + 0.5, color="#f0f0f0", zorder=0)

    if n_intersections:
        x = np.arange(n_intersections)
        sizes = intersections["size"].to_numpy(dtype=int)
        top.bar(x, sizes, width=0.58, color="black")
        top.set_xlim(-0.55, n_intersections - 0.45)
        matrix.set_xlim(-0.55, n_intersections - 0.45)

        for column, members in enumerate(intersections["members"]):
            selected_rows = [row for row, name in enumerate(names) if name in members]
            matrix.scatter(
                np.full(n_sets, column),
                y,
                s=31,
                color="#cccccc",
                edgecolors="none",
                zorder=2,
            )
            if len(selected_rows) > 1:
                matrix.plot(
                    [column, column],
                    [min(selected_rows), max(selected_rows)],
                    color="black",
                    linewidth=1.35,
                    zorder=3,
                )
            matrix.scatter(
                np.full(len(selected_rows), column),
                selected_rows,
                s=45,
                color="black",
                edgecolors="black",
                linewidths=0.3,
                zorder=4,
            )
    else:
        top.set_xlim(-0.5, 0.5)
        matrix.set_xlim(-0.5, 0.5)
        top.text(
            0.5,
            0.5,
            "No significant genes",
            transform=top.transAxes,
            ha="center",
            va="center",
            color="0.35",
        )

    top.set_ylabel("Shared genes", fontsize=11, fontweight="bold")
    top.yaxis.set_major_locator(MaxNLocator(integer=True, min_n_ticks=3))
    top.tick_params(axis="x", bottom=False, labelbottom=False)
    top.spines["top"].set_visible(False)
    top.spines["right"].set_visible(False)
    if title:
        top.set_title(title, fontsize=12, fontweight="bold", pad=8)

    matrix.set_ylim(n_sets - 0.5, -0.5)
    matrix.set_xticks([])
    matrix.set_yticks([])
    for spine in matrix.spines.values():
        spine.set_visible(False)
    labels.set_ylim(n_sets - 0.5, -0.5)
    set_sizes.set_ylim(n_sets - 0.5, -0.5)

    figure.subplots_adjust(left=0.04, right=0.99, top=0.96, bottom=0.05)
    return figure


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s: %(message)s",
    )
    if args.min_agreement is not None:
        if args.min_agreement == 0:
            logging.info(
                "MI candidate rule: significant by %s in at least one draw "
                "(agreement > 0).",
                args.criterion,
            )
        else:
            logging.info(
                "MI candidate rule: agreement_%s >= %g%%.",
                args.criterion,
                args.min_agreement,
            )
    elif args.combination == "expectation":
        logging.info(
            "Pooled candidate rule: significant_%s_expectation (E[z] p-values).",
            args.criterion,
        )
    else:
        logging.info(
            "Pooled candidate rule: significant_%s (ACAT-combined MI p-values).",
            args.criterion,
        )
    try:
        arms = list(args.arm)
        scopes = list(dict.fromkeys(args.gene_scope))
        arm_gene_sets: dict[str, dict[str, set[str]]] = {}
        for arm in arms:
            try:
                arm_gene_sets[arm] = load_arm_gene_sets(args, arm)
            except FileNotFoundError as error:
                logging.warning("Skipping arm %s: %s", arm, error)
        if not arm_gene_sets:
            raise FileNotFoundError(
                f"No */{{{','.join(arms)}}}/results.csv files found under "
                f"{args.input_dir}."
            )

        mhc_gene_ids: set[str] | None = None
        if any(scope != "all" for scope in scopes):
            annotated_ids, mhc_gene_ids = load_mhc_gene_ids(
                args.gtf, args.mhc_region
            )
            candidate_ids = {
                _gene_identifier_key(gene)
                for gene_sets in arm_gene_sets.values()
                for genes in gene_sets.values()
                for gene in genes
            }
            missing_annotation = candidate_ids - annotated_ids
            if missing_annotation:
                logging.warning(
                    "%d candidate gene(s) are absent from %s; they are treated "
                    "as outside the MHC interval.",
                    len(missing_annotation),
                    args.gtf,
                )

        if args.output is not None:
            args.output.parent.mkdir(parents=True, exist_ok=True)

        figures: dict[str, Figure] = {}
        wandb_summary: dict[str, int | float] = {}
        tables: dict[str, pd.DataFrame] = {}
        hit_kind = (
            "E[z] TWAS hits"
            if args.combination == "expectation"
            else "TWAS hits"
        )
        multiple_arms = len(arm_gene_sets) > 1
        multiple_scopes = len(scopes) > 1

        for arm, base_gene_sets in arm_gene_sets.items():
            for scope in scopes:
                gene_sets = filter_gene_sets_by_scope(
                    base_gene_sets, scope, mhc_gene_ids
                )
                gene_sets = order_gene_sets(gene_sets, sort_by=args.sort_sets)
                intersections = compute_intersections(
                    gene_sets,
                    min_size=args.min_intersection_size,
                    max_intersections=args.max_intersections,
                    sort_by=args.sort_intersections,
                )
                figure = plot_upset(
                    gene_sets,
                    intersections,
                    title=_figure_title(args.title, arm),
                )
                sharing = sharing_summary(gene_sets)
                stats = _scope_statistics(sharing, intersections)
                figures[_log_key(arm, scope, "upset")] = figure
                wandb_summary.update(
                    {
                        _log_key(arm, scope, name): value
                        for name, value in stats.items()
                    }
                )
                tables[_log_key(arm, scope, "set_sizes")] = _set_size_table(
                    gene_sets, arm
                )
                tables[_log_key(arm, scope, "intersections")] = (
                    _intersection_table(intersections, arm)
                )

                if args.output is not None:
                    output = output_for_variant(
                        args.output,
                        arm=arm,
                        scope=scope,
                        multiple_arms=multiple_arms,
                        multiple_scopes=multiple_scopes,
                    )
                    figure.savefig(output, dpi=args.dpi, bbox_inches="tight")
                    logging.info("Wrote %s [%s/%s].", output, arm, scope)

                sentence = format_sharing_summary(sharing, hit_kind=hit_kind)
                logging.info(
                    "%s [%s/%s]: %d cell types, %d candidate genes, %d of %d "
                    "intersections shown.",
                    GENERAL_RUN_NAME,
                    arm,
                    scope,
                    stats["n_cell_types"],
                    stats["n_hits"],
                    stats["n_intersections_shown"],
                    stats["n_intersections_available"],
                )
                logging.info("%s", sentence)

        logger = TwasWandBLogger(
            project=args.wandb_project, entity=args.wandb_entity
        )
        config = _wandb_config(args)
        config["arms"] = list(arm_gene_sets)
        logger.start(GENERAL_RUN_NAME, config=config)
        try:
            logger.log_results(wandb_summary, figures, tables=tables)
            logging.info(
                "Logged %d figure(s) and %d statistic(s) to WandB run %r "
                "in project %s.",
                len(figures),
                len(wandb_summary),
                GENERAL_RUN_NAME,
                args.wandb_project,
            )
        finally:
            logger.finish()
            for figure in figures.values():
                plt.close(figure)
    except (FileNotFoundError, NotADirectoryError, OSError, ValueError) as error:
        logging.error("%s", error)
        raise SystemExit(1) from error


if __name__ == "__main__":
    main()
