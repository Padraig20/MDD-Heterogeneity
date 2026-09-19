"""Benchmark TWAS arms against the SLE silver standard.

Reads the per-cell-type tables written by ``src.twas.run``
(``<cell type>/<arm>/results.csv``) and the locus-gene TSV from
``src.preprocessing.onek1k.get_sle_silver_standard``. For every cell type
that has both ``this-study`` and ``ctPred`` results, each arm is scored on
how well its significant genes and LD blocks recover the silver standard,
split into all / in-MHC / outside-MHC.

The comparison is restricted to genes both arms actually tested, so recall
uses the same denominator. A silver LD block counts as recovered only when
the arm hits a silver-standard gene assigned to that GWAS locus -- not when
an unrelated gene in the same Berisa-Pickrell block happens to be
significant. Precision and F1 are logged but are not the headline: the
silver standard is an incomplete list of known SLE genes, so extra TWAS
hits are additional discoveries, not false positives.

WandB logging follows :mod:`src.twas.analysis.generate_upset_plot`: one run
per (arm, cell type), with every figure, table, and scalar under
``Benchmark/...``. Run names are ``benchmark-<arm>/<cell-type>``. After the
per-cell-type runs, recall, precision, F1 and the other metrics are averaged
across cell types and logged as ``benchmark-<arm>/bulk``. Each run also gets
a standard QQ plot (with genomic-control ``lambda`` and ``N``) and a QQ that
splits silver-standard genes from all others. Comparison figures, overlap
metrics, and the this-study-vs-ctPred QQ (silver-standard vs other genes)
are logged only on the this-study run. The bulk QQs combine each gene's
p-values across cell types with ACAT.

    python -m src.twas.analysis.compare_silver_standard \\
        --input-dir sle-twas/reg-model-mi-sep-norm/results \\
        --silver-standard sle_silver_standard.tsv \\
        --wandb-project sle-twas \\
        --criterion fdr
"""

from __future__ import annotations

import argparse
import logging
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from matplotlib.ticker import MaxNLocator

from src.twas.analysis.generate_upset_plot import (
    ARMS,
    COMBINATIONS,
    CRITERIA,
    DEFAULT_MHC_REGION,
    GENE_SCOPE_ALIASES,
    GENE_SCOPE_CHOICES,
    GENE_SCOPES,
    _agreement_mask,
    _agreement_values,
    _boolean_mask,
    _combination_suffix,
    _gene_identifier_key,
    discover_result_files,
    load_mhc_gene_ids,
    parse_genomic_region,
)
from src.twas.aggregate import acat_rows, genomic_inflation
from src.twas.compare import normalize_cell_type, two_sample_quantiles
from src.twas.ld_blocks import UNASSIGNED
from src.twas.wandb_logger import TwasWandBLogger


LOGGER = logging.getLogger("compare_silver_standard")

BENCHMARK_PREFIX = "Benchmark"
BULK_LABEL = "bulk"
SHARED_ARM = "this-study"
DEFAULT_GTF = Path("data/reference-genome/Homo_sapiens.GRCh38.115.gtf")
ARM_LABELS = {"this-study": "This study", "ctPred": "ctPred"}
ARM_COLORS = {"this-study": "#3f6fb0", "ctPred": "#b06a3f"}
QQ_SILVER_COLOR = "#ff7f0e"
QQ_OTHER_COLOR = "#1f77b4"
QQ_BONFERRONI_ALPHA = 0.05
MAX_LOG10P = 320.0
BLOCK_PATTERN = re.compile(
    r"^(?:chr)?([^:]+):(\d+)-(\d+)$", re.IGNORECASE
)


@dataclass
class SilverStandard:
    """Locus-gene silver standard with LD-block assignments."""

    frame: pd.DataFrame
    gene_keys: set[str]
    gene_names: dict[str, str]
    gene_blocks: dict[str, set[int]]
    gene_block_labels: dict[str, set[str]]
    blocks: dict[int, str]
    gene_in_mhc: set[str]
    block_in_mhc: set[int]


@dataclass
class ArmHits:
    """Significant genes and LD blocks from one arm in one cell type."""

    cell_type: str
    arm: str
    tested_genes: set[str]
    hit_genes: set[str]
    gene_names: dict[str, str]
    gene_blocks: dict[str, int]
    block_labels: dict[int, str]
    tested_blocks: set[int]
    hit_blocks: set[int]
    gene_in_mhc: set[str]
    block_in_mhc: set[int]
    pvalues: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))


@dataclass
class Recovery:
    """Precision / recall of one arm against one silver-standard slice."""

    n_silver: float = 0
    n_silver_tested: float = 0
    n_hits: float = 0
    n_recovered: float = 0
    n_missed: float = 0
    n_extra: float = 0
    precision: float = float("nan")
    recall: float = float("nan")
    recall_tested: float = float("nan")
    f1: float = float("nan")
    recovered: list[str] = field(default_factory=list)
    missed: list[str] = field(default_factory=list)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare this-study and ctPred TWAS hits to the SLE silver "
            "standard, for genes and LD blocks, inside and outside the MHC."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "-i",
        "--input-dir",
        type=Path,
        required=True,
        help="The --output-dir of a completed src.twas.run invocation.",
    )
    parser.add_argument(
        "-s",
        "--silver-standard",
        type=Path,
        required=True,
        help="TSV written by get_sle_silver_standard.py.",
    )
    parser.add_argument(
        "--wandb-project",
        type=str,
        required=True,
        help=(
            "WandB project. Each (arm, cell type) pair becomes a run named "
            "benchmark-<arm>/<cell-type>, with artifacts under Benchmark/..."
        ),
    )
    parser.add_argument("--wandb-entity", type=str, default=None)
    parser.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        default=None,
        help="Optional directory for local copies of figures and tables.",
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
            "Pooled significance track. 'acat' uses significant_<criterion>; "
            "'expectation' uses significant_<criterion>_expectation."
        ),
    )
    parser.add_argument(
        "--min-agreement",
        type=float,
        default=None,
        metavar="PERCENT",
        help=(
            "For an MI this-study arm, call hits by agreement_<criterion>. "
            "ctPred always uses pooled significance."
        ),
    )
    parser.add_argument(
        "--gene-scope",
        nargs="+",
        choices=GENE_SCOPE_CHOICES,
        default=list(GENE_SCOPES),
        help="Which MHC slices to score. Default is all three.",
    )
    parser.add_argument(
        "--gtf",
        type=Path,
        default=DEFAULT_GTF,
        help="Gene annotation used to place genes in or out of the MHC.",
    )
    parser.add_argument(
        "--mhc-region",
        default=DEFAULT_MHC_REGION,
        metavar="CHR:START-END",
        help=(
            "One-based inclusive MHC interval. Must match the LD-block build "
            "(hg19 for the Berisa-Pickrell EUR file used by SLE TWAS)."
        ),
    )
    parser.add_argument(
        "--cell-types",
        nargs="+",
        default=None,
        metavar="CELL_TYPE",
        help="Restrict to these cell types. Default is every type with both arms.",
    )
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)

    if args.min_agreement is not None and not 0 <= args.min_agreement <= 100:
        parser.error("--min-agreement must be between 0 and 100.")
    if args.dpi < 1:
        parser.error("--dpi must be at least 1.")
    args.gene_scope = [
        GENE_SCOPE_ALIASES.get(scope, scope) for scope in args.gene_scope
    ]
    args.gene_scope = list(dict.fromkeys(args.gene_scope))
    try:
        parse_genomic_region(args.mhc_region)
    except ValueError as error:
        parser.error(str(error))
    return args


def _parse_block_indices(value: object) -> list[int]:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return []
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none"}:
        return []
    indices: list[int] = []
    for part in text.split(";"):
        part = part.strip()
        if not part:
            continue
        try:
            index = int(float(part))
        except ValueError:
            continue
        if index != UNASSIGNED:
            indices.append(index)
    return indices


def _parse_block_labels(value: object) -> list[str]:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return []
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none"}:
        return []
    return [part.strip() for part in text.split(";") if part.strip()]


def _block_interval(label: str) -> tuple[str, int, int] | None:
    match = BLOCK_PATTERN.match(str(label).strip())
    if match is None:
        return None
    chrom = match.group(1).upper().removeprefix("CHR")
    return chrom, int(match.group(2)), int(match.group(3))


def _overlaps_region(
    chrom: str, start: int, stop: int, region: tuple[str, int, int]
) -> bool:
    region_chrom, region_start, region_end = region
    if chrom.upper().removeprefix("CHR") != region_chrom:
        return False
    return stop >= region_start and start <= region_end


def load_silver_standard(
    path: Path,
    mhc_region: str,
    mhc_gene_ids: set[str] | None,
) -> SilverStandard:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Silver standard not found: {path}")
    frame = pd.read_csv(path, sep="\t")
    required = {"locus", "gene"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(
            f"{path} is missing required column(s) {sorted(missing)}."
        )
    region = parse_genomic_region(mhc_region)

    gene_keys: set[str] = set()
    gene_names: dict[str, str] = {}
    gene_blocks: dict[str, set[int]] = {}
    gene_block_labels: dict[str, set[str]] = {}
    blocks: dict[int, str] = {}
    gene_in_mhc: set[str] = set()
    block_in_mhc: set[int] = set()

    for row in frame.itertuples(index=False):
        ensid = str(getattr(row, "ensid", "") or "").strip()
        symbol = str(getattr(row, "gene", "") or "").strip()
        key = _gene_identifier_key(ensid) if ensid else symbol.upper()
        if not key or key in {"NAN", "NONE"}:
            continue
        gene_keys.add(key)
        if symbol:
            gene_names[key] = symbol
        indices = _parse_block_indices(getattr(row, "ld_block_index", ""))
        labels = _parse_block_labels(getattr(row, "ld_block", ""))
        gene_blocks.setdefault(key, set()).update(indices)
        gene_block_labels.setdefault(key, set()).update(labels)
        for index, label in zip(indices, labels):
            blocks[index] = label
        in_mhc = False
        if mhc_gene_ids is not None and ensid:
            in_mhc = _gene_identifier_key(ensid) in mhc_gene_ids
        elif labels:
            in_mhc = any(
                interval is not None and _overlaps_region(*interval, region)
                for interval in (_block_interval(label) for label in labels)
            )
        if in_mhc:
            gene_in_mhc.add(key)

    for index, label in blocks.items():
        interval = _block_interval(label)
        if interval is not None and _overlaps_region(*interval, region):
            block_in_mhc.add(index)

    LOGGER.info(
        "Silver standard: %d genes, %d LD blocks (%d genes / %d blocks in MHC) "
        "from %s.",
        len(gene_keys),
        len(blocks),
        len(gene_in_mhc),
        len(block_in_mhc),
        path,
    )
    return SilverStandard(
        frame=frame,
        gene_keys=gene_keys,
        gene_names=gene_names,
        gene_blocks=gene_blocks,
        gene_block_labels=gene_block_labels,
        blocks=blocks,
        gene_in_mhc=gene_in_mhc,
        block_in_mhc=block_in_mhc,
    )


def _hit_mask(
    frame: pd.DataFrame,
    path: Path,
    criterion: str,
    combination: str,
    min_agreement: float | None,
) -> pd.Series:
    if min_agreement is not None:
        column = f"agreement_{criterion}"
        if column not in frame.columns:
            raise ValueError(
                f"{path} has no {column} column; --min-agreement is only "
                "available for MI this-study results."
            )
        agreement = _agreement_values(frame[column], path, column)
        return _agreement_mask(agreement, min_agreement)
    suffix = _combination_suffix(combination)
    column = f"significant_{criterion}{suffix}"
    if column not in frame.columns:
        raise ValueError(
            f"{path} is missing {column}. Pass --criterion/--combination "
            "matching the output of src.twas.run."
        )
    return _boolean_mask(frame[column], path, column)


def _extract_pvalues(
    frame: pd.DataFrame,
    keys: pd.Series,
    valid: pd.Series,
    combination: str,
    path: Path,
) -> pd.Series:
    """Gene-indexed pooled p-values for the selected combination track."""
    column = f"pvalue{_combination_suffix(combination)}"
    if column not in frame.columns:
        LOGGER.warning(
            "%s has no %r column; QQ plots for this arm will be skipped.",
            path,
            column,
        )
        return pd.Series(dtype=float)
    numeric = pd.to_numeric(frame[column], errors="coerce")
    keep = valid & numeric.notna()
    return pd.Series(
        numeric[keep].to_numpy(dtype=float),
        index=keys[keep].astype(str),
        name="pvalue",
    )


def load_arm_hits(
    path: Path,
    cell_type: str,
    arm: str,
    criterion: str,
    combination: str,
    min_agreement: float | None,
    mhc_region: str,
    mhc_gene_ids: set[str] | None,
) -> ArmHits:
    path = Path(path)
    frame = pd.read_csv(path)
    if "gene" not in frame.columns:
        raise ValueError(f"{path} has no 'gene' column.")
    genes = frame["gene"].astype("string").str.strip()
    keys = genes.map(_gene_identifier_key)
    valid = keys.notna() & keys.ne("") & keys.ne("NAN")
    names: dict[str, str] = {}
    if "gene_name" in frame.columns:
        for key, name in zip(keys[valid], frame.loc[valid, "gene_name"].astype(str)):
            name = name.strip()
            if name and name.lower() not in {"nan", "none"}:
                names[str(key)] = name

    tested = set(keys[valid].astype(str))
    mask = _hit_mask(frame, path, criterion, combination, min_agreement) & valid
    hit_genes = set(keys[mask].astype(str))

    gene_blocks: dict[str, int] = {}
    block_labels: dict[int, str] = {}
    if "block_index" in frame.columns:
        indices = pd.to_numeric(frame["block_index"], errors="coerce")
        labels = (
            frame["block"].astype(str)
            if "block" in frame.columns
            else pd.Series("", index=frame.index)
        )
        for key, index, label in zip(keys[valid], indices[valid], labels[valid]):
            if not np.isfinite(index):
                continue
            index_i = int(index)
            if index_i == UNASSIGNED:
                continue
            gene_blocks[str(key)] = index_i
            if label and label.lower() not in {"nan", "none"}:
                block_labels.setdefault(index_i, label)

    tested_blocks = {gene_blocks[gene] for gene in tested if gene in gene_blocks}
    hit_blocks = {gene_blocks[gene] for gene in hit_genes if gene in gene_blocks}
    hits = _build_arm_hits(
        cell_type=cell_type,
        arm=arm,
        tested_genes=tested,
        hit_genes=hit_genes,
        gene_names=names,
        gene_blocks=gene_blocks,
        block_labels=block_labels,
        tested_blocks=tested_blocks,
        hit_blocks=hit_blocks,
        mhc_region=mhc_region,
        mhc_gene_ids=mhc_gene_ids,
    )
    hits.pvalues = _extract_pvalues(frame, keys, valid, combination, path)
    return hits


def _build_arm_hits(
    *,
    cell_type: str,
    arm: str,
    tested_genes: set[str],
    hit_genes: set[str],
    gene_names: dict[str, str],
    gene_blocks: dict[str, int],
    block_labels: dict[int, str],
    tested_blocks: set[int],
    hit_blocks: set[int],
    mhc_region: str,
    mhc_gene_ids: set[str] | None,
) -> ArmHits:
    region = parse_genomic_region(mhc_region)
    block_in_mhc = {
        index
        for index, label in block_labels.items()
        if (interval := _block_interval(label)) is not None
        and _overlaps_region(*interval, region)
    }
    if mhc_gene_ids is not None:
        gene_in_mhc = {gene for gene in tested_genes if gene in mhc_gene_ids}
    else:
        gene_in_mhc = {
            gene
            for gene, index in gene_blocks.items()
            if index in block_in_mhc
        }
    return ArmHits(
        cell_type=cell_type,
        arm=arm,
        tested_genes=tested_genes,
        hit_genes=hit_genes,
        gene_names=gene_names,
        gene_blocks=gene_blocks,
        block_labels=block_labels,
        tested_blocks=tested_blocks,
        hit_blocks=hit_blocks,
        gene_in_mhc=gene_in_mhc,
        block_in_mhc=block_in_mhc,
    )


def paired_cell_types(
    input_dir: Path, requested: Sequence[str] | None
) -> list[tuple[str, Path, Path]]:
    """``(display_name, this-study path, ctPred path)`` for cell types with both arms."""
    ours = {
        normalize_cell_type(name): (name, path)
        for name, path in discover_result_files(input_dir, "this-study")
    }
    theirs = {
        normalize_cell_type(name): (name, path)
        for name, path in discover_result_files(input_dir, "ctPred")
    }
    keys = sorted(set(ours) & set(theirs))
    if requested is not None:
        wanted = [normalize_cell_type(name) for name in requested]
        missing = [name for name, key in zip(requested, wanted) if key not in keys]
        if missing:
            available = sorted(ours[key][0] for key in keys)
            raise FileNotFoundError(
                f"No paired this-study/ctPred results for {missing}. "
                f"Paired cell types: {available}"
            )
        keys = list(dict.fromkeys(wanted))
    if not keys:
        raise FileNotFoundError(
            f"No cell type under {input_dir} has both this-study and ctPred "
            "results.csv files."
        )
    paired = [(ours[key][0], ours[key][1], theirs[key][1]) for key in keys]
    LOGGER.info("Found %d cell type(s) with both arms.", len(paired))
    return paired


def _scoped_genes(silver: SilverStandard, scope: str) -> set[str]:
    if scope == "all":
        return set(silver.gene_keys)
    if scope == "in-mhc":
        return set(silver.gene_in_mhc)
    if scope == "outside-mhc":
        return silver.gene_keys - silver.gene_in_mhc
    raise ValueError(f"Unknown gene scope: {scope}")


def _scoped_blocks(silver: SilverStandard, scope: str) -> set[int]:
    if scope == "all":
        return set(silver.blocks)
    if scope == "in-mhc":
        return set(silver.block_in_mhc)
    if scope == "outside-mhc":
        return set(silver.blocks) - silver.block_in_mhc
    raise ValueError(f"Unknown gene scope: {scope}")


def _scoped_hits(
    hits: ArmHits, scope: str, silver: SilverStandard | None = None
) -> tuple[set[str], set[int]]:
    """TWAS yield in this MHC slice, using GTF genes and block intervals."""
    hit_genes = set(hits.hit_genes)
    hit_blocks = set(hits.hit_blocks)
    if scope == "all":
        return hit_genes, hit_blocks
    mhc_genes = set(hits.gene_in_mhc)
    if silver is not None:
        mhc_genes |= hit_genes & silver.gene_in_mhc
    mhc_blocks = set(hits.block_in_mhc)
    if scope == "in-mhc":
        return hit_genes & mhc_genes, hit_blocks & mhc_blocks
    if scope == "outside-mhc":
        return hit_genes - mhc_genes, hit_blocks - mhc_blocks
    raise ValueError(f"Unknown gene scope: {scope}")


def shared_universe(ours: ArmHits, theirs: ArmHits) -> set[str]:
    """Genes both arms tested, so recall uses one denominator."""
    return ours.tested_genes & theirs.tested_genes


def _rate(numerator: int, denominator: int) -> float:
    if denominator <= 0:
        return float("nan")
    return numerator / denominator


def _f1(precision: float, recall: float) -> float:
    if not math.isfinite(precision) or not math.isfinite(recall):
        return float("nan")
    if precision + recall == 0:
        return float("nan")
    return 2.0 * precision * recall / (precision + recall)


def _hits_in_silver(
    hit_genes: set[str], hits: ArmHits, silver_genes: set[str], silver: SilverStandard
) -> set[str]:
    """Map TWAS hit Ensembl ids (and symbols) onto silver-standard gene keys."""
    recovered = hit_genes & silver_genes
    symbol_to_key = {
        name.upper(): key
        for key, name in silver.gene_names.items()
        if key in silver_genes
    }
    for gene in hit_genes:
        name = hits.gene_names.get(gene, "").upper()
        if name and name in symbol_to_key:
            recovered.add(symbol_to_key[name])
    return recovered


def _tested_silver(
    silver_genes: set[str],
    hits: ArmHits,
    silver: SilverStandard,
    universe: set[str] | None = None,
) -> set[str]:
    """Silver genes this arm tested, optionally restricted to a shared universe."""
    tested = silver_genes & hits.tested_genes
    if universe is not None:
        tested &= universe
    eligible_names = {
        name.upper()
        for key, name in hits.gene_names.items()
        if universe is None or key in universe
    }
    for key in silver_genes:
        name = silver.gene_names.get(key, "").upper()
        if name and name in eligible_names:
            tested.add(key)
    return tested


def _silver_blocks_for_genes(
    genes: set[str], silver: SilverStandard, silver_blocks: set[int]
) -> set[int]:
    recovered: set[int] = set()
    for gene in genes:
        recovered.update(silver.gene_blocks.get(gene, ()))
    return recovered & silver_blocks


def recover_genes(
    hits: ArmHits,
    silver: SilverStandard,
    scope: str,
    universe: set[str] | None = None,
) -> Recovery:
    silver_genes = _scoped_genes(silver, scope)
    hit_genes, _ = _scoped_hits(hits, scope, silver)
    if universe is not None:
        hit_genes &= universe
    tested = _tested_silver(silver_genes, hits, silver, universe=universe)
    recovered = sorted(_hits_in_silver(hit_genes, hits, silver_genes, silver))
    missed = sorted(tested - set(recovered))
    extra = sorted(hit_genes - silver_genes)
    precision = _rate(len(recovered), len(hit_genes))
    recall = _rate(len(recovered), len(silver_genes))
    recall_tested = _rate(len(recovered), len(tested))
    return Recovery(
        n_silver=len(silver_genes),
        n_silver_tested=len(tested),
        n_hits=len(hit_genes),
        n_recovered=len(recovered),
        n_missed=len(missed),
        n_extra=len(extra),
        precision=precision,
        recall=recall,
        recall_tested=recall_tested,
        f1=_f1(precision, recall_tested),
        recovered=recovered,
        missed=missed,
    )


def recover_blocks(
    hits: ArmHits,
    silver: SilverStandard,
    scope: str,
    universe: set[str] | None = None,
) -> Recovery:
    """Recover a silver GWAS LD block only by hitting a silver gene in it."""
    silver_blocks = _scoped_blocks(silver, scope)
    hit_genes, hit_blocks = _scoped_hits(hits, scope, silver)
    if universe is not None:
        hit_genes &= universe
        hit_blocks = {
            hits.gene_blocks[gene]
            for gene in hit_genes
            if gene in hits.gene_blocks
        }
        if scope == "in-mhc":
            hit_blocks &= hits.block_in_mhc
        elif scope == "outside-mhc":
            hit_blocks -= hits.block_in_mhc
    recovered_genes = _hits_in_silver(hit_genes, hits, silver.gene_keys, silver)
    recovered = sorted(
        _silver_blocks_for_genes(set(recovered_genes), silver, silver_blocks)
    )
    tested_genes = _tested_silver(
        silver.gene_keys, hits, silver, universe=universe
    )
    tested = _silver_blocks_for_genes(tested_genes, silver, silver_blocks)
    missed = sorted(tested - set(recovered))
    extra = sorted(hit_blocks - silver_blocks)
    precision = _rate(len(recovered), len(hit_blocks))
    recall = _rate(len(recovered), len(silver_blocks))
    recall_tested = _rate(len(recovered), len(tested))
    return Recovery(
        n_silver=len(silver_blocks),
        n_silver_tested=len(tested),
        n_hits=len(hit_blocks),
        n_recovered=len(recovered),
        n_missed=len(missed),
        n_extra=len(extra),
        precision=precision,
        recall=recall,
        recall_tested=recall_tested,
        f1=_f1(precision, recall_tested),
        recovered=[str(index) for index in recovered],
        missed=[str(index) for index in missed],
    )


def _log_key(scope: str, name: str) -> str:
    return f"{BENCHMARK_PREFIX}/{scope}/{name}"


def _recovery_metrics(prefix: str, recovery: Recovery) -> dict[str, int | float]:
    return {
        f"{prefix}n_silver": recovery.n_silver,
        f"{prefix}n_silver_tested": recovery.n_silver_tested,
        f"{prefix}n_hits": recovery.n_hits,
        f"{prefix}n_recovered": recovery.n_recovered,
        f"{prefix}n_missed": recovery.n_missed,
        f"{prefix}n_extra": recovery.n_extra,
        f"{prefix}precision": recovery.precision,
        f"{prefix}recall": recovery.recall,
        f"{prefix}recall_tested": recovery.recall_tested,
        f"{prefix}f1": recovery.f1,
    }


def _item_table(
    keys: Sequence[str],
    names: dict[str, str],
    *,
    kind: str,
    status: str,
    blocks: dict[str, int] | None = None,
    block_labels: dict[int, str] | None = None,
    silver_names: dict[str, str] | None = None,
) -> pd.DataFrame:
    rows = []
    silver_names = silver_names or {}
    for key in keys:
        index = (blocks or {}).get(key)
        rows.append(
            {
                "kind": kind,
                "status": status,
                "id": key,
                "name": names.get(key) or silver_names.get(key, ""),
                "block_index": "" if index is None else index,
                "block": (
                    ""
                    if index is None
                    else (block_labels or {}).get(index, "")
                ),
            }
        )
    columns = ["kind", "status", "id", "name", "block_index", "block"]
    return pd.DataFrame(rows, columns=columns)


def _metrics_table(
    cell_type: str,
    scope: str,
    scores: dict[str, dict[str, Recovery]],
) -> pd.DataFrame:
    rows = []
    for arm, pair in scores.items():
        for kind, recovery in pair.items():
            rows.append(
                {
                    "cell_type": cell_type,
                    "arm": arm,
                    "scope": scope,
                    "kind": kind,
                    "n_silver": recovery.n_silver,
                    "n_silver_tested": recovery.n_silver_tested,
                    "n_hits": recovery.n_hits,
                    "n_recovered": recovery.n_recovered,
                    "n_missed": recovery.n_missed,
                    "n_extra": recovery.n_extra,
                    "precision": recovery.precision,
                    "recall": recovery.recall,
                    "recall_tested": recovery.recall_tested,
                    "f1": recovery.f1,
                }
            )
    return pd.DataFrame(rows)


def _overlap_metrics(
    ours: Recovery, theirs: Recovery
) -> dict[str, float]:
    ours_set = set(ours.recovered)
    theirs_set = set(theirs.recovered)
    return {
        "n_recovered_both": float(len(ours_set & theirs_set)),
        "n_recovered_this_study_only": float(len(ours_set - theirs_set)),
        "n_recovered_ctPred_only": float(len(theirs_set - ours_set)),
    }


_RECOVERY_MEAN_FIELDS = (
    "n_silver",
    "n_silver_tested",
    "n_hits",
    "n_recovered",
    "n_missed",
    "n_extra",
    "precision",
    "recall",
    "recall_tested",
    "f1",
)


def _mean(values: Sequence[float]) -> float:
    finite = [float(value) for value in values if math.isfinite(value)]
    if not finite:
        return float("nan")
    return float(np.mean(finite))


def _average_recovery(items: Sequence[Recovery]) -> Recovery:
    kwargs = {
        name: _mean([getattr(item, name) for item in items])
        for name in _RECOVERY_MEAN_FIELDS
    }
    return Recovery(**kwargs)


def _average_overlap(
    items: Sequence[dict[str, dict[str, float]]],
    scopes: Sequence[str],
) -> dict[str, dict[str, float]]:
    keys = items[0][scopes[0]].keys()
    return {
        scope: {
            key: _mean([item[scope][key] for item in items])
            for key in keys
        }
        for scope in scopes
    }


def average_cell_type_scores(
    all_scores: Sequence[dict[str, dict[str, dict[str, Recovery]]]],
    scopes: Sequence[str],
) -> dict[str, dict[str, dict[str, Recovery]]]:
    averaged: dict[str, dict[str, dict[str, Recovery]]] = {
        "this-study": {},
        "ctPred": {},
    }
    for arm in ARMS:
        for scope in scopes:
            averaged[arm][scope] = {
                kind: _average_recovery(
                    [scores[arm][scope][kind] for scores in all_scores]
                )
                for kind in ("genes", "ld_blocks")
            }
    return averaged


def _chart_title(cell_type: str, suffix: str) -> str:
    if cell_type == BULK_LABEL:
        return f"bulk — mean across cell types — {suffix}"
    return f"{cell_type} — {suffix}"


def _format_count(value: float) -> str:
    if not math.isfinite(value):
        return "?"
    if abs(value - round(value)) < 1e-6:
        return str(int(round(value)))
    return f"{value:.1f}"


def _fraction_label(numerator: float, denominator: float) -> str:
    return f"{_format_count(numerator)}/{_format_count(denominator)}"


def _annotate_bar_fractions(
    ax,
    container,
    recoveries: Sequence[Recovery],
    *,
    denominator: str,
) -> None:
    for bar, recovery in zip(container, recoveries):
        denom = getattr(recovery, denominator)
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            bar.get_height(),
            _fraction_label(recovery.n_recovered, denom),
            ha="center",
            va="bottom",
            fontsize=6.5,
            clip_on=False,
        )


def plot_recovery(
    scores: dict[str, dict[str, dict[str, Recovery]]],
    cell_type: str,
    scopes: Sequence[str],
) -> Figure:
    """this-study vs ctPred: recovered counts and recall on the shared universe."""
    kinds = ("genes", "ld_blocks")
    panels = (
        ("n_recovered", "Recovered silver-standard items", False, "n_silver_tested"),
        ("recall_tested", "Recall", True, "n_silver_tested"),
    )
    figure, axes = plt.subplots(2, 2, figsize=(10.8, 7.8), squeeze=False)
    x = np.arange(len(scopes))
    width = 0.36
    offsets = (-width / 2.0, width / 2.0)
    panel_titles = {
        (0, 0): "Recovered genes",
        (0, 1): "Gene recall (recovered / tested)",
        (1, 0): "Recovered LD blocks",
        (1, 1): "LD-block recall (recovered / tested)",
    }

    for row, kind in enumerate(kinds):
        for col, (metric, ylabel, rate, denom) in enumerate(panels):
            ax = axes[row, col]
            for offset, arm in zip(offsets, ARMS):
                recoveries = [scores[arm][scope][kind] for scope in scopes]
                values = [
                    0.0 if not math.isfinite(recovery.__dict__[metric]) else recovery.__dict__[metric]
                    for recovery in recoveries
                ]
                bars = ax.bar(
                    x + offset,
                    values,
                    width=width * 0.95,
                    color=ARM_COLORS[arm],
                    edgecolor="white",
                    linewidth=0.4,
                    label=ARM_LABELS[arm],
                )
                _annotate_bar_fractions(ax, bars, recoveries, denominator=denom)
            ax.set_xticks(x)
            ax.set_xticklabels(scopes)
            ax.set_title(panel_titles[(row, col)])
            ax.set_ylabel("Count" if not rate else "Recall")
            if rate:
                ax.set_ylim(0, 1.22)
            else:
                if cell_type != BULK_LABEL:
                    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
                _, ymax = ax.get_ylim()
                ax.set_ylim(0, ymax * 1.18 if ymax > 0 else 1)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.set_axisbelow(True)
            ax.yaxis.grid(True, color="0.9", linewidth=0.6)

    axes[0, 0].legend(frameon=False, fontsize=8)
    figure.suptitle(
        _chart_title(cell_type, "silver-standard recovery"),
        fontsize=12,
    )
    figure.tight_layout()
    return figure


def _plot_rate_metric(
    scores: dict[str, dict[str, dict[str, Recovery]]],
    cell_type: str,
    scopes: Sequence[str],
    metric: str,
    title: str,
    ylabel: str,
    denominator: str | None = None,
) -> Figure:
    """Two-row this-study vs ctPred bars for one 0–1 metric (genes, then LD blocks)."""
    kinds = (("genes", "Genes"), ("ld_blocks", "LD blocks"))
    figure, axes = plt.subplots(2, 1, figsize=(7.6, 7.2), sharex=True, squeeze=False)
    x = np.arange(len(scopes))
    width = 0.36
    offsets = (-width / 2.0, width / 2.0)
    for row, (kind, kind_title) in enumerate(kinds):
        ax = axes[row, 0]
        for offset, arm in zip(offsets, ARMS):
            recoveries = [scores[arm][scope][kind] for scope in scopes]
            values = [
                0.0 if not math.isfinite(recovery.__dict__[metric]) else recovery.__dict__[metric]
                for recovery in recoveries
            ]
            bars = ax.bar(
                x + offset,
                values,
                width=width * 0.95,
                color=ARM_COLORS[arm],
                edgecolor="white",
                linewidth=0.4,
                label=ARM_LABELS[arm],
            )
            if denominator is not None:
                _annotate_bar_fractions(ax, bars, recoveries, denominator=denominator)
        ax.set_ylabel(ylabel)
        ax.set_title(kind_title)
        ax.set_ylim(0, 1.22 if denominator else 1.05)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.set_axisbelow(True)
        ax.yaxis.grid(True, color="0.9", linewidth=0.6)
    axes[-1, 0].set_xticks(x)
    axes[-1, 0].set_xticklabels(scopes)
    axes[0, 0].legend(frameon=False, fontsize=8)
    figure.suptitle(_chart_title(cell_type, title), fontsize=12)
    figure.tight_layout()
    return figure


def plot_precision(
    scores: dict[str, dict[str, dict[str, Recovery]]],
    cell_type: str,
    scopes: Sequence[str],
) -> Figure:
    return _plot_rate_metric(
        scores,
        cell_type,
        scopes,
        "precision",
        "silver-standard precision (recovered / hits)",
        "Precision",
        denominator="n_hits",
    )


def plot_f1(
    scores: dict[str, dict[str, dict[str, Recovery]]],
    cell_type: str,
    scopes: Sequence[str],
) -> Figure:
    return _plot_rate_metric(
        scores, cell_type, scopes, "f1", "silver-standard F1", "F1"
    )


def plot_overlap(
    gene_overlap: dict[str, dict[str, float]],
    block_overlap: dict[str, dict[str, float]],
    cell_type: str,
    scopes: Sequence[str],
) -> Figure:
    """How the two arms share recovered silver-standard items."""
    figure, axes = plt.subplots(1, 2, figsize=(9.6, 4.2), sharey=True)
    categories = (
        ("n_recovered_both", "Both", "#4f7f5f"),
        ("n_recovered_this_study_only", "This study only", ARM_COLORS["this-study"]),
        ("n_recovered_ctPred_only", "ctPred only", ARM_COLORS["ctPred"]),
    )
    panels = (
        (axes[0], gene_overlap, "Genes"),
        (axes[1], block_overlap, "LD blocks"),
    )
    x = np.arange(len(scopes))
    width = 0.24
    offsets = (-width, 0.0, width)
    for ax, data, title in panels:
        for offset, (key, label, color) in zip(offsets, categories):
            values = [data[scope][key] for scope in scopes]
            ax.bar(x + offset, values, width=width * 0.95, color=color, label=label)
        ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels(scopes)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.set_axisbelow(True)
        ax.yaxis.grid(True, color="0.9", linewidth=0.6)
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    axes[0].set_ylabel("Recovered silver-standard items")
    axes[1].legend(frameon=False, fontsize=8)
    figure.suptitle(
        _chart_title(cell_type, "arm overlap on recovered silver standard")
    )
    figure.tight_layout()
    return figure


def _finite_pvalues(pvalues: np.ndarray | pd.Series) -> np.ndarray:
    values = np.asarray(pvalues, dtype=float)
    return values[np.isfinite(values) & (values > 0) & (values <= 1.0)]


def _neg_log10(pvalues: np.ndarray) -> np.ndarray:
    with np.errstate(divide="ignore", invalid="ignore"):
        values = -np.log10(pvalues)
    return np.clip(values, None, MAX_LOG10P)


def _qq_coordinates(pvalues: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Blom expected uniform quantiles vs observed -log10 p, sorted."""
    p = np.sort(_finite_pvalues(pvalues))
    if p.size == 0:
        empty = np.asarray([], dtype=float)
        return empty, empty
    expected = _neg_log10((np.arange(1, p.size + 1) - 0.5) / p.size)
    return expected, _neg_log10(p)


def _qq_stats(pvalues: np.ndarray | pd.Series) -> tuple[int, float]:
    values = _finite_pvalues(pvalues)
    return int(values.size), genomic_inflation(values)


def _bonferroni_log10(n: int, alpha: float = QQ_BONFERRONI_ALPHA) -> float:
    if n <= 0:
        return float("nan")
    return -math.log10(alpha / n)


def _silver_pvalue_mask(
    gene_keys: pd.Index,
    gene_names: dict[str, str],
    silver: SilverStandard,
) -> np.ndarray:
    """True for tested genes that match a silver-standard Ensembl id or symbol."""
    keys = pd.Index(gene_keys.astype(str))
    in_keys = np.asarray(keys.isin(silver.gene_keys), dtype=bool)
    silver_symbols = {
        name.upper() for name in silver.gene_names.values() if name
    }
    by_symbol = np.fromiter(
        (
            gene_names.get(key, "").upper() in silver_symbols
            for key in keys
        ),
        dtype=bool,
        count=len(keys),
    )
    return in_keys | by_symbol


def _split_silver_pvalues(
    pvalues: pd.Series,
    gene_names: dict[str, str],
    silver: SilverStandard,
) -> tuple[np.ndarray, np.ndarray]:
    if pvalues.empty:
        empty = np.asarray([], dtype=float)
        return empty, empty
    mask = _silver_pvalue_mask(pvalues.index, gene_names, silver)
    values = pvalues.to_numpy(dtype=float)
    return values[mask], values[~mask]


def _acat_arm_pvalues(
    items: Sequence[ArmHits],
) -> tuple[pd.Series, dict[str, str]]:
    """ACAT-combine p-values per gene across cell types for one arm."""
    names: dict[str, str] = {}
    series = []
    for item in items:
        names.update(item.gene_names)
        if not item.pvalues.empty:
            series.append(item.pvalues.groupby(level=0).first())
    if not series:
        return pd.Series(dtype=float), names
    combined = acat_rows(pd.concat(series, axis=1)).dropna()
    combined.name = "pvalue"
    return combined, names


def _qq_title(cell_type: str, arm: str) -> str:
    label = ARM_LABELS.get(arm, arm)
    if cell_type == BULK_LABEL:
        return f"{label}: bulk (ACAT across cell types)"
    return f"{label}: {cell_type}"


def plot_qq(pvalues: np.ndarray, cell_type: str, arm: str) -> Figure | None:
    """Standard GWAS-style QQ of one arm's gene-level p-values."""
    expected, observed = _qq_coordinates(pvalues)
    if expected.size == 0:
        return None
    n, lambda_gc = _qq_stats(pvalues)
    figure, ax = plt.subplots(figsize=(6.4, 5.6))
    ax.scatter(
        expected,
        observed,
        s=10,
        alpha=0.75,
        color="C0",
        edgecolors="none",
        zorder=2,
    )
    limit = max(float(expected.max()), float(observed.max()), 1.0) * 1.05
    x_max = float(expected.max()) * 1.05
    y_max = float(observed.max()) * 1.05 if observed.size else 1.0
    ax.plot([0, limit], [0, limit], linestyle="--", color="C0", linewidth=1.2, zorder=1)
    ax.set_xlim(0, x_max)
    ax.set_ylim(0, y_max)
    ax.set_xlabel(r"Expected $-\log_{10}(p)$")
    ax.set_ylabel(r"Observed $-\log_{10}(p)$")
    ax.set_title(_qq_title(cell_type, arm))
    ax.text(
        0.03,
        0.97,
        f"N = {n:,}\n$\\lambda_{{GC}}$ = {lambda_gc:.3f}",
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=10,
    )
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    figure.tight_layout()
    return figure


def plot_qq_silver(
    silver_pvalues: np.ndarray,
    other_pvalues: np.ndarray,
    cell_type: str,
    arm: str,
) -> Figure | None:
    """QQ of silver-standard genes vs all other tested genes, with Bonferroni lines."""
    silver_x, silver_y = _qq_coordinates(silver_pvalues)
    other_x, other_y = _qq_coordinates(other_pvalues)
    if silver_x.size == 0 and other_x.size == 0:
        return None

    figure, ax = plt.subplots(figsize=(6.8, 5.6))
    if other_x.size:
        ax.scatter(
            other_x,
            other_y,
            s=14,
            alpha=0.9,
            color=QQ_OTHER_COLOR,
            edgecolors="none",
            label="Other genes",
            zorder=2,
        )
    if silver_x.size:
        ax.scatter(
            silver_x,
            silver_y,
            s=16,
            alpha=0.95,
            color=QQ_SILVER_COLOR,
            edgecolors="none",
            label="Silver standard genes",
            zorder=3,
        )

    n_silver = int(silver_x.size)
    n_other = int(other_x.size)
    silver_thr = _bonferroni_log10(n_silver)
    other_thr = _bonferroni_log10(n_other)
    if math.isfinite(silver_thr):
        ax.axhline(
            silver_thr,
            color="0.45",
            linestyle="--",
            linewidth=1.1,
            label="Bonferroni (silver standard)",
            zorder=1,
        )
    if math.isfinite(other_thr):
        ax.axhline(
            other_thr,
            color="tab:cyan",
            linestyle="--",
            linewidth=1.1,
            label="Bonferroni (other genes)",
            zorder=1,
        )

    xs = [arr.max() for arr in (silver_x, other_x) if arr.size]
    ys = [arr.max() for arr in (silver_y, other_y) if arr.size]
    ax.set_xlim(0, (max(xs) if xs else 1.0) * 1.05)
    ax.set_ylim(0, (max(ys) if ys else 1.0) * 1.08)
    ax.set_xlabel(r"Expected $-\log_{10} p$")
    ax.set_ylabel(r"Observed $-\log_{10} p$")
    ax.set_title(_qq_title(cell_type, arm))
    handles, labels = ax.get_legend_handles_labels()
    order = []
    for wanted in (
        "Silver standard genes",
        "Bonferroni (silver standard)",
        "Other genes",
        "Bonferroni (other genes)",
    ):
        if wanted in labels:
            order.append(labels.index(wanted))
    if order:
        ax.legend(
            [handles[i] for i in order],
            [labels[i] for i in order],
            frameon=False,
            fontsize=8,
            loc="upper left",
        )
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    figure.tight_layout()
    return figure


def _comparison_qq_title(cell_type: str) -> str:
    if cell_type == BULK_LABEL:
        return "bulk (ACAT across cell types) — this study vs ctPred"
    return f"{cell_type} — this study vs ctPred"


def _matched_log_quantiles(
    ours: np.ndarray, theirs: np.ndarray
) -> tuple[np.ndarray, np.ndarray, int, int]:
    """ctPred (x) vs this-study (y) matched -log10 p quantiles."""
    ours_p = _finite_pvalues(ours)
    theirs_p = _finite_pvalues(theirs)
    x, y = two_sample_quantiles(_neg_log10(theirs_p), _neg_log10(ours_p))
    return x, y, int(ours_p.size), int(theirs_p.size)


def plot_qq_comparison(
    ours_silver: np.ndarray,
    theirs_silver: np.ndarray,
    ours_other: np.ndarray,
    theirs_other: np.ndarray,
    cell_type: str,
) -> Figure | None:
    """Quantile-quantile plot of this-study vs ctPred, silver vs other genes."""
    silver_x, silver_y, n_ours_silver, n_theirs_silver = _matched_log_quantiles(
        ours_silver, theirs_silver
    )
    other_x, other_y, n_ours_other, n_theirs_other = _matched_log_quantiles(
        ours_other, theirs_other
    )
    if silver_x.size == 0 and other_x.size == 0:
        return None

    xs = [arr.max() for arr in (silver_x, other_x, silver_y, other_y) if arr.size]
    limit = (max(xs) if xs else 1.0) * 1.05
    figure, ax = plt.subplots(figsize=(6.4, 5.6))
    ax.plot(
        [0, limit], [0, limit], linestyle="--", color="0.35", linewidth=1.2, zorder=1
    )
    if other_x.size:
        ax.scatter(
            other_x,
            other_y,
            s=12,
            alpha=0.8,
            color=QQ_OTHER_COLOR,
            edgecolors="none",
            label="Other genes",
            zorder=2,
        )
    if silver_x.size:
        ax.scatter(
            silver_x,
            silver_y,
            s=16,
            alpha=0.95,
            color=QQ_SILVER_COLOR,
            edgecolors="none",
            label="Silver standard genes",
            zorder=3,
        )
    ax.set_xlim(0, limit)
    ax.set_ylim(0, limit)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel(rf"{ARM_LABELS['ctPred']}  $-\log_{{10}} p$")
    ax.set_ylabel(rf"{ARM_LABELS['this-study']}  $-\log_{{10}} p$")
    ax.set_title(_comparison_qq_title(cell_type))
    silver_above = float(np.mean(silver_y > silver_x)) if silver_y.size else float("nan")
    other_above = float(np.mean(other_y > other_x)) if other_y.size else float("nan")
    annotation_lines = []
    if silver_x.size:
        annotation_lines.append(
            f"Silver: {n_ours_silver:,} vs {n_theirs_silver:,}, "
            f"{silver_above:.0%} above y = x"
        )
    if other_x.size:
        annotation_lines.append(
            f"Other: {n_ours_other:,} vs {n_theirs_other:,}, "
            f"{other_above:.0%} above y = x"
        )
    if annotation_lines:
        ax.text(
            0.97,
            0.97,
            "\n".join(annotation_lines),
            transform=ax.transAxes,
            va="top",
            ha="right",
            fontsize=8,
        )
    handles, labels = ax.get_legend_handles_labels()
    order = []
    for wanted in ("Silver standard genes", "Other genes"):
        if wanted in labels:
            order.append(labels.index(wanted))
    if order:
        ax.legend(
            [handles[i] for i in order],
            [labels[i] for i in order],
            frameon=False,
            fontsize=8,
            loc="upper left",
        )
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    figure.tight_layout()
    return figure


def _arm_min_agreement(args: argparse.Namespace, arm: str) -> float | None:
    if arm != "this-study":
        return None
    return args.min_agreement


def _write_local(
    output_dir: Path | None,
    cell_type: str,
    arm: str,
    figures: dict[str, Figure],
    tables: dict[str, pd.DataFrame],
    dpi: int,
) -> None:
    if output_dir is None:
        return
    directory = output_dir / cell_type / arm
    directory.mkdir(parents=True, exist_ok=True)
    for key, figure in figures.items():
        path = directory / f"{key.replace('/', '_')}.png"
        figure.savefig(path, dpi=dpi, bbox_inches="tight")
        LOGGER.info("Wrote %s.", path)
    for key, table in tables.items():
        path = directory / f"{key.replace('/', '_')}.tsv"
        table.to_csv(path, sep="\t", index=False)


def _wandb_config(args: argparse.Namespace, arm: str, cell_type: str) -> dict:
    return {
        "arm": arm,
        "analysis": BENCHMARK_PREFIX,
        "cell_type": cell_type,
        "criterion": args.criterion,
        "combination": args.combination,
        "min_agreement": _arm_min_agreement(args, arm),
        "gene_scope": list(args.gene_scope),
        "gtf": str(args.gtf),
        "mhc_region": args.mhc_region,
        "input_dir": str(args.input_dir),
        "silver_standard": str(args.silver_standard),
    }


def score_cell_type(
    silver: SilverStandard,
    ours: ArmHits,
    theirs: ArmHits,
    scopes: Sequence[str],
) -> tuple[
    dict[str, dict[str, dict[str, Recovery]]],
    dict[str, dict[str, float]],
    dict[str, dict[str, float]],
    set[str],
]:
    universe = shared_universe(ours, theirs)
    scores: dict[str, dict[str, dict[str, Recovery]]] = {
        "this-study": {},
        "ctPred": {},
    }
    gene_overlap: dict[str, dict[str, float]] = {}
    block_overlap: dict[str, dict[str, float]] = {}
    for scope in scopes:
        ours_genes = recover_genes(ours, silver, scope, universe=universe)
        theirs_genes = recover_genes(theirs, silver, scope, universe=universe)
        ours_blocks = recover_blocks(ours, silver, scope, universe=universe)
        theirs_blocks = recover_blocks(theirs, silver, scope, universe=universe)
        scores["this-study"][scope] = {"genes": ours_genes, "ld_blocks": ours_blocks}
        scores["ctPred"][scope] = {"genes": theirs_genes, "ld_blocks": theirs_blocks}
        gene_overlap[scope] = _overlap_metrics(ours_genes, theirs_genes)
        block_overlap[scope] = _overlap_metrics(ours_blocks, theirs_blocks)
    return scores, gene_overlap, block_overlap, universe


def _resolve_pvalues(
    arm: str,
    hits: dict[str, ArmHits] | None,
    pvalues_by_arm: dict[str, pd.Series] | None,
    gene_names_by_arm: dict[str, dict[str, str]] | None,
) -> tuple[pd.Series | None, dict[str, str] | None]:
    if pvalues_by_arm is not None and arm in pvalues_by_arm:
        return pvalues_by_arm[arm], (gene_names_by_arm or {}).get(arm)
    if hits is not None and arm in hits:
        return hits[arm].pvalues, hits[arm].gene_names
    return None, None


def _add_arm_qq(
    summary: dict[str, int | float],
    figures: dict[str, Figure],
    pvalues: pd.Series,
    gene_names: dict[str, str],
    silver: SilverStandard | None,
    cell_type: str,
    arm: str,
) -> None:
    n_genes, lambda_gc = _qq_stats(pvalues)
    summary[_log_key("qq", "n")] = n_genes
    summary[_log_key("qq", "lambda_gc")] = lambda_gc
    qq_figure = plot_qq(pvalues.to_numpy(dtype=float), cell_type, arm)
    if qq_figure is not None:
        figures[f"{BENCHMARK_PREFIX}/qq"] = qq_figure
    LOGGER.info(
        "%s %s QQ: N=%d, lambda_GC=%.3f.",
        cell_type,
        arm,
        n_genes,
        lambda_gc,
    )
    if silver is None:
        return
    silver_p, other_p = _split_silver_pvalues(pvalues, gene_names, silver)
    n_silver, lambda_silver = _qq_stats(silver_p)
    n_other, lambda_other = _qq_stats(other_p)
    summary[_log_key("qq", "n_silver")] = n_silver
    summary[_log_key("qq", "lambda_gc_silver")] = lambda_silver
    summary[_log_key("qq", "n_other")] = n_other
    summary[_log_key("qq", "lambda_gc_other")] = lambda_other
    split_figure = plot_qq_silver(silver_p, other_p, cell_type, arm)
    if split_figure is not None:
        figures[f"{BENCHMARK_PREFIX}/qq_silver"] = split_figure


def _add_comparison_qq(
    summary: dict[str, int | float],
    figures: dict[str, Figure],
    ours: pd.Series,
    theirs: pd.Series,
    ours_names: dict[str, str],
    theirs_names: dict[str, str],
    silver: SilverStandard,
    cell_type: str,
) -> None:
    ours_silver, ours_other = _split_silver_pvalues(ours, ours_names, silver)
    theirs_silver, theirs_other = _split_silver_pvalues(theirs, theirs_names, silver)
    subsets = (
        ("qq_silver", ours_silver, theirs_silver),
        ("qq_other", ours_other, theirs_other),
    )
    for key, ours_p, theirs_p in subsets:
        x, y, n_ours, n_theirs = _matched_log_quantiles(ours_p, theirs_p)
        above = float(np.mean(y > x)) if y.size else float("nan")
        summary[_log_key("comparison", f"{key}_n_this_study")] = n_ours
        summary[_log_key("comparison", f"{key}_n_ctPred")] = n_theirs
        summary[_log_key("comparison", f"{key}_frac_above")] = above
    figure = plot_qq_comparison(
        ours_silver,
        theirs_silver,
        ours_other,
        theirs_other,
        cell_type,
    )
    if figure is not None:
        figures[f"{BENCHMARK_PREFIX}/comparison/qq"] = figure


def _arm_payload(
    arm: str,
    cell_type: str,
    scores: dict[str, dict[str, dict[str, Recovery]]],
    gene_overlap: dict[str, dict[str, float]],
    block_overlap: dict[str, dict[str, float]],
    hits: dict[str, ArmHits] | None,
    silver: SilverStandard | None,
    scopes: Sequence[str],
    universe_size: float,
    pvalues: pd.Series | None = None,
    pvalue_gene_names: dict[str, str] | None = None,
    other_pvalues: pd.Series | None = None,
    other_pvalue_gene_names: dict[str, str] | None = None,
    include_shared: bool = True,
) -> tuple[dict[str, int | float], dict[str, Figure], dict[str, pd.DataFrame]]:
    summary: dict[str, int | float] = {}
    tables: dict[str, pd.DataFrame] = {}
    figures: dict[str, Figure] = {}
    this_hits = hits[arm] if hits is not None else None
    if include_shared:
        summary[_log_key("shared", "n_tested_genes")] = universe_size
    for scope in scopes:
        gene = scores[arm][scope]["genes"]
        block = scores[arm][scope]["ld_blocks"]
        summary.update(_recovery_metrics(_log_key(scope, "gene_"), gene))
        summary.update(_recovery_metrics(_log_key(scope, "ld_block_"), block))
        if include_shared:
            for key, value in gene_overlap[scope].items():
                summary[_log_key(scope, f"gene_{key}")] = value
            for key, value in block_overlap[scope].items():
                summary[_log_key(scope, f"ld_block_{key}")] = value
            tables[_log_key(scope, "metrics")] = _metrics_table(
                cell_type, scope, {name: scores[name][scope] for name in ARMS}
            )
        if this_hits is None or silver is None:
            continue
        tables[_log_key(scope, "recovered_genes")] = _item_table(
            gene.recovered,
            this_hits.gene_names,
            kind="gene",
            status="recovered",
            blocks=this_hits.gene_blocks,
            block_labels=this_hits.block_labels,
            silver_names=silver.gene_names,
        )
        tables[_log_key(scope, "missed_genes")] = _item_table(
            gene.missed,
            this_hits.gene_names,
            kind="gene",
            status="missed",
            blocks=this_hits.gene_blocks,
            block_labels=this_hits.block_labels,
            silver_names=silver.gene_names,
        )
        tables[_log_key(scope, "recovered_blocks")] = _item_table(
            block.recovered,
            {},
            kind="ld_block",
            status="recovered",
            blocks={key: int(key) for key in block.recovered if str(key).lstrip("-").isdigit()},
            block_labels=silver.blocks,
        )
        tables[_log_key(scope, "missed_blocks")] = _item_table(
            block.missed,
            {},
            kind="ld_block",
            status="missed",
            blocks={key: int(key) for key in block.missed if str(key).lstrip("-").isdigit()},
            block_labels=silver.blocks,
        )
    if include_shared:
        figures[f"{BENCHMARK_PREFIX}/comparison/recovery"] = plot_recovery(
            scores, cell_type, scopes
        )
        figures[f"{BENCHMARK_PREFIX}/comparison/precision"] = plot_precision(
            scores, cell_type, scopes
        )
        figures[f"{BENCHMARK_PREFIX}/comparison/f1"] = plot_f1(
            scores, cell_type, scopes
        )
        figures[f"{BENCHMARK_PREFIX}/comparison/overlap"] = plot_overlap(
            gene_overlap, block_overlap, cell_type, scopes
        )
    if pvalues is None and this_hits is not None:
        pvalues = this_hits.pvalues
        pvalue_gene_names = this_hits.gene_names
    if pvalues is not None and not pvalues.empty:
        _add_arm_qq(
            summary,
            figures,
            pvalues,
            pvalue_gene_names or {},
            silver,
            cell_type,
            arm,
        )
    if (
        include_shared
        and silver is not None
        and pvalues is not None
        and not pvalues.empty
        and other_pvalues is not None
        and not other_pvalues.empty
    ):
        _add_comparison_qq(
            summary,
            figures,
            pvalues,
            other_pvalues,
            pvalue_gene_names or {},
            other_pvalue_gene_names or {},
            silver,
            cell_type,
        )
    return summary, figures, tables


def _log_scores(
    cell_type: str,
    scores: dict[str, dict[str, dict[str, Recovery]]],
    *,
    n_cell_types: int | None = None,
) -> None:
    log_scope = "all" if "all" in scores["this-study"] else next(iter(scores["this-study"]))
    ours = scores["this-study"][log_scope]
    theirs = scores["ctPred"][log_scope]
    if n_cell_types is None:
        LOGGER.info(
            "%s silver recovery (%s): this-study %.3g genes / %.3g blocks; "
            "ctPred %.3g genes / %.3g blocks.",
            cell_type,
            log_scope,
            ours["genes"].n_recovered,
            ours["ld_blocks"].n_recovered,
            theirs["genes"].n_recovered,
            theirs["ld_blocks"].n_recovered,
        )
        return
    LOGGER.info(
        "Bulk mean across %d cell type(s) (%s): this-study %.3g genes / "
        "%.3g blocks; ctPred %.3g genes / %.3g blocks.",
        n_cell_types,
        log_scope,
        ours["genes"].n_recovered,
        ours["ld_blocks"].n_recovered,
        theirs["genes"].n_recovered,
        theirs["ld_blocks"].n_recovered,
    )


def _emit_from_scores(
    *,
    cell_type: str,
    scores: dict[str, dict[str, dict[str, Recovery]]],
    gene_overlap: dict[str, dict[str, float]],
    block_overlap: dict[str, dict[str, float]],
    hits: dict[str, ArmHits] | None,
    silver: SilverStandard | None,
    args: argparse.Namespace,
    logger: TwasWandBLogger,
    universe_size: float,
    extra_config: dict | None = None,
    pvalues_by_arm: dict[str, pd.Series] | None = None,
    gene_names_by_arm: dict[str, dict[str, str]] | None = None,
) -> None:
    for arm in ARMS:
        include_shared = arm == SHARED_ARM
        pvalues, names = _resolve_pvalues(
            arm, hits, pvalues_by_arm, gene_names_by_arm
        )
        other_pvalues, other_names = (None, None)
        if include_shared:
            other_pvalues, other_names = _resolve_pvalues(
                "ctPred", hits, pvalues_by_arm, gene_names_by_arm
            )
        summary, figures, tables = _arm_payload(
            arm,
            cell_type,
            scores,
            gene_overlap,
            block_overlap,
            hits,
            silver,
            args.gene_scope,
            universe_size,
            pvalues=pvalues,
            pvalue_gene_names=names,
            other_pvalues=other_pvalues,
            other_pvalue_gene_names=other_names,
            include_shared=include_shared,
        )
        _write_local(args.output_dir, cell_type, arm, figures, tables, args.dpi)
        run_name = f"benchmark-{arm}/{cell_type}"
        config = _wandb_config(args, arm, cell_type)
        if extra_config:
            config.update(extra_config)
        logger.start(cell_type, config=config, name=run_name)
        try:
            logger.log_results(summary, figures, tables=tables)
            LOGGER.info("Logged Benchmark artifacts to WandB run %r.", run_name)
        finally:
            logger.finish()
            for figure in figures.values():
                plt.close(figure)


def _emit_benchmark(
    *,
    cell_type: str,
    ours: ArmHits,
    theirs: ArmHits,
    silver: SilverStandard,
    args: argparse.Namespace,
    logger: TwasWandBLogger,
) -> tuple[
    dict[str, dict[str, dict[str, Recovery]]],
    dict[str, dict[str, float]],
    dict[str, dict[str, float]],
    int,
]:
    scores, gene_overlap, block_overlap, universe = score_cell_type(
        silver, ours, theirs, args.gene_scope
    )
    hits = {"this-study": ours, "ctPred": theirs}
    ours_shared = ours.hit_genes & universe
    theirs_shared = theirs.hit_genes & universe
    LOGGER.info(
        "%s: %d genes tested by both arms. Shared-universe hits: "
        "this-study %d genes / %d blocks; ctPred %d genes / %d blocks.",
        cell_type,
        len(universe),
        len(ours_shared),
        len({ours.gene_blocks[g] for g in ours_shared if g in ours.gene_blocks}),
        len(theirs_shared),
        len({theirs.gene_blocks[g] for g in theirs_shared if g in theirs.gene_blocks}),
    )
    _log_scores(cell_type, scores)
    _emit_from_scores(
        cell_type=cell_type,
        scores=scores,
        gene_overlap=gene_overlap,
        block_overlap=block_overlap,
        hits=hits,
        silver=silver,
        args=args,
        logger=logger,
        universe_size=float(len(universe)),
    )
    return scores, gene_overlap, block_overlap, len(universe)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s: %(message)s",
    )
    try:
        paired = paired_cell_types(args.input_dir, args.cell_types)
        mhc_gene_ids: set[str] | None = None
        if args.gtf.is_file():
            _, mhc_gene_ids = load_mhc_gene_ids(args.gtf, args.mhc_region)
        else:
            LOGGER.warning(
                "GTF %s is missing; MHC gene membership falls back to LD-block overlap.",
                args.gtf,
            )
        silver = load_silver_standard(
            args.silver_standard, args.mhc_region, mhc_gene_ids
        )
        logger = TwasWandBLogger(
            project=args.wandb_project, entity=args.wandb_entity
        )
        collected_scores: list[dict[str, dict[str, dict[str, Recovery]]]] = []
        collected_gene_overlap: list[dict[str, dict[str, float]]] = []
        collected_block_overlap: list[dict[str, dict[str, float]]] = []
        universe_sizes: list[float] = []
        collected_ours: list[ArmHits] = []
        collected_theirs: list[ArmHits] = []
        for cell_type, ours_path, theirs_path in paired:
            ours = load_arm_hits(
                ours_path,
                cell_type,
                "this-study",
                args.criterion,
                args.combination,
                _arm_min_agreement(args, "this-study"),
                args.mhc_region,
                mhc_gene_ids,
            )
            theirs = load_arm_hits(
                theirs_path,
                cell_type,
                "ctPred",
                args.criterion,
                args.combination,
                None,
                args.mhc_region,
                mhc_gene_ids,
            )
            scores, gene_overlap, block_overlap, n_universe = _emit_benchmark(
                cell_type=cell_type,
                ours=ours,
                theirs=theirs,
                silver=silver,
                args=args,
                logger=logger,
            )
            collected_scores.append(scores)
            collected_gene_overlap.append(gene_overlap)
            collected_block_overlap.append(block_overlap)
            universe_sizes.append(float(n_universe))
            collected_ours.append(ours)
            collected_theirs.append(theirs)
        mean_scores = average_cell_type_scores(collected_scores, args.gene_scope)
        mean_gene_overlap = _average_overlap(
            collected_gene_overlap, args.gene_scope
        )
        mean_block_overlap = _average_overlap(
            collected_block_overlap, args.gene_scope
        )
        ours_pvalues, ours_names = _acat_arm_pvalues(collected_ours)
        theirs_pvalues, theirs_names = _acat_arm_pvalues(collected_theirs)
        _log_scores(BULK_LABEL, mean_scores, n_cell_types=len(paired))
        _emit_from_scores(
            cell_type=BULK_LABEL,
            scores=mean_scores,
            gene_overlap=mean_gene_overlap,
            block_overlap=mean_block_overlap,
            hits=None,
            silver=silver,
            args=args,
            logger=logger,
            universe_size=_mean(universe_sizes),
            extra_config={"n_cell_types": len(paired), "bulk": True},
            pvalues_by_arm={
                "this-study": ours_pvalues,
                "ctPred": theirs_pvalues,
            },
            gene_names_by_arm={
                "this-study": ours_names,
                "ctPred": theirs_names,
            },
        )
    except (FileNotFoundError, NotADirectoryError, OSError, ValueError) as error:
        LOGGER.error("%s", error)
        raise SystemExit(1) from error


if __name__ == "__main__":
    main()
