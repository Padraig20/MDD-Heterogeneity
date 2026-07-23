#!/usr/bin/env python3

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp


"""
preprocess_data.py

Build OneK1K per-cell-type expression tables from a single-cell h5ad file.

Cells are first aggregated into a per-(individual, cell type) pseudo-bulk
(summed counts plus a per-group cell count). A per-cell-type gene filter
(filterByExpr-style expression filter plus a near-zero variance filter) is then
applied. Two output flavours are available, selected with
--aggregate-across-individuals:

  * default (no flag) -> ground truth. One CSV per cell type with rows=genes and
    columns:

        gene,chrom,tss,<individual 1>,<individual 2>,...

    Values are the per-(individual, cell type) mean expression (counts per
    cell), optionally min-max normalized per gene across individuals with
    --normalize. The `gene` column is the ENSID from the gene-metadata CSV.

  * --aggregate-across-individuals -> target variables. A single CSV with
    rows=cell types and columns=genes. Values are the cell-weighted mean
    expression across a donor subset (the additional aggregation across
    individuals). With --normalize, per-donor mean expression is min-max
    normalized per gene before the additional donor aggregation. With
    --percentiles, the mean expressions are instead ranked across genes within
    each cell type after donor aggregation.

  * --population-h5ad -> population target. A per-(individual, cell type)
    pseudo-bulk .h5ad (obs: individual_id, cell_type, n_cells; var.index: gene
    IDs; X: mean expression, counts per cell). Every donor in the subset is kept
    as a separate row -- no averaging across individuals -- so a heteroscedastic
    / deep-ensemble model trained with the reference features can learn per-gene
    population variance (the "version a" uncertainty setup). Restricted to the
    training donor subset by default (same as --aggregate-across-individuals;
    override with --target-individual / --all-individuals) so the held-out
    evaluation cohort does not leak into the variance estimate. Gene filters are
    not applied here; the target stays dense and gene selection happens via the
    reference feature set at training time.
"""


# OneK1K obs columns are stable, so they are hardcoded rather than exposed.
INDIVIDUAL_COL = "donor_id"
CELL_TYPE_COL = "celltype"
DEFAULT_LAYER = "counts"

REQUIRED_GENE_COLUMNS = ("chrom", "ensid", "tss")

# Default donor subset used for the aggregated target variables (the OneK1K
# training individuals).
INDIVIDUAL_IDS = [
    "1000_1001",
    "1001_1002",
    "1002_1003",
    "1003_1004",
    "1004_1005",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build OneK1K per-cell-type expression tables from a single-cell "
            "h5ad. Writes per-individual ground truth by default, or "
            "individual-aggregated target variables with "
            "--aggregate-across-individuals."
        )
    )
    parser.add_argument(
        "-i",
        "--input",
        type=Path,
        required=True,
        help=(
            f"Input single-cell h5ad. obs must contain {INDIVIDUAL_COL!r} and "
            f"{CELL_TYPE_COL!r} columns."
        ),
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        required=True,
        help=(
            "Output path. Default mode: a directory that will hold one CSV per "
            "cell type. With --aggregate-across-individuals: a single CSV file."
        ),
    )
    parser.add_argument(
        "-g",
        "--gene-metadata",
        type=Path,
        default=None,
        help=(
            "CSV with gene metadata (required columns: chrom, ensid, tss). "
            "Required for the default per-individual ground-truth output; "
            "ignored with --aggregate-across-individuals."
        ),
    )
    parser.add_argument(
        "--aggregate-across-individuals",
        action="store_true",
        help=(
            "Aggregate across individuals and write a single cell-type x gene "
            "target-variable CSV instead of per-individual ground-truth CSVs."
        ),
    )
    parser.add_argument(
        "--population-h5ad",
        action="store_true",
        help=(
            "Write a per-(individual, cell type) pseudo-bulk .h5ad (to -o/"
            "--output) instead of the per-individual ground-truth CSVs. This is "
            "the target file for the population-variance ('version a') training "
            "setup: it keeps every donor as a separate row so a heteroscedastic "
            "/ deep-ensemble model can learn per-gene population variance. "
            "Values are mean expression (counts per cell); obs has "
            "individual_id, cell_type and n_cells; var.index holds gene IDs. "
            "Gene filters are NOT applied (the target stays dense; gene "
            "selection happens via the reference feature set at training time). "
            "Requires neither --gene-metadata nor --aggregate-across-individuals."
        ),
    )

    matrix_group = parser.add_mutually_exclusive_group()
    matrix_group.add_argument(
        "--layer",
        default=DEFAULT_LAYER,
        help=f'Input layer to aggregate (raw counts). Default: "{DEFAULT_LAYER}".',
    )
    matrix_group.add_argument(
        "--use-x",
        action="store_true",
        help="Aggregate adata.X instead of a layer.",
    )

    parser.add_argument(
        "--min-cells",
        type=int,
        default=125,
        help=(
            "Pseudo-bulk quality filter: minimum number of cells a "
            "(donor, cell type) pseudo-bulk sample must contain. Samples with "
            "fewer cells are dropped. Set 0 to disable. Default: 125."
        ),
    )
    parser.add_argument(
        "--min-cpm",
        type=float,
        default=0.1,
        help=(
            "Per-cell-type expression filter: a gene must reach this CPM in at "
            "least --expr-min-prop of donors. CPM is computed within each "
            "(donor, cell type) pseudo-bulk sample. Default: 0.1."
        ),
    )
    parser.add_argument(
        "--min-reads",
        type=float,
        default=6.0,
        help=(
            "Per-cell-type expression filter: a gene must reach this many "
            "summed reads in at least --expr-min-prop of donors. Default: 6."
        ),
    )
    parser.add_argument(
        "--expr-min-prop",
        type=float,
        default=0.2,
        help=(
            "Fraction of donors (out of all donors in the input) that must "
            "satisfy both the CPM and read thresholds for a gene to be kept in "
            "a given cell type. Default: 0.2."
        ),
    )
    parser.add_argument(
        "--min-variance",
        type=float,
        default=1e-8,
        help=(
            "Per-cell-type variance filter: genes whose per-donor mean "
            "expression varies by this much or less across donors are dropped "
            "(near-zero variation). Default: 1e-8."
        ),
    )
    parser.add_argument(
        "--disable-gene-filters",
        action="store_true",
        help=(
            "Disable the per-cell-type expression and variance gene filters. "
            "The --min-cells pseudo-bulk QC still applies; pass --min-cells 0 "
            "to disable that as well."
        ),
    )
    parser.add_argument(
        "--keep-empty",
        action="store_true",
        help=(
            "Keep cell types and genes that are entirely empty (all NaN) after "
            "filtering. By default empty genes are dropped, and empty cell "
            "types are dropped (ground-truth CSVs are skipped, aggregated rows "
            "are removed)."
        ),
    )
    parser.add_argument(
        "--normalize",
        action="store_true",
        help=(
            "Normalize each gene's expression values across individuals to the "
            "0-1 range before writing output. Disabled by default."
        ),
    )
    parser.add_argument(
        "--percentiles",
        action="store_true",
        help=(
            "With --aggregate-across-individuals, convert the aggregated mean "
            "expression values to percentile ranks across genes, separately "
            "for each cell type. Donors are aggregated before ranks are "
            "computed. Ties receive their average rank. Cannot be combined "
            "with --normalize."
        ),
    )

    parser.add_argument(
        "--target-individual",
        action="append",
        default=None,
        help=(
            "Donor ID to include in the donor subset. May be passed multiple "
            "times. Defaults to the OneK1K training individuals. Used with "
            "--aggregate-across-individuals and --population-h5ad (both restrict "
            "to this subset by default); ignored for the per-individual "
            "ground-truth CSVs, which always keep the full cohort."
        ),
    )
    parser.add_argument(
        "--all-individuals",
        action="store_true",
        help=(
            "For --aggregate-across-individuals / --population-h5ad, use every "
            "donor in the input instead of the default training individuals."
        ),
    )
    parser.add_argument(
        "--missing-individuals",
        choices=["error", "drop"],
        default="error",
        help=(
            "What to do if requested donors are missing. Default: error. "
            "Ignored with --all-individuals."
        ),
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing output files.",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="count",
        default=0,
        help="Increase verbosity.",
    )

    args = parser.parse_args()

    if args.aggregate_across_individuals and args.population_h5ad:
        parser.error(
            "--population-h5ad cannot be combined with "
            "--aggregate-across-individuals; they are mutually exclusive output "
            "modes."
        )
    if (
        not args.aggregate_across_individuals
        and not args.population_h5ad
        and args.gene_metadata is None
    ):
        parser.error(
            "--gene-metadata is required for the default per-individual "
            "ground-truth output (omit it only with "
            "--aggregate-across-individuals or --population-h5ad)."
        )
    if args.all_individuals and args.target_individual is not None:
        parser.error("--all-individuals cannot be combined with --target-individual.")
    if args.percentiles and not args.aggregate_across_individuals:
        parser.error(
            "--percentiles is only available with --aggregate-across-individuals."
        )
    if args.percentiles and args.normalize:
        parser.error("--percentiles cannot be combined with --normalize.")

    return args


def setup_logging(verbosity: int) -> None:
    level = logging.DEBUG if verbosity >= 1 else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )


# ---------------------------------------------------------------------------
# Loading and pseudo-bulk aggregation
# ---------------------------------------------------------------------------


def validate_obs_columns(obs: pd.DataFrame) -> None:
    missing = [c for c in (INDIVIDUAL_COL, CELL_TYPE_COL) if c not in obs.columns]
    if missing:
        raise ValueError(
            f"Input h5ad obs is missing required column(s): {missing}. "
            f"Available obs columns: {obs.columns.tolist()}"
        )


def load_matrix(adata: ad.AnnData, layer: str, use_x: bool):
    if use_x:
        logging.info("Using adata.X for expression values")
        return adata.X

    if layer not in adata.layers:
        raise KeyError(
            f"Layer {layer!r} not found in input h5ad. Available layers: "
            f"{list(adata.layers.keys())}. Pass --use-x to aggregate adata.X."
        )
    logging.info("Using adata.layers[%r] for expression values", layer)
    return adata.layers[layer]


def prepare_grouping(obs: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
    keep = obs[INDIVIDUAL_COL].notna() & obs[CELL_TYPE_COL].notna()
    if not bool(keep.all()):
        logging.warning(
            "Dropping %d cells with missing %s or %s",
            int((~keep).sum()),
            INDIVIDUAL_COL,
            CELL_TYPE_COL,
        )

    grouping = pd.DataFrame(
        {
            "individual_id": obs.loc[keep, INDIVIDUAL_COL].astype(str).to_numpy(),
            "cell_type": obs.loc[keep, CELL_TYPE_COL].astype(str).to_numpy(),
        }
    )
    return grouping, keep.to_numpy()


def aggregate_sparse(
    X: sp.spmatrix,
    grouping: pd.DataFrame,
) -> tuple[sp.csr_matrix, pd.DataFrame]:
    grouped = grouping.groupby(["individual_id", "cell_type"], sort=True)
    keys = list(grouped.indices.keys())
    n_groups = len(keys)
    n_cells = len(grouping)

    group_codes = np.empty(n_cells, dtype=np.int64)
    n_cells_per_group = np.empty(n_groups, dtype=np.int64)
    for code, key in enumerate(keys):
        idx = np.asarray(grouped.indices[key], dtype=np.int64)
        group_codes[idx] = code
        n_cells_per_group[code] = len(idx)

    indicator = sp.csr_matrix(
        (
            np.ones(n_cells, dtype=np.float32),
            (group_codes, np.arange(n_cells, dtype=np.int64)),
        ),
        shape=(n_groups, n_cells),
    )

    sums = (indicator @ X).tocsr()
    sums = sums.astype(np.float32, copy=False)

    obs = pd.DataFrame(keys, columns=["individual_id", "cell_type"])
    obs["n_cells"] = n_cells_per_group
    return sums, obs


def aggregate_dense(
    X: np.ndarray,
    grouping: pd.DataFrame,
) -> tuple[np.ndarray, pd.DataFrame]:
    grouped = grouping.groupby(["individual_id", "cell_type"], sort=True)
    keys = list(grouped.indices.keys())
    sums = np.zeros((len(keys), X.shape[1]), dtype=np.float32)
    n_cells_per_group = np.empty(len(keys), dtype=np.int64)

    for code, key in enumerate(keys):
        idx = np.asarray(grouped.indices[key], dtype=np.int64)
        sums[code] = np.asarray(X[idx, :].sum(axis=0)).ravel()
        n_cells_per_group[code] = len(idx)

    obs = pd.DataFrame(keys, columns=["individual_id", "cell_type"])
    obs["n_cells"] = n_cells_per_group
    return sums, obs


def aggregate_by_individual_and_cell_type(
    X,
    grouping: pd.DataFrame,
) -> tuple[sp.csr_matrix | np.ndarray, pd.DataFrame]:
    if grouping.empty:
        raise ValueError("No cells remain after filtering missing donor/cell-type labels.")

    logging.info(
        "Aggregating %d cells into donor/cell-type pseudo-bulk groups",
        len(grouping),
    )
    if sp.issparse(X):
        return aggregate_sparse(X.tocsr(), grouping)
    return aggregate_dense(np.asarray(X), grouping)


def row_to_dense(row: sp.spmatrix | np.ndarray) -> np.ndarray:
    if sp.issparse(row):
        return np.asarray(row.toarray()).ravel()
    return np.asarray(row).ravel()


# ---------------------------------------------------------------------------
# Gene metadata (ground-truth mode only)
# ---------------------------------------------------------------------------


def load_gene_metadata(path: Path) -> pd.DataFrame:
    logging.info("Reading gene metadata from %s", path)
    gene_meta = pd.read_csv(path, usecols=list(REQUIRED_GENE_COLUMNS))

    missing = [col for col in REQUIRED_GENE_COLUMNS if col not in gene_meta.columns]
    if missing:
        raise ValueError(
            f"Gene metadata is missing required column(s): {missing}. "
            f"Required columns: {list(REQUIRED_GENE_COLUMNS)}"
        )

    gene_meta = gene_meta.rename(columns={"ensid": "gene"})
    gene_meta["gene"] = gene_meta["gene"].astype(str)
    gene_meta["chrom"] = gene_meta["chrom"].astype(str)
    gene_meta = gene_meta[["gene", "chrom", "tss"]]

    if gene_meta.empty:
        raise ValueError(f"Gene metadata file is empty: {path}")

    return gene_meta


def align_gene_metadata_to_var(
    gene_meta: pd.DataFrame,
    var_index: pd.Index,
) -> tuple[pd.DataFrame, np.ndarray]:
    genes = var_index.astype(str).to_numpy()
    duplicated = pd.Index(genes).duplicated()
    if bool(duplicated.any()):
        examples = pd.Index(genes)[duplicated].unique().tolist()[:10]
        raise ValueError(
            "Input h5ad var.index contains duplicate ENSIDs, so expression "
            f"columns cannot be mapped unambiguously. Examples: {examples}"
        )

    gene_to_pos = {gene: pos for pos, gene in enumerate(genes)}
    in_var = gene_meta["gene"].map(gene_to_pos).notna()
    n_missing = int((~in_var).sum())
    if n_missing:
        missing_examples = gene_meta.loc[~in_var, "gene"].head(10).tolist()
        logging.warning(
            "Dropping %d / %d genes from metadata because they are absent from "
            "h5ad var.index. Examples: %s",
            n_missing,
            len(gene_meta),
            missing_examples,
        )

    aligned = gene_meta.loc[in_var].reset_index(drop=True)
    if aligned.empty:
        raise ValueError("No genes from the metadata CSV were found in h5ad var.index.")

    var_positions = np.fromiter(
        (gene_to_pos[gene] for gene in aligned["gene"]),
        dtype=np.int64,
        count=len(aligned),
    )
    logging.info("Using %d genes present in both metadata CSV and h5ad", len(aligned))
    return aligned, var_positions


# ---------------------------------------------------------------------------
# Per-cell-type gene filtering (shared by both modes)
# ---------------------------------------------------------------------------


def compute_gene_keep_mask(
    sums: sp.csr_matrix | np.ndarray,
    obs: pd.DataFrame,
    n_genes: int,
    *,
    min_cells: int,
    min_cpm: float,
    min_reads: float,
    min_prop: float,
    min_variance: float,
) -> dict[str, np.ndarray]:
    """Compute a per-cell-type gene keep mask from the all-donor pseudo-bulk.

    Returns a mapping from cell type to a boolean array of length ``n_genes``
    where True means the gene is retained for that cell type. The genes axis is
    positional and aligned with the input ``sums`` columns.

    For every cell type the following filters are applied across donors:
      1. Pseudo-bulk QC: only (donor, cell type) samples with at least
         ``min_cells`` cells are considered.
      2. Expression (filterByExpr-style): a gene is kept only if it reaches
         ``min_reads`` summed reads in at least ``min_prop`` of all donors AND
         ``min_cpm`` CPM (computed within each pseudo-bulk sample) in at least
         ``min_prop`` of all donors.
      3. Variance: a gene is kept only if its per-donor mean expression varies
         by more than ``min_variance`` across the donors of that cell type.
    """
    cell_types = sorted(obs["cell_type"].astype(str).unique().tolist())
    n_donors_total = int(obs["individual_id"].astype(str).nunique())
    n_required = max(1.0, float(np.ceil(min_prop * n_donors_total)))

    ct_values = obs["cell_type"].astype(str).to_numpy()
    n_cells_all = obs["n_cells"].to_numpy()

    logging.info(
        "Computing per-cell-type gene filters over %d donors "
        "(min_cells=%d, min_cpm=%g, min_reads=%g, min_prop=%g -> >=%g donors, "
        "min_variance=%g)",
        n_donors_total,
        min_cells,
        min_cpm,
        min_reads,
        min_prop,
        n_required,
        min_variance,
    )

    keep_by_ct: dict[str, np.ndarray] = {}
    for cell_type in cell_types:
        sel = ct_values == cell_type
        if min_cells > 0:
            sel = sel & (n_cells_all >= min_cells)
        group_rows = np.flatnonzero(sel)
        n_valid = len(group_rows)
        if n_valid == 0:
            logging.debug("Cell type %r: no donor samples pass QC", cell_type)
            keep_by_ct[cell_type] = np.zeros(n_genes, dtype=bool)
            continue

        counts = sums[group_rows, :]
        counts = np.asarray(counts.toarray()) if sp.issparse(counts) else np.asarray(counts)
        counts = counts.astype(np.float64, copy=False)

        # CPM within each donor/cell-type pseudo-bulk sample (no gene length;
        # treated as the "TPM" criterion for UMI data).
        lib = counts.sum(axis=1, keepdims=True)
        safe_lib = np.where(lib > 0, lib, 1.0)
        cpm = counts / safe_lib * 1e6

        reads_pass = (counts >= min_reads).sum(axis=0)
        cpm_pass = (cpm >= min_cpm).sum(axis=0)
        expr_keep = (reads_pass >= n_required) & (cpm_pass >= n_required)

        if n_valid >= 2:
            per_cell = counts / n_cells_all[group_rows][:, None]
            var_keep = per_cell.var(axis=0) > min_variance
        else:
            # A single (or zero) donor sample carries no across-donor variation.
            var_keep = np.zeros(n_genes, dtype=bool)

        keep_by_ct[cell_type] = expr_keep & var_keep
        logging.debug(
            "Cell type %r: %d donor samples pass QC, %d/%d genes kept",
            cell_type,
            n_valid,
            int(keep_by_ct[cell_type].sum()),
            n_genes,
        )

    if cell_types:
        logging.info(
            "Gene filter keeps on average %.1f / %d genes per cell type",
            float(np.mean([m.sum() for m in keep_by_ct.values()])),
            n_genes,
        )
    return keep_by_ct


# ---------------------------------------------------------------------------
# Output: per-(individual, cell type) pseudo-bulk h5ad (--population-h5ad)
# ---------------------------------------------------------------------------


def write_population_h5ad(
    *,
    sums: sp.csr_matrix | np.ndarray,
    obs: pd.DataFrame,
    var_index: pd.Index,
    selected_individuals: list[str],
    output_path: Path,
    overwrite: bool,
) -> None:
    """Write a per-(individual, cell type) pseudo-bulk AnnData target.

    ``sums`` holds summed counts per (individual, cell type) group and ``obs``
    the matching ``individual_id`` / ``cell_type`` / ``n_cells`` columns. Only
    the ``selected_individuals`` donors are written (the training donor subset
    by default) so downstream population-variance estimation does not leak the
    held-out evaluation cohort. Each row is divided by its cell count so the
    stored value is mean expression (counts per cell), matching the
    per-individual ground-truth CSVs and the target consumed by the training
    datasets. Every selected donor stays a separate row so downstream training
    can estimate per-gene population variance.
    """
    if output_path.exists() and not overwrite:
        raise FileExistsError(
            f"Output file already exists: {output_path}. Use --overwrite."
        )

    selected = set(selected_individuals)
    keep_rows = np.flatnonzero(
        obs["individual_id"].astype(str).isin(selected).to_numpy()
    )
    if keep_rows.size == 0:
        raise ValueError("No pseudo-bulk rows remain for the selected donors.")
    sums = sums[keep_rows]
    obs = obs.iloc[keep_rows].reset_index(drop=True)

    n_cells = obs["n_cells"].to_numpy(dtype=np.float64)
    if np.any(n_cells <= 0):
        raise ValueError("Encountered a pseudo-bulk group with zero cells.")

    if sp.issparse(sums):
        X = sums.multiply(1.0 / n_cells[:, None]).tocsr().astype(np.float32)
    else:
        X = (np.asarray(sums, dtype=np.float64) / n_cells[:, None]).astype(np.float32)

    out_obs = obs.copy()
    out_obs["individual_id"] = out_obs["individual_id"].astype(str).astype("category")
    out_obs["cell_type"] = out_obs["cell_type"].astype(str).astype("category")
    out_obs.index = pd.Index(
        [
            f"{individual_id}__{cell_type}"
            for individual_id, cell_type in zip(
                out_obs["individual_id"].astype(str),
                out_obs["cell_type"].astype(str),
            )
        ],
        name="obs_id",
    )

    out_var = pd.DataFrame(index=var_index.astype(str))

    out = ad.AnnData(X=X, obs=out_obs, var=out_var)
    out.uns["aggregation"] = "mean"
    out.uns["source"] = "src/preprocessing/onek1k/preprocess_data.py --population-h5ad"

    output_path.parent.mkdir(parents=True, exist_ok=True)
    logging.info(
        "Writing per-(individual, cell type) pseudo-bulk h5ad with shape %s to %s",
        out.shape,
        output_path,
    )
    out.write_h5ad(output_path, compression="gzip")


# ---------------------------------------------------------------------------
# Output: per-individual ground truth (default mode)
# ---------------------------------------------------------------------------


def make_unique_cell_type_names(cell_types: list[str]) -> dict[str, str]:
    used: dict[str, int] = {}
    mapping: dict[str, str] = {}
    for cell_type in cell_types:
        count = used.get(cell_type, 0)
        used[cell_type] = count + 1
        mapping[cell_type] = cell_type if count == 0 else f"{cell_type}_{count + 1}"
    return mapping


def harmonize_individual_ids(individuals: list[str]) -> list[str]:
    harmonized = []
    for individual in individuals:
        ids = individual.split("_")
        if len(ids) == 2:
            harmonized.append(f"OneK1K_{ids[1]}")
        else:
            raise ValueError(f"Unexpected individual ID format: {individual!r}")
    return harmonized


def normalize_expression_across_individuals(values: np.ndarray) -> np.ndarray:
    """Min-max normalize each gene row across finite individual values."""
    normalized = values.astype(np.float32, copy=True)
    finite = np.isfinite(normalized)
    has_value = finite.any(axis=1)
    if not bool(has_value.any()):
        return normalized

    valid_values = normalized[has_value]
    valid_finite = finite[has_value]
    row_min = np.min(np.where(valid_finite, valid_values, np.inf), axis=1)
    row_max = np.max(np.where(valid_finite, valid_values, -np.inf), axis=1)
    span = row_max - row_min

    rows = np.flatnonzero(has_value)
    variable = span > 0
    if bool(variable.any()):
        var_rows = rows[variable]
        normalized[var_rows] = (
            normalized[var_rows] - row_min[variable, None]
        ) / span[variable, None]

    if bool((~variable).any()):
        const_rows = rows[~variable]
        const_values = normalized[const_rows]
        const_values[np.isfinite(const_values)] = 0.0
        normalized[const_rows] = const_values

    return normalized


def write_ground_truth(
    *,
    sums: sp.csr_matrix | np.ndarray,
    obs: pd.DataFrame,
    gene_meta: pd.DataFrame,
    var_positions: np.ndarray,
    output_dir: Path,
    min_cells: int,
    keep_by_ct: dict[str, np.ndarray] | None,
    drop_empty: bool,
    normalize: bool,
    overwrite: bool,
) -> None:
    all_individuals = sorted(obs["individual_id"].astype(str).unique().tolist())
    cell_types = sorted(obs["cell_type"].astype(str).unique().tolist())
    if not all_individuals:
        raise ValueError("No individuals found in the input.")
    if not cell_types:
        raise ValueError("No cell types found in the input.")

    individual_to_col = {ind: pos for pos, ind in enumerate(all_individuals)}
    harmonized_individuals = harmonize_individual_ids(all_individuals)
    ct_to_name = make_unique_cell_type_names(cell_types)
    n_genes_meta = len(gene_meta)

    obs_individual = obs["individual_id"].astype(str).to_numpy()
    obs_cell_type = obs["cell_type"].astype(str).to_numpy()
    n_cells_all = obs["n_cells"].to_numpy(dtype=np.float64)

    output_dir.mkdir(parents=True, exist_ok=True)
    logging.info(
        "Preparing %d cell-type CSVs with %d individual columns in %s",
        len(cell_types),
        len(all_individuals),
        output_dir,
    )
    if normalize:
        logging.info("Normalizing each gene across individuals to the 0-1 range")

    n_written = 0
    n_skipped = 0
    for cell_type in cell_types:
        output_path = output_dir / f"{ct_to_name[cell_type]}.csv"
        if output_path.exists() and not overwrite:
            raise FileExistsError(
                f"Output file already exists: {output_path}. Use --overwrite."
            )

        group_rows = np.flatnonzero(obs_cell_type == cell_type)
        # Per-(individual, cell type) mean expression (counts per cell).
        values = np.full(
            (n_genes_meta, len(all_individuals)), np.nan, dtype=np.float32
        )
        n_dropped = 0
        for row in group_rows:
            n_cells = n_cells_all[row]
            if min_cells > 0 and n_cells < min_cells:
                n_dropped += 1
                continue
            col = individual_to_col[obs_individual[row]]
            mean_expr = row_to_dense(sums[row, :])[var_positions] / n_cells
            values[:, col] = mean_expr.astype(np.float32, copy=False)

        if keep_by_ct is not None:
            keep_full = keep_by_ct.get(cell_type)
            if keep_full is None:
                keep_genes = np.zeros(n_genes_meta, dtype=bool)
            else:
                keep_genes = keep_full[var_positions]
            values[~keep_genes, :] = np.nan
        else:
            keep_genes = np.ones(n_genes_meta, dtype=bool)

        if normalize:
            values = normalize_expression_across_individuals(values)

        logging.info(
            "%s (%d/%d donor samples pass QC, %d total columns, "
            "%d/%d genes kept)",
            output_path.name,
            len(group_rows) - n_dropped,
            len(group_rows),
            len(all_individuals),
            int(keep_genes.sum()),
            n_genes_meta,
        )

        out_gene_meta = gene_meta.reset_index(drop=True)
        if drop_empty:
            gene_has_value = np.isfinite(values).any(axis=1)
            if not gene_has_value.any():
                if keep_by_ct is None and min_cells > 0:
                    logging.warning(
                        "Cell type %r has no expression values because no donor "
                        "samples passed --min-cells %d; skipping (pass "
                        "--min-cells 0 to disable pseudo-bulk sample QC, or "
                        "--keep-empty to write the all-NaN CSV)",
                        cell_type,
                        min_cells,
                    )
                elif keep_by_ct is None:
                    logging.warning(
                        "Cell type %r has no expression values; skipping (pass "
                        "--keep-empty to write the all-NaN CSV)",
                        cell_type,
                    )
                else:
                    logging.warning(
                        "Cell type %r has no genes with values after filtering; "
                        "skipping (pass --keep-empty to write it anyway)",
                        cell_type,
                    )
                n_skipped += 1
                continue
            n_empty_genes = int((~gene_has_value).sum())
            if n_empty_genes and keep_by_ct is not None:
                logging.debug(
                    "Cell type %r: dropping %d empty genes", cell_type, n_empty_genes
                )
                out_gene_meta = out_gene_meta.loc[gene_has_value].reset_index(drop=True)
                values = values[gene_has_value]
            elif n_empty_genes:
                logging.debug(
                    "Cell type %r: keeping %d empty genes because gene filters "
                    "are disabled",
                    cell_type,
                    n_empty_genes,
                )

        out = pd.concat(
            [
                out_gene_meta,
                pd.DataFrame(values, columns=harmonized_individuals),
            ],
            axis=1,
        )
        out.to_csv(output_path, index=False)
        n_written += 1

    logging.info(
        "Wrote %d cell-type CSVs%s",
        n_written,
        f" (skipped {n_skipped} empty cell types)" if n_skipped else "",
    )
    if n_written == 0:
        raise ValueError(
            "All cell types were empty after filtering; nothing written. "
            "Loosen the filter thresholds or pass --keep-empty."
        )


# ---------------------------------------------------------------------------
# Output: individual-aggregated target variables (--aggregate-across-individuals)
# ---------------------------------------------------------------------------


def select_individuals(
    obs: pd.DataFrame,
    requested: list[str],
    all_individuals: bool,
    missing_individuals: str,
) -> list[str]:
    available = set(obs["individual_id"].astype(str).unique().tolist())
    if all_individuals:
        selected = sorted(available)
        logging.info("Using all %d donors for aggregation", len(selected))
        return selected

    missing = [ind for ind in requested if ind not in available]
    if missing:
        msg = (
            f"{len(missing)} of {len(requested)} requested donors are not present "
            f"in the input: {missing[:10]}"
        )
        if missing_individuals == "error":
            raise ValueError(msg)
        logging.warning("%s. Dropping missing donors.", msg)

    selected = [ind for ind in requested if ind in available]
    if not selected:
        raise ValueError("None of the requested donors are present in the input.")

    logging.info(
        "Using %d of %d requested donors for aggregation: %s",
        len(selected),
        len(requested),
        selected,
    )
    return selected


def rank_genes_within_cell_types(target: pd.DataFrame) -> pd.DataFrame:
    """Convert expression to percentile ranks across genes within each row.

    Missing genes remain missing and are excluded from the rank denominator.
    Ties receive their average rank, yielding finite percentiles in ``(0, 1]``.
    """
    return target.rank(axis=1, method="average", pct=True).astype(np.float32)


def write_target_vars(
    *,
    sums: sp.csr_matrix | np.ndarray,
    obs: pd.DataFrame,
    genes: pd.Index,
    selected_individuals: list[str],
    output_path: Path,
    min_cells: int,
    keep_by_ct: dict[str, np.ndarray] | None,
    drop_empty: bool,
    normalize: bool,
    overwrite: bool,
    percentiles: bool = False,
) -> None:
    if output_path.exists() and not overwrite:
        raise FileExistsError(
            f"Output file already exists: {output_path}. Use --overwrite."
        )

    selected = set(selected_individuals)
    sub_obs = obs[obs["individual_id"].astype(str).isin(selected)].copy()
    if sub_obs.empty:
        raise ValueError("No pseudo-bulk rows remain for the selected donors.")

    cell_types = sorted(sub_obs["cell_type"].astype(str).unique().tolist())
    n_genes = len(genes)
    n_cells_all = obs["n_cells"]
    sub_ct = sub_obs["cell_type"].astype(str)

    # Cells with no valid donor sample (or fully filtered genes) stay NaN so
    # downstream code can treat them as missing.
    values = np.full((len(cell_types), n_genes), np.nan, dtype=np.float32)
    if normalize:
        logging.info(
            "Normalizing selected donor expression per gene before aggregation"
        )

    for row_idx, cell_type in enumerate(cell_types):
        group_rows = sub_obs.index[sub_ct == cell_type].to_numpy()
        if min_cells > 0 and len(group_rows) > 0:
            n_cells_rows = n_cells_all.loc[group_rows].to_numpy()
            group_rows = group_rows[n_cells_rows >= min_cells]
        if len(group_rows) == 0:
            continue

        if keep_by_ct is not None:
            keep_genes = keep_by_ct.get(cell_type)
            if keep_genes is None:
                continue  # cell type fully filtered out -> leave NaN
        else:
            keep_genes = np.ones(n_genes, dtype=bool)

        n_cells = n_cells_all.loc[group_rows].to_numpy(dtype=np.float64)
        if normalize:
            counts = sums[group_rows, :]
            counts = (
                np.asarray(counts.toarray())
                if sp.issparse(counts)
                else np.asarray(counts)
            )
            donor_values = counts.astype(np.float64, copy=False) / n_cells[:, None]
            donor_values[:, ~keep_genes] = np.nan
            donor_values = normalize_expression_across_individuals(donor_values.T).T

            weights = n_cells[:, None]
            finite = np.isfinite(donor_values)
            weight_sums = np.sum(np.where(finite, weights, 0.0), axis=0)
            weighted_sums = np.nansum(donor_values * weights, axis=0)
            row = np.full(n_genes, np.nan, dtype=np.float32)
            has_weight = weight_sums > 0
            row[has_weight] = (
                weighted_sums[has_weight] / weight_sums[has_weight]
            ).astype(np.float32, copy=False)
        else:
            # Cell-weighted mean: sum counts over donors / total cells.
            total_cells = float(n_cells.sum())
            summed = sums[group_rows, :].sum(axis=0)
            row = row_to_dense(summed) / total_cells
            row = np.asarray(row, dtype=np.float32).copy()
            row[~keep_genes] = np.nan

        values[row_idx] = row

    target = pd.DataFrame(values, index=cell_types, columns=genes.astype(str))
    target.index.name = "cell_type"

    if target.columns.has_duplicates:
        n_duplicates = int(target.columns.duplicated().sum())
        logging.warning(
            "Gene axis has %d duplicate gene IDs; collapsing duplicates by mean",
            n_duplicates,
        )
        target = target.T.groupby(level=0, sort=False).mean().T

    if drop_empty:
        n_ct_before, n_gene_before = target.shape
        target = target.dropna(axis=0, how="all").dropna(axis=1, how="all")
        n_ct_dropped = n_ct_before - target.shape[0]
        n_gene_dropped = n_gene_before - target.shape[1]
        if n_ct_dropped or n_gene_dropped:
            logging.info(
                "Dropped %d empty cell types and %d empty genes after filtering "
                "(%d cell types x %d genes remain)",
                n_ct_dropped,
                n_gene_dropped,
                target.shape[0],
                target.shape[1],
            )
        if target.empty:
            raise ValueError(
                "All cell types/genes were dropped after filtering; nothing to "
                "write. Loosen the filter thresholds or pass --keep-empty."
            )

    if percentiles:
        logging.info(
            "Ranking aggregated mean expression across genes within each cell type"
        )
        target = rank_genes_within_cell_types(target)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    logging.info("Writing target-variable CSV with shape %s to %s", target.shape, output_path)
    target.to_csv(output_path)


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def main() -> None:
    args = parse_args()
    setup_logging(args.verbose)
    logging.debug("Arguments: %s", args)

    input_path = args.input.expanduser().resolve()
    if not input_path.exists():
        print(f"Error: input file does not exist: {input_path}", file=sys.stderr)
        sys.exit(1)

    output_path = args.output.expanduser().resolve()

    try:
        gene_meta = None
        write_ground_truth_csvs = (
            not args.aggregate_across_individuals and not args.population_h5ad
        )
        if write_ground_truth_csvs:
            gene_meta = load_gene_metadata(args.gene_metadata)

        logging.info("Reading %s", input_path)
        adata = ad.read_h5ad(input_path)
        validate_obs_columns(adata.obs)
        X = load_matrix(adata, args.layer, args.use_x)

        grouping, keep_cells = prepare_grouping(adata.obs)
        if not bool(keep_cells.all()):
            X = X[keep_cells, :]

        sums, obs = aggregate_by_individual_and_cell_type(X, grouping)
        logging.info(
            "Pseudo-bulk shape before gene axis: %d groups x %d genes",
            obs.shape[0],
            adata.n_vars,
        )

        keep_by_ct = None
        if args.population_h5ad:
            # The population target stays dense (no per-cell-type gene filter);
            # gene selection happens later via the reference feature set.
            logging.info(
                "Population h5ad mode: per-cell-type gene filters are not "
                "applied (target kept dense)."
            )
        elif args.disable_gene_filters:
            logging.info("Per-cell-type expression and variance gene filters disabled")
            if args.min_cells > 0:
                logging.info(
                    "Pseudo-bulk sample QC is still active (--min-cells %d); "
                    "pass --min-cells 0 to disable it as well",
                    args.min_cells,
                )
        else:
            keep_by_ct = compute_gene_keep_mask(
                sums=sums,
                obs=obs,
                n_genes=adata.n_vars,
                min_cells=args.min_cells,
                min_cpm=args.min_cpm,
                min_reads=args.min_reads,
                min_prop=args.expr_min_prop,
                min_variance=args.min_variance,
            )

        if args.population_h5ad:
            requested = args.target_individual or list(INDIVIDUAL_IDS)
            selected_individuals = select_individuals(
                obs=obs,
                requested=requested,
                all_individuals=args.all_individuals,
                missing_individuals=args.missing_individuals,
            )
            write_population_h5ad(
                sums=sums,
                obs=obs,
                var_index=adata.var.index,
                selected_individuals=selected_individuals,
                output_path=output_path,
                overwrite=args.overwrite,
            )
        elif args.aggregate_across_individuals:
            requested = args.target_individual or list(INDIVIDUAL_IDS)
            selected_individuals = select_individuals(
                obs=obs,
                requested=requested,
                all_individuals=args.all_individuals,
                missing_individuals=args.missing_individuals,
            )
            write_target_vars(
                sums=sums,
                obs=obs,
                genes=adata.var.index,
                selected_individuals=selected_individuals,
                output_path=output_path,
                min_cells=args.min_cells,
                keep_by_ct=keep_by_ct,
                drop_empty=not args.keep_empty,
                normalize=args.normalize,
                percentiles=args.percentiles,
                overwrite=args.overwrite,
            )
        else:
            gene_meta, var_positions = align_gene_metadata_to_var(
                gene_meta, adata.var.index
            )
            write_ground_truth(
                sums=sums,
                obs=obs,
                gene_meta=gene_meta,
                var_positions=var_positions,
                output_dir=output_path,
                min_cells=args.min_cells,
                keep_by_ct=keep_by_ct,
                drop_empty=not args.keep_empty,
                normalize=args.normalize,
                overwrite=args.overwrite,
            )

    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)

    logging.info("Done.")


if __name__ == "__main__":
    main()
