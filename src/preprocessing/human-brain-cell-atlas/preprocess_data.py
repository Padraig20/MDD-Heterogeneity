#!/usr/bin/env python3

from __future__ import annotations

import argparse
import logging
import re
import sys
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp


"""
get_target_vars.py

Build Human Brain Cell Atlas target-variable expression tables from a
single-cell h5ad file.

Cells are first aggregated into per-(donor, cell type) pseudo-bulk samples.
Per-cell-type gene filters are computed from those donor pseudo-bulk samples,
then each output row is a cell-weighted mean across donors. No per-individual
"ground truth" files are produced for this dataset.

https://cellxgene.cziscience.com/collections/283d65eb-dd53-496d-adb7-7570c7caa443
"""


DEFAULT_INDIVIDUAL_COL = "donor_id"
DEFAULT_CELL_TYPE_COL = "supercluster_term"
DEFAULT_AGE_COL = "development_stage"
DEFAULT_CHROMOSOME_COL = "Chromosome"
DEFAULT_LAYER = "counts"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build Human Brain Cell Atlas target variables from a single-cell "
            "h5ad file. Rows are cell types (or cell type/chromosome pairs); "
            "columns are genes."
        )
    )
    parser.add_argument(
        "-i",
        "--input",
        type=Path,
        required=True,
        help="Input single-cell h5ad file.",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        required=True,
        help=(
            "Output target-variable CSV. With --split-by-age, one age-specific "
            "CSV is written next to this path."
        ),
    )
    parser.add_argument(
        "--individual-col",
        default=DEFAULT_INDIVIDUAL_COL,
        help=f'Input obs column containing donor IDs. Default: "{DEFAULT_INDIVIDUAL_COL}".',
    )
    parser.add_argument(
        "--cell-type-col",
        default=DEFAULT_CELL_TYPE_COL,
        help=f'Input obs column containing cell-type labels. Default: "{DEFAULT_CELL_TYPE_COL}".',
    )
    parser.add_argument(
        "--age-col",
        default=DEFAULT_AGE_COL,
        help=f'Input obs column containing age/development-stage labels. Default: "{DEFAULT_AGE_COL}".',
    )
    parser.add_argument(
        "--chromosome-col",
        default=DEFAULT_CHROMOSOME_COL,
        help=f'Input var column containing chromosome labels. Default: "{DEFAULT_CHROMOSOME_COL}".',
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
        "-chr",
        "--chromosome",
        action="store_true",
        help="Write rows for each cell-type/chromosome pair.",
    )
    parser.add_argument(
        "--mdd_genes",
        type=Path,
        default=None,
        help="Optional TSV with an ENSID column; target variables are limited to these genes.",
    )
    parser.add_argument(
        "--split-by-age",
        action="store_true",
        help="Write one target-variable CSV per observed age group.",
    )
    parser.add_argument(
        "--target-individual",
        action="append",
        default=None,
        help=(
            "Donor ID to include in donor aggregation. May be passed multiple "
            "times. Defaults to every donor in the input."
        ),
    )
    parser.add_argument(
        "--missing-individuals",
        choices=["error", "drop"],
        default="error",
        help="What to do if requested donors are missing. Default: error.",
    )
    parser.add_argument(
        "--min-cells",
        type=int,
        default=125,
        help=(
            "Pseudo-bulk quality filter: minimum number of cells a "
            "(donor, cell type) sample must contain. Set 0 to disable. "
            "Default: 125."
        ),
    )
    parser.add_argument(
        "--min-cpm",
        type=float,
        default=0.1,
        help=(
            "Per-cell-type expression filter: a gene must reach this CPM in at "
            "least --expr-min-prop of donors. Default: 0.1."
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
            "Fraction of donors in the current aggregation scope that must "
            "satisfy both expression thresholds. Default: 0.2."
        ),
    )
    parser.add_argument(
        "--min-variance",
        type=float,
        default=1e-8,
        help=(
            "Per-cell-type variance filter: genes whose per-donor mean "
            "expression varies by this much or less are dropped. Default: 1e-8."
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
            "Keep cell types and genes that are entirely empty after filtering. "
            "By default empty rows/columns are dropped."
        ),
    )
    parser.add_argument(
        "--normalize",
        action="store_true",
        help=(
            "Normalize each gene's per-donor expression values to the 0-1 range "
            "before aggregating donors."
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
    if args.split_by_age and args.chromosome:
        parser.error("--split-by-age is currently only supported without --chromosome.")
    return args


def setup_logging(verbosity: int) -> None:
    level = logging.DEBUG if verbosity >= 1 else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )


def validate_columns(
    adata: ad.AnnData,
    *,
    individual_col: str,
    cell_type_col: str,
    age_col: str,
    chromosome_col: str,
    split_by_age: bool,
    group_by_chromosome: bool,
) -> None:
    missing_obs = [
        col
        for col in (individual_col, cell_type_col)
        if col not in adata.obs.columns
    ]
    if split_by_age and age_col not in adata.obs.columns:
        missing_obs.append(age_col)
    if missing_obs:
        raise ValueError(
            f"Input h5ad obs is missing required column(s): {missing_obs}. "
            f"Available obs columns: {adata.obs.columns.tolist()}"
        )

    if group_by_chromosome and chromosome_col not in adata.var.columns:
        raise ValueError(
            f"Input h5ad var is missing required column {chromosome_col!r}. "
            f"Available var columns: {adata.var.columns.tolist()}"
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


def simplify_chr(chromosome: str) -> str | None:
    match = re.match(r"^(chr(?:\d+|X|Y))\b", str(chromosome))
    return match.group(1) if match else None


def age_group_label(age_stage: str) -> str:
    return str(age_stage).split("-")[0]


def sanitize_label(label: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(label)).strip("_") or "unknown"


def filter_to_mdd_genes(
    X,
    var: pd.DataFrame,
    mdd_genes_path: Path | None,
) -> tuple[sp.spmatrix | np.ndarray, pd.DataFrame]:
    if mdd_genes_path is None:
        return X, var

    logging.info("Filtering target variables to MDD genes from %s", mdd_genes_path)
    mdd_genes = pd.read_csv(mdd_genes_path, sep="\t")
    if "ENSID" not in mdd_genes.columns:
        raise ValueError(
            f"MDD gene list is missing required column 'ENSID': {mdd_genes_path}"
        )

    requested = pd.Index(mdd_genes["ENSID"].dropna().astype(str))
    gene_names = var.index.astype(str)
    keep = gene_names.isin(requested)
    if not bool(keep.any()):
        raise ValueError("No h5ad genes match the supplied MDD gene list.")

    logging.info(
        "Keeping %d h5ad genes that match %d requested MDD genes",
        int(keep.sum()),
        len(requested.unique()),
    )
    return X[:, keep], var.loc[keep].copy()


def prepare_grouping(
    obs: pd.DataFrame,
    *,
    individual_col: str,
    cell_type_col: str,
    age_col: str,
    split_by_age: bool,
) -> tuple[pd.DataFrame, np.ndarray]:
    required = [individual_col, cell_type_col]
    if split_by_age:
        required.append(age_col)

    keep = pd.Series(True, index=obs.index)
    for col in required:
        keep &= obs[col].notna()

    if not bool(keep.all()):
        logging.warning(
            "Dropping %d cells with missing grouping labels",
            int((~keep).sum()),
        )

    grouping = pd.DataFrame(
        {
            "individual_id": obs.loc[keep, individual_col].astype(str).to_numpy(),
            "cell_type": obs.loc[keep, cell_type_col].astype(str).to_numpy(),
        }
    )
    if split_by_age:
        grouping["age_group"] = (
            obs.loc[keep, age_col].map(age_group_label).astype(str).to_numpy()
        )
    return grouping, keep.to_numpy()


def aggregate_sparse(
    X: sp.spmatrix,
    grouping: pd.DataFrame,
) -> tuple[sp.csr_matrix, pd.DataFrame]:
    group_cols = grouping.columns.tolist()
    grouped = grouping.groupby(group_cols, sort=True)
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

    obs = pd.DataFrame(keys, columns=group_cols)
    obs["n_cells"] = n_cells_per_group
    return sums, obs


def aggregate_dense(
    X: np.ndarray,
    grouping: pd.DataFrame,
) -> tuple[np.ndarray, pd.DataFrame]:
    group_cols = grouping.columns.tolist()
    grouped = grouping.groupby(group_cols, sort=True)
    keys = list(grouped.indices.keys())
    sums = np.zeros((len(keys), X.shape[1]), dtype=np.float32)
    n_cells_per_group = np.empty(len(keys), dtype=np.int64)

    for code, key in enumerate(keys):
        idx = np.asarray(grouped.indices[key], dtype=np.int64)
        sums[code] = np.asarray(X[idx, :].sum(axis=0)).ravel()
        n_cells_per_group[code] = len(idx)

    obs = pd.DataFrame(keys, columns=group_cols)
    obs["n_cells"] = n_cells_per_group
    return sums, obs


def aggregate_by_group(
    X,
    grouping: pd.DataFrame,
) -> tuple[sp.csr_matrix | np.ndarray, pd.DataFrame]:
    if grouping.empty:
        raise ValueError("No cells remain after filtering missing grouping labels.")

    logging.info("Aggregating %d cells into pseudo-bulk groups", len(grouping))
    if sp.issparse(X):
        return aggregate_sparse(X.tocsr(), grouping)
    return aggregate_dense(np.asarray(X), grouping)


def row_to_dense(row: sp.spmatrix | np.ndarray) -> np.ndarray:
    if sp.issparse(row):
        return np.asarray(row.toarray()).ravel()
    return np.asarray(row).ravel()


def scope_key(row: pd.Series, split_by_age: bool) -> tuple[str | None, str]:
    age_group = str(row["age_group"]) if split_by_age else None
    return age_group, str(row["cell_type"])


def normalize_expression_across_donors(values: np.ndarray) -> np.ndarray:
    normalized = values.astype(np.float32, copy=True)
    finite = np.isfinite(normalized)
    has_value = finite.any(axis=0)
    if not bool(has_value.any()):
        return normalized

    valid_values = normalized[:, has_value]
    valid_finite = finite[:, has_value]
    col_min = np.min(np.where(valid_finite, valid_values, np.inf), axis=0)
    col_max = np.max(np.where(valid_finite, valid_values, -np.inf), axis=0)
    span = col_max - col_min

    variable = span > 0
    if bool(variable.any()):
        valid_values[:, variable] = (
            valid_values[:, variable] - col_min[variable]
        ) / span[variable]
    if bool((~variable).any()):
        constant = valid_values[:, ~variable]
        constant[np.isfinite(constant)] = 0.0
        valid_values[:, ~variable] = constant

    normalized[:, has_value] = valid_values
    return normalized


def compute_gene_keep_mask(
    sums: sp.csr_matrix | np.ndarray,
    obs: pd.DataFrame,
    n_genes: int,
    *,
    split_by_age: bool,
    min_cells: int,
    min_cpm: float,
    min_reads: float,
    min_prop: float,
    min_variance: float,
) -> dict[tuple[str | None, str], np.ndarray]:
    keep_by_scope: dict[tuple[str | None, str], np.ndarray] = {}
    n_cells_all = obs["n_cells"].to_numpy()
    n_donors_total = int(obs["individual_id"].astype(str).nunique())
    donors_by_age = None
    if split_by_age:
        donors_by_age = (
            obs.groupby("age_group")["individual_id"]
            .nunique()
            .astype(int)
            .to_dict()
        )

    scope_cols = ["cell_type"]
    if split_by_age:
        scope_cols.insert(0, "age_group")

    logging.info(
        "Computing per-cell-type gene filters (min_cells=%d, min_cpm=%g, "
        "min_reads=%g, min_prop=%g, min_variance=%g)",
        min_cells,
        min_cpm,
        min_reads,
        min_prop,
        min_variance,
    )

    for raw_key, scope_obs in obs.groupby(scope_cols, sort=True):
        key_tuple = raw_key if isinstance(raw_key, tuple) else (None, raw_key)
        key = (str(key_tuple[0]) if split_by_age else None, str(key_tuple[-1]))
        group_rows = scope_obs.index.to_numpy()
        if min_cells > 0:
            group_rows = group_rows[n_cells_all[group_rows] >= min_cells]

        n_valid = len(group_rows)
        if n_valid == 0:
            keep_by_scope[key] = np.zeros(n_genes, dtype=bool)
            continue

        counts = sums[group_rows, :]
        counts = np.asarray(counts.toarray()) if sp.issparse(counts) else np.asarray(counts)
        counts = counts.astype(np.float64, copy=False)

        scope_donors_total = (
            int(donors_by_age[key[0]]) if donors_by_age is not None else n_donors_total
        )
        n_required = max(1.0, float(np.ceil(min_prop * scope_donors_total)))

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
            var_keep = np.zeros(n_genes, dtype=bool)

        keep_by_scope[key] = expr_keep & var_keep
        logging.debug(
            "Scope %s: %d donor samples pass QC, %d/%d genes kept",
            key,
            n_valid,
            int(keep_by_scope[key].sum()),
            n_genes,
        )

    if keep_by_scope:
        logging.info(
            "Gene filter keeps on average %.1f / %d genes per output scope",
            float(np.mean([mask.sum() for mask in keep_by_scope.values()])),
            n_genes,
        )
    return keep_by_scope


def select_individuals(
    obs: pd.DataFrame,
    requested: list[str] | None,
    missing_individuals: str,
) -> list[str]:
    available = sorted(obs["individual_id"].astype(str).unique().tolist())
    if requested is None:
        logging.info("Using all %d donors for aggregation", len(available))
        return available

    available_set = set(available)
    missing = [individual_id for individual_id in requested if individual_id not in available_set]
    if missing:
        msg = (
            f"{len(missing)} of {len(requested)} requested donors are not present "
            f"in the input: {missing[:10]}"
        )
        if missing_individuals == "error":
            raise ValueError(msg)
        logging.warning("%s. Dropping missing donors.", msg)

    selected = [individual_id for individual_id in requested if individual_id in available_set]
    if not selected:
        raise ValueError("None of the requested donors are present in the input.")
    logging.info("Using %d requested donors for aggregation", len(selected))
    return selected


def build_target_frame(
    *,
    sums: sp.csr_matrix | np.ndarray,
    obs: pd.DataFrame,
    genes: pd.Index,
    selected_individuals: list[str],
    split_by_age: bool,
    age_group: str | None,
    keep_by_scope: dict[tuple[str | None, str], np.ndarray] | None,
    min_cells: int,
    normalize: bool,
    chromosome_names: pd.Series | None,
    drop_empty: bool,
) -> pd.DataFrame:
    selected = set(selected_individuals)
    sub_obs = obs[obs["individual_id"].astype(str).isin(selected)].copy()
    if split_by_age:
        sub_obs = sub_obs[sub_obs["age_group"].astype(str) == str(age_group)]
    if sub_obs.empty:
        raise ValueError("No pseudo-bulk rows remain for the selected donors.")

    cell_types = sorted(sub_obs["cell_type"].astype(str).unique().tolist())
    n_genes = len(genes)
    n_cells_all = obs["n_cells"]
    sub_ct = sub_obs["cell_type"].astype(str)

    rows: list[np.ndarray] = []
    row_index: list[str] = []

    chromosomes = None
    if chromosome_names is not None:
        chromosomes = sorted(
            chromosome
            for chromosome in chromosome_names.dropna().astype(str).unique().tolist()
            if chromosome
        )

    for cell_type in cell_types:
        group_rows = sub_obs.index[sub_ct == cell_type].to_numpy()
        if min_cells > 0 and len(group_rows) > 0:
            n_cells_rows = n_cells_all.loc[group_rows].to_numpy()
            group_rows = group_rows[n_cells_rows >= min_cells]
        if len(group_rows) == 0:
            base_row = np.full(n_genes, np.nan, dtype=np.float32)
        else:
            key = (str(age_group) if split_by_age else None, cell_type)
            if keep_by_scope is not None:
                keep_genes = keep_by_scope.get(key)
                if keep_genes is None:
                    keep_genes = np.zeros(n_genes, dtype=bool)
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
                donor_values = normalize_expression_across_donors(donor_values)

                weights = n_cells[:, None]
                finite = np.isfinite(donor_values)
                weight_sums = np.sum(np.where(finite, weights, 0.0), axis=0)
                weighted_sums = np.nansum(donor_values * weights, axis=0)
                base_row = np.full(n_genes, np.nan, dtype=np.float32)
                has_weight = weight_sums > 0
                base_row[has_weight] = (
                    weighted_sums[has_weight] / weight_sums[has_weight]
                ).astype(np.float32, copy=False)
            else:
                total_cells = float(n_cells.sum())
                summed = sums[group_rows, :].sum(axis=0)
                base_row = row_to_dense(summed) / total_cells
                base_row = np.asarray(base_row, dtype=np.float32).copy()
                base_row[~keep_genes] = np.nan

        if chromosomes is None:
            row_index.append(cell_type)
            rows.append(base_row)
            continue

        for chromosome in chromosomes:
            mask_ch = chromosome_names.astype(str).to_numpy() == chromosome
            row = np.zeros(n_genes, dtype=np.float32)
            row[mask_ch] = base_row[mask_ch]
            row_index.append(f"{cell_type},{chromosome}")
            rows.append(row)

    target = pd.DataFrame(rows, index=row_index, columns=genes.astype(str))
    target.index.name = "cell_type" if chromosomes is None else "cell_type,chromosome"

    if target.columns.has_duplicates:
        n_duplicates = int(target.columns.duplicated().sum())
        logging.warning(
            "Gene axis has %d duplicate gene IDs; collapsing duplicates by mean",
            n_duplicates,
        )
        target = target.T.groupby(level=0, sort=False).mean().T

    if drop_empty:
        n_rows_before, n_genes_before = target.shape
        target = target.dropna(axis=0, how="all").dropna(axis=1, how="all")
        n_rows_dropped = n_rows_before - target.shape[0]
        n_genes_dropped = n_genes_before - target.shape[1]
        if n_rows_dropped or n_genes_dropped:
            logging.info(
                "Dropped %d empty rows and %d empty genes after filtering "
                "(%d rows x %d genes remain)",
                n_rows_dropped,
                n_genes_dropped,
                target.shape[0],
                target.shape[1],
            )
        if target.empty:
            raise ValueError(
                "All target rows/genes were dropped after filtering; nothing to "
                "write. Loosen the filter thresholds or pass --keep-empty."
            )

    return target


def age_output_path(output_path: Path, age_group: str) -> Path:
    label = sanitize_label(age_group)
    if output_path.suffix:
        return output_path.with_name(f"{output_path.stem}_age_{label}{output_path.suffix}")
    return output_path / f"age_{label}.csv"


def write_target(path: Path, target: pd.DataFrame, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"Output file already exists: {path}. Use --overwrite.")
    path.parent.mkdir(parents=True, exist_ok=True)
    logging.info("Writing target-variable CSV with shape %s to %s", target.shape, path)
    target.to_csv(path)


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
        logging.info("Reading %s", input_path)
        adata = ad.read_h5ad(input_path)
        validate_columns(
            adata,
            individual_col=args.individual_col,
            cell_type_col=args.cell_type_col,
            age_col=args.age_col,
            chromosome_col=args.chromosome_col,
            split_by_age=args.split_by_age,
            group_by_chromosome=args.chromosome,
        )

        X = load_matrix(adata, args.layer, args.use_x)
        X, var = filter_to_mdd_genes(X, adata.var, args.mdd_genes)
        gene_index = var.index.astype(str)

        chromosome_names = None
        if args.chromosome:
            chromosome_names = var[args.chromosome_col].map(simplify_chr)

        grouping, keep_cells = prepare_grouping(
            adata.obs,
            individual_col=args.individual_col,
            cell_type_col=args.cell_type_col,
            age_col=args.age_col,
            split_by_age=args.split_by_age,
        )
        if not bool(keep_cells.all()):
            X = X[keep_cells, :]

        sums, obs = aggregate_by_group(X, grouping)
        logging.info(
            "Pseudo-bulk shape before gene filtering: %d groups x %d genes",
            obs.shape[0],
            len(gene_index),
        )

        selected_individuals = select_individuals(
            obs=obs,
            requested=args.target_individual,
            missing_individuals=args.missing_individuals,
        )

        keep_by_scope = None
        if args.disable_gene_filters:
            logging.info("Per-cell-type expression and variance gene filters disabled")
            if args.min_cells > 0:
                logging.info(
                    "Pseudo-bulk sample QC is still active (--min-cells %d); "
                    "pass --min-cells 0 to disable it as well",
                    args.min_cells,
                )
        else:
            keep_by_scope = compute_gene_keep_mask(
                sums=sums,
                obs=obs,
                n_genes=len(gene_index),
                split_by_age=args.split_by_age,
                min_cells=args.min_cells,
                min_cpm=args.min_cpm,
                min_reads=args.min_reads,
                min_prop=args.expr_min_prop,
                min_variance=args.min_variance,
            )

        if args.split_by_age:
            age_groups = sorted(obs["age_group"].astype(str).unique().tolist())
            for age_group in age_groups:
                target = build_target_frame(
                    sums=sums,
                    obs=obs,
                    genes=gene_index,
                    selected_individuals=selected_individuals,
                    split_by_age=True,
                    age_group=age_group,
                    keep_by_scope=keep_by_scope,
                    min_cells=args.min_cells,
                    normalize=args.normalize,
                    chromosome_names=None,
                    drop_empty=not args.keep_empty,
                )
                write_target(age_output_path(output_path, age_group), target, args.overwrite)
        else:
            target = build_target_frame(
                sums=sums,
                obs=obs,
                genes=gene_index,
                selected_individuals=selected_individuals,
                split_by_age=False,
                age_group=None,
                keep_by_scope=keep_by_scope,
                min_cells=args.min_cells,
                normalize=args.normalize,
                chromosome_names=chromosome_names,
                drop_empty=not args.keep_empty,
            )
            write_target(output_path, target, args.overwrite)

    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)

    logging.info("Done.")


if __name__ == "__main__":
    main()
