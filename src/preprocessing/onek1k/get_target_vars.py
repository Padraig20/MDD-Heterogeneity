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
            "Compute OneK1K pseudo-bulk expression from a single h5ad file and "
            "optionally write target-variable means for a donor subset."
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
        "-o-h5ad",
        "--output-h5ad",
        type=Path,
        default=None,
        help=(
            "Output pseudo-bulk h5ad. Shape is "
            "(individual_id x cell_type groups) by genes."
        ),
    )
    parser.add_argument(
        "-o-csv",
        "--output-csv",
        type=Path,
        default=None,
        help=(
            "Output target-variable CSV with rows=cell types and columns=genes. "
            "Values are mean expression across the selected donor subset."
        ),
    )
    parser.add_argument(
        "--individual-col",
        default="donor_id",
        help='Input obs column containing donor/individual IDs. Default: "donor_id".',
    )
    parser.add_argument(
        "--cell-type-col",
        default="celltype",
        help='Input obs column containing cell-type labels. Default: "celltype".',
    )
    matrix_group = parser.add_mutually_exclusive_group()
    matrix_group.add_argument(
        "--layer",
        default="counts",
        help='Input layer to aggregate. Default: "counts".',
    )
    matrix_group.add_argument(
        "--use-x",
        action="store_true",
        help="Aggregate adata.X instead of a layer.",
    )
    parser.add_argument(
        "--aggregation",
        choices=["sum", "mean"],
        default="mean",
        help=(
            "How to aggregate cells within each (individual_id, cell_type) "
            "row in the output h5ad. Default: mean."
        ),
    )
    parser.add_argument(
        "--csv-mean-mode",
        choices=["cell", "individual"],
        default="cell",
        help=(
            "How to average selected donors for -o-csv. 'cell' matches "
            "get_target_vars.py semantics by computing a cell-weighted mean "
            "over all selected cells. 'individual' first averages within each "
            "donor/cell type, then averages donors equally. Default: cell."
        ),
    )
    parser.add_argument(
        "--target-individual",
        action="append",
        default=None,
        help=(
            "Donor ID to include in -o-csv. May be passed multiple times. "
            "Defaults to INDIVIDUAL_IDS in this script."
        ),
    )
    parser.add_argument(
        "--all-individuals",
        action="store_true",
        help="For -o-csv, use every donor in the input instead of INDIVIDUAL_IDS.",
    )
    parser.add_argument(
        "--missing-individuals",
        choices=["error", "drop"],
        default="error",
        help=(
            "What to do if requested CSV donors are missing. Default: error. "
            "Ignored with --all-individuals."
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite output files if they already exist.",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="count",
        default=0,
        help="Increase verbosity.",
    )

    args = parser.parse_args()

    if args.output_h5ad is None and args.output_csv is None:
        parser.error("At least one output is required: -o-h5ad and/or -o-csv.")
    if args.all_individuals and args.target_individual is not None:
        parser.error("--all-individuals cannot be combined with --target-individual.")

    return args


def setup_logging(verbosity: int) -> None:
    level = logging.INFO
    if verbosity >= 1:
        level = logging.DEBUG
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )


def check_output_path(path: Path | None, overwrite: bool) -> None:
    if path is None:
        return
    if path.exists() and not overwrite:
        raise FileExistsError(f"Output file already exists: {path}. Use --overwrite.")


def load_matrix(adata: ad.AnnData, layer: str, use_x: bool):
    if use_x:
        logging.info("Using adata.X for expression values")
        return adata.X, "X"

    if layer not in adata.layers:
        raise KeyError(
            f"Layer {layer!r} not found in input h5ad. Available layers: "
            f"{list(adata.layers.keys())}. Pass --use-x to aggregate adata.X."
        )
    logging.info("Using adata.layers[%r] for expression values", layer)
    return adata.layers[layer], layer


def validate_obs_columns(
    obs: pd.DataFrame,
    individual_col: str,
    cell_type_col: str,
) -> None:
    missing = [c for c in (individual_col, cell_type_col) if c not in obs.columns]
    if missing:
        raise ValueError(
            f"Input h5ad obs is missing required column(s): {missing}. "
            f"Available obs columns: {obs.columns.tolist()}"
        )


def prepare_grouping(
    obs: pd.DataFrame,
    individual_col: str,
    cell_type_col: str,
) -> tuple[pd.DataFrame, np.ndarray]:
    keep = obs[individual_col].notna() & obs[cell_type_col].notna()
    if not bool(keep.all()):
        logging.warning(
            "Dropping %d cells with missing %s or %s",
            int((~keep).sum()),
            individual_col,
            cell_type_col,
        )

    grouping = pd.DataFrame(
        {
            "individual_id": obs.loc[keep, individual_col].astype(str).to_numpy(),
            "cell_type": obs.loc[keep, cell_type_col].astype(str).to_numpy(),
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


def apply_h5ad_aggregation(
    sums: sp.csr_matrix | np.ndarray,
    n_cells: np.ndarray,
    aggregation: str,
) -> sp.csr_matrix | np.ndarray:
    if aggregation == "sum":
        return sums
    if aggregation == "mean":
        if sp.issparse(sums):
            return sums.multiply(1.0 / n_cells[:, None]).tocsr()
        return sums / n_cells[:, None]
    raise ValueError(f"Unknown aggregation: {aggregation}")


def build_pseudobulk_anndata(
    sums: sp.csr_matrix | np.ndarray,
    obs: pd.DataFrame,
    var: pd.DataFrame,
    aggregation: str,
    source_input: Path,
    matrix_source: str,
) -> ad.AnnData:
    X = apply_h5ad_aggregation(
        sums=sums.copy() if sp.issparse(sums) else sums.copy(),
        n_cells=obs["n_cells"].to_numpy(),
        aggregation=aggregation,
    )

    out_obs = obs.copy()
    out_obs["individual_id"] = out_obs["individual_id"].astype("category")
    out_obs["cell_type"] = out_obs["cell_type"].astype("category")
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

    out_var = var.copy()
    out_var.index = out_var.index.astype(str)

    out = ad.AnnData(X=X, obs=out_obs, var=out_var)
    out.uns["aggregation"] = aggregation
    out.uns["matrix_source"] = matrix_source
    out.uns["source"] = "src/preprocessing/onek1k/get_ge_per_ind.py"
    out.uns["source_input"] = str(source_input)
    return out


def select_csv_individuals(
    obs: pd.DataFrame,
    requested: list[str],
    all_individuals: bool,
    missing_individuals: str,
) -> list[str]:
    available = set(obs["individual_id"].astype(str).unique().tolist())
    if all_individuals:
        selected = sorted(available)
        logging.info("Using all %d donors for CSV output", len(selected))
        return selected

    missing = [individual_id for individual_id in requested if individual_id not in available]
    if missing:
        msg = (
            f"{len(missing)} of {len(requested)} requested donors are not present "
            f"in the input: {missing[:10]}"
        )
        if missing_individuals == "error":
            raise ValueError(msg)
        logging.warning("%s. Dropping missing donors.", msg)

    selected = [individual_id for individual_id in requested if individual_id in available]
    if not selected:
        raise ValueError("None of the requested donors are present in the input.")

    logging.info(
        "Using %d of %d requested donors for CSV output: %s",
        len(selected),
        len(requested),
        selected,
    )
    return selected


def row_to_dense(row: sp.spmatrix | np.ndarray) -> np.ndarray:
    if sp.issparse(row):
        return np.asarray(row.toarray()).ravel()
    return np.asarray(row).ravel()


def build_target_csv(
    sums: sp.csr_matrix | np.ndarray,
    obs: pd.DataFrame,
    genes: pd.Index,
    selected_individuals: list[str],
    mean_mode: str,
) -> pd.DataFrame:
    selected = set(selected_individuals)
    sub_obs = obs[obs["individual_id"].astype(str).isin(selected)].copy()
    if sub_obs.empty:
        raise ValueError("No pseudo-bulk rows remain for the selected CSV donors.")

    cell_types = sorted(sub_obs["cell_type"].astype(str).unique().tolist())
    values = np.zeros((len(cell_types), len(genes)), dtype=np.float32)

    for row_idx, cell_type in enumerate(cell_types):
        group_rows = sub_obs.index[sub_obs["cell_type"].astype(str) == cell_type].to_numpy()
        if len(group_rows) == 0:
            continue

        if mean_mode == "cell":
            total_cells = float(obs.loc[group_rows, "n_cells"].sum())
            if sp.issparse(sums):
                summed = sums[group_rows, :].sum(axis=0)
            else:
                summed = sums[group_rows, :].sum(axis=0)
            values[row_idx] = row_to_dense(summed) / total_cells
        elif mean_mode == "individual":
            n_cells = obs.loc[group_rows, "n_cells"].to_numpy(dtype=np.float32)
            if sp.issparse(sums):
                means = sums[group_rows, :].multiply(1.0 / n_cells[:, None]).tocsr()
                values[row_idx] = row_to_dense(means.mean(axis=0))
            else:
                means = sums[group_rows, :] / n_cells[:, None]
                values[row_idx] = np.asarray(means.mean(axis=0)).ravel()
        else:
            raise ValueError(f"Unknown CSV mean mode: {mean_mode}")

    target = pd.DataFrame(values, index=cell_types, columns=genes.astype(str))
    target.index.name = "cell_type"

    if target.columns.has_duplicates:
        n_duplicates = int(target.columns.duplicated().sum())
        logging.warning(
            "CSV gene axis has %d duplicate gene IDs; collapsing duplicates by mean",
            n_duplicates,
        )
        target = target.T.groupby(level=0, sort=False).mean().T

    return target


def write_h5ad(path: Path, adata: ad.AnnData) -> None:
    logging.info("Writing pseudo-bulk h5ad to %s", path)
    path.parent.mkdir(parents=True, exist_ok=True)
    adata.write_h5ad(path, compression="gzip")


def write_csv(path: Path, target: pd.DataFrame) -> None:
    logging.info("Writing target CSV to %s", path)
    path.parent.mkdir(parents=True, exist_ok=True)
    target.to_csv(path)


def main() -> None:
    args = parse_args()
    setup_logging(args.verbose)

    input_path = args.input.expanduser().resolve()
    if not input_path.exists():
        print(f"Error: input file does not exist: {input_path}", file=sys.stderr)
        sys.exit(1)

    output_h5ad = (
        args.output_h5ad.expanduser().resolve()
        if args.output_h5ad is not None
        else None
    )
    output_csv = (
        args.output_csv.expanduser().resolve()
        if args.output_csv is not None
        else None
    )

    try:
        check_output_path(output_h5ad, args.overwrite)
        check_output_path(output_csv, args.overwrite)

        logging.info("Reading %s", input_path)
        adata = ad.read_h5ad(input_path)
        validate_obs_columns(adata.obs, args.individual_col, args.cell_type_col)
        X, matrix_source = load_matrix(adata, args.layer, args.use_x)

        grouping, keep_cells = prepare_grouping(
            adata.obs,
            individual_col=args.individual_col,
            cell_type_col=args.cell_type_col,
        )
        if not bool(keep_cells.all()):
            X = X[keep_cells, :]

        sums, obs = aggregate_by_individual_and_cell_type(X, grouping)
        logging.info(
            "Pseudo-bulk shape before gene axis: %d groups x %d genes",
            obs.shape[0],
            adata.n_vars,
        )

        if output_h5ad is not None:
            pseudobulk = build_pseudobulk_anndata(
                sums=sums,
                obs=obs,
                var=adata.var,
                aggregation=args.aggregation,
                source_input=input_path,
                matrix_source=matrix_source,
            )
            write_h5ad(output_h5ad, pseudobulk)
            logging.info(
                "Wrote AnnData with shape %s and aggregation=%s",
                pseudobulk.shape,
                args.aggregation,
            )

        if output_csv is not None:
            requested = args.target_individual or list(INDIVIDUAL_IDS)
            selected_individuals = select_csv_individuals(
                obs=obs,
                requested=requested,
                all_individuals=args.all_individuals,
                missing_individuals=args.missing_individuals,
            )
            target = build_target_csv(
                sums=sums,
                obs=obs,
                genes=adata.var.index,
                selected_individuals=selected_individuals,
                mean_mode=args.csv_mean_mode,
            )
            write_csv(output_csv, target)
            logging.info(
                "Wrote target CSV with shape %s using csv_mean_mode=%s",
                target.shape,
                args.csv_mean_mode,
            )

    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)

    logging.info("Done.")


if __name__ == "__main__":
    main()
