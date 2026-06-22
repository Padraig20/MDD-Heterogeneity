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
get_ground_truth.py

Build one ground-truth expression CSV per OneK1K cell type from a pseudo-bulk
h5ad file and the gene metadata CSV produced by src/preprocessing/get_obs_vars.py.

Each output CSV has rows=genes and columns:

gene,chrom,tss,<individual 1>,<individual 2>,...

The `gene` column is the ENSID from the obs-vars CSV.
"""


REQUIRED_GENE_COLUMNS = ("chrom", "ensid", "tss")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create per-cell-type ground-truth expression CSVs from a OneK1K "
            "pseudo-bulk h5ad file."
        )
    )
    parser.add_argument(
        "-i",
        "--input-h5ad",
        type=Path,
        required=True,
        help=(
            "Input pseudo-bulk h5ad. obs must contain individual_id and "
            "cell_type columns; var.index must contain ENSIDs."
        ),
    )
    parser.add_argument(
        "-g",
        "--gene-metadata",
        type=Path,
        required=True,
        help=(
            "CSV produced by src/preprocessing/get_obs_vars.py. Required "
            "columns: chrom, ensid, tss."
        ),
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        required=True,
        help="Directory where one subdirectory per cell type will be written.",
    )
    parser.add_argument(
        "--individual-col",
        default="individual_id",
        help='obs column containing individual IDs. Default: "individual_id".',
    )
    parser.add_argument(
        "--cell-type-col",
        default="cell_type",
        help='obs column containing cell-type labels. Default: "cell_type".',
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing output CSVs.",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="count",
        default=0,
        help="Increase verbosity.",
    )
    return parser.parse_args()


def setup_logging(verbosity: int) -> None:
    level = logging.INFO
    if verbosity >= 1:
        level = logging.DEBUG
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )


def validate_obs_columns(
    obs: pd.DataFrame,
    individual_col: str,
    cell_type_col: str,
) -> None:
    missing = [col for col in (individual_col, cell_type_col) if col not in obs.columns]
    if missing:
        raise ValueError(
            f"Input h5ad obs is missing required column(s): {missing}. "
            f"Available obs columns: {obs.columns.tolist()}"
        )


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


def align_gene_metadata_to_h5ad(
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
    in_h5ad = gene_meta["gene"].map(gene_to_pos).notna()
    n_missing = int((~in_h5ad).sum())
    if n_missing:
        missing_examples = gene_meta.loc[~in_h5ad, "gene"].head(10).tolist()
        logging.warning(
            "Dropping %d / %d genes from metadata because they are absent from h5ad var.index. "
            "Examples: %s",
            n_missing,
            len(gene_meta),
            missing_examples,
        )

    aligned = gene_meta.loc[in_h5ad].reset_index(drop=True)
    if aligned.empty:
        raise ValueError("No genes from the metadata CSV were found in h5ad var.index.")

    var_positions = np.fromiter(
        (gene_to_pos[gene] for gene in aligned["gene"]),
        dtype=np.int64,
        count=len(aligned),
    )
    logging.info("Using %d genes present in both metadata CSV and h5ad", len(aligned))
    return aligned, var_positions


def unique_labels(values: pd.Series) -> list[str]:
    return sorted(values.dropna().astype(str).unique().tolist())


def make_unique_cell_type_dirs(cell_types: list[str]) -> dict[str, str]:
    used: dict[str, int] = {}
    mapping: dict[str, str] = {}

    for cell_type in cell_types:
        count = used.get(cell_type, 0)
        used[cell_type] = count + 1
        mapping[cell_type] = cell_type if count == 0 else f"{cell_type}_{count + 1}"

    return mapping


def to_dense_array(matrix) -> np.ndarray:
    if sp.issparse(matrix):
        return np.asarray(matrix.toarray())
    return np.asarray(matrix)


def write_cell_type_csv(
    *,
    adata: ad.AnnData,
    gene_meta: pd.DataFrame,
    var_positions: np.ndarray,
    cell_type: str,
    output_path: Path,
    all_individuals: list[str],
    individual_col: str,
    cell_type_col: str,
    overwrite: bool,
) -> None:
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"Output file already exists: {output_path}. Use --overwrite.")

    obs = adata.obs
    ct_mask = obs[cell_type_col].astype(str).to_numpy() == cell_type
    ct_obs = obs.loc[ct_mask, [individual_col]]
    if ct_obs.empty:
        raise ValueError(f"No h5ad obs rows found for cell_type={cell_type!r}")

    duplicated = ct_obs[individual_col].astype(str).duplicated()
    if bool(duplicated.any()):
        examples = ct_obs.loc[duplicated, individual_col].astype(str).head(10).tolist()
        raise ValueError(
            "Input h5ad contains duplicate rows for the same "
            f"(individual, cell_type={cell_type!r}). Examples: {examples}"
        )

    obs_positions = np.flatnonzero(ct_mask)
    individuals_present = ct_obs[individual_col].astype(str).tolist()
    individual_to_col = {individual: pos for pos, individual in enumerate(all_individuals)}
    present_cols = np.fromiter(
        (individual_to_col[individual] for individual in individuals_present),
        dtype=np.int64,
        count=len(individuals_present),
    )

    logging.info(
        "Writing %s (%d present individuals, %d total columns)",
        output_path,
        len(individuals_present),
        len(all_individuals),
    )

    # Slice rows first so backed dense h5ad files do not need fancy indexing on
    # both axes at once. Then select the requested genes in metadata order.
    expression = adata.X[obs_positions, :]
    expression = expression[:, var_positions]
    expression = to_dense_array(expression).astype(np.float32, copy=False).T

    values = np.full((len(gene_meta), len(all_individuals)), np.nan, dtype=np.float32)
    values[:, present_cols] = expression

    all_individuals_harmonized = all_individuals.copy()

    for i in range(len(all_individuals_harmonized)):
        ids = all_individuals_harmonized[i].split("_")
        if len(ids) == 2:
            all_individuals_harmonized[i] = f"OneK1K_{ids[1]}"
        else:
            raise ValueError(f"Unexpected individual ID format: {all_individuals_harmonized[i]!r}")

    out = pd.concat(
        [
            gene_meta.reset_index(drop=True),
            pd.DataFrame(values, columns=all_individuals_harmonized),
        ],
        axis=1,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_path, index=False)


def build_ground_truth_csvs(
    input_h5ad: Path,
    gene_metadata: Path,
    output_dir: Path,
    individual_col: str,
    cell_type_col: str,
    overwrite: bool,
) -> None:
    gene_meta = load_gene_metadata(gene_metadata)

    logging.info("Reading h5ad from %s", input_h5ad)
    adata = ad.read_h5ad(input_h5ad, backed="r")
    try:
        validate_obs_columns(adata.obs, individual_col, cell_type_col)
        gene_meta, var_positions = align_gene_metadata_to_h5ad(gene_meta, adata.var.index)

        obs = adata.obs
        keep = obs[individual_col].notna() & obs[cell_type_col].notna()
        if not bool(keep.all()):
            logging.warning(
                "Ignoring %d obs rows with missing %s or %s",
                int((~keep).sum()),
                individual_col,
                cell_type_col,
            )
            adata = adata[keep.to_numpy(), :]
            obs = adata.obs

        all_individuals = unique_labels(obs[individual_col])
        cell_types = unique_labels(obs[cell_type_col])
        if not all_individuals:
            raise ValueError(f"No individuals found in obs column {individual_col!r}.")
        if not cell_types:
            raise ValueError(f"No cell types found in obs column {cell_type_col!r}.")

        ct_to_dir = make_unique_cell_type_dirs(cell_types)
        output_dir.mkdir(parents=True, exist_ok=True)

        logging.info(
            "Preparing %d cell-type CSVs with %d individual columns",
            len(cell_types),
            len(all_individuals),
        )
        for cell_type in cell_types:
            output_path = output_dir / f"{ct_to_dir[cell_type]}.csv"
            write_cell_type_csv(
                adata=adata,
                gene_meta=gene_meta,
                var_positions=var_positions,
                cell_type=cell_type,
                output_path=output_path,
                all_individuals=all_individuals,
                individual_col=individual_col,
                cell_type_col=cell_type_col,
                overwrite=overwrite,
            )
    finally:
        if adata.isbacked:
            adata.file.close()


def main() -> None:
    args = parse_args()
    setup_logging(args.verbose)
    logging.debug("Arguments: %s", args)

    build_ground_truth_csvs(
        input_h5ad=args.input_h5ad,
        gene_metadata=args.gene_metadata,
        output_dir=args.output_dir,
        individual_col=args.individual_col,
        cell_type_col=args.cell_type_col,
        overwrite=args.overwrite,
    )

    logging.info("Done.")


if __name__ == "__main__":
    main()
