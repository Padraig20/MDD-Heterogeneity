from __future__ import annotations
import argparse
import logging
import scanpy as sc
import pandas as pd
import numpy as np
import re
from pathlib import Path

"""
get_target_vars.py

Script that takes input from the Human Brain Cell Atlas v1.0 and extracts
target variables for downstream asnalysis.

https://cellxgene.cziscience.com/collections/283d65eb-dd53-496d-adb7-7570c7caa443
"""

def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Prepare target variables for downstream analysis."
    )
    parser.add_argument(
        "-i", "--input",
        type=Path,
        required=True,
        help="Path to input file (*.h5ad)."
    )
    parser.add_argument(
        "-o", "--output",
        type=Path,
        required=True,
        help="Path to write output file (*.csv)."
    )
    parser.add_argument(
        "-v", "--verbose",
        action="count",
        default=0,
        help="Increase verbosity."
    )
    parser.add_argument(
        "-chr", "--chromosome",
        default=False,
        action="store_true",
        help="Specify if should be grouped by chromosome."
    )
    parser.add_argument(
        "--mdd_genes",
        type=Path,
        default=None,
        help="If provided, path to MDD gene list (TSV) to filter target variables."
    )
    return parser.parse_args()

def setup_logging(verbosity: int) -> None:
    """Configure basic logging based on verbosity level."""
    level = logging.INFO
    if verbosity >= 1:
        level = logging.DEBUG
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

def load_data(input_path: Path) -> sc.AnnData:
    """Load data from `input_path`."""
    logging.info("Loading data from %s", input_path)
    return sc.read_h5ad(input_path)

# chromosomes are named weirly sometimes
# we don't like that, so we map back (if possible) to standard names
# examples: 'chrUn_GL000219v1', 'chr1_KI270711v1', 'chrUn_GL000218v1', 'chrM'
def simplify_chr(c: str) -> str | None:
        m = re.match(r"^(chr(?:\d+|X|Y))\b", c)
        return m.group(1) if m else None

def prepare_target_vars(data: sc.AnnData, mdd_genes_path: Path | None, group_by_chromosome: bool = True) -> pd.DataFrame:
    """Transform raw data into target variables."""
    logging.info("Preparing target variables")

    cell_types = data.obs['cluster_id'] # dim <= 461 for sc human brain atlas v1.0
    cell_types_unique = cell_types.unique().tolist()
    logging.debug("Cell types: %s", cell_types_unique)

    gene_names = data.var.index # using Ensembl IDs
    gene_names_unique = gene_names.unique().tolist()
    logging.debug("Genes (first 10): %s", gene_names_unique[:10])

    if mdd_genes_path is not None:
        logging.info("Filtering target variables to MDD genes from %s", mdd_genes_path)
        mdd_genes_df = pd.read_csv(mdd_genes_path, sep="\t")
        mdd_gene_list = mdd_genes_df['ENSID'].tolist()
        mask_mdd = gene_names.isin(mdd_gene_list)
        gene_names = gene_names[mask_mdd]
        data = data[:, mask_mdd]
        logging.info("After filtering, %d genes of %d remain...", len(gene_names), len(mdd_gene_list))

    if group_by_chromosome:
        chromosome_names_raw = data.var['Chromosome']
        chromosome_names = chromosome_names_raw.map(simplify_chr)
        chromosome_names_unique = chromosome_names.unique().tolist()
        logging.debug("Chromosomes: %s", chromosome_names_unique)

        # combine cell type and chromosome into "cell_type,chromo"
        combined = [f"{ct},{ch}"
                    for ct in cell_types_unique
                    for ch in chromosome_names_unique
                    if ch is not None]
        logging.debug("Combined cell_type,chromo (unique): %s", combined)

        Y = pd.DataFrame(
            index=combined,       # cell type x chromosome
            columns=gene_names,   # genes
        )

        for ct_chr in Y.index:
            ct, ch = ct_chr.split(",")
            idx_cells = np.where(cell_types == ct)[0]
            if idx_cells.size == 0:
                continue
            means = data.X[idx_cells].mean(axis=0).A1
            mask_ch = (chromosome_names == ch).values
            row = np.full_like(means, 0, dtype=float)
            row[mask_ch] = means[mask_ch]
            Y.loc[ct_chr] = row

    else: # don't group by chromosome
        Y = pd.DataFrame(
            index=cell_types_unique,  # cell types
            columns=gene_names,       # genes
        )
        for ct in Y.index:
            idx = np.where(cell_types == ct)[0]
            Y.loc[ct] = data.X[idx].mean(axis=0).A1

    for g in gene_names_unique:
        cols = np.where(gene_names == g)[0]
        if len(cols) > 1: # collapse duplicate genes by averaging
            Y[g] = Y.iloc[:, cols].mean(axis=1)
    
    logging.info("Prepared target variables with shape %s", Y.shape)
    logging.debug("Target variables preview:\n%s", Y.head())

    return Y

def save_output(data: pd.DataFrame, output_path: Path) -> None:
    """Save processed data to `output_path`."""
    logging.info("Saving output to %s", output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(output_path)

def main() -> None:
    args = parse_args()
    setup_logging(args.verbose)
    logging.debug("Arguments: %s", args)

    data = load_data(args.input)
    targets = prepare_target_vars(data, args.mdd_genes, args.chromosome)

    save_output(targets, args.output)
    logging.info("Done.")

if __name__ == "__main__":
    main()