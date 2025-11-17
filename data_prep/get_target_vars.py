from __future__ import annotations
import argparse
import logging
import scanpy as sc
import pandas as pd
import numpy as np
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

def prepare_target_vars(data: sc.AnnData) -> pd.DataFrame:
    """Transform raw data into target variables."""
    logging.info("Preparing target variables")

    cell_types = data.obs['cell_type']
    cell_types_unique = cell_types.unique().tolist()
    logging.debug("Identified cell types: %s", cell_types_unique)

    Y = pd.DataFrame(
        index=cell_types_unique, # cell types
        columns = data.var_names # genes
    )

    for ct in Y.index:
        idx = np.where(cell_types == ct)[0]
        Y.loc[ct] = data.X[idx].mean(axis=0).A1 # matrices are sparse
    
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
    targets = prepare_target_vars(data)

    save_output(targets, args.output)
    logging.info("Done.")

if __name__ == "__main__":
    main()