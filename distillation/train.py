from __future__ import annotations
import argparse
import logging
from pathlib import Path

import os
import numpy as np
import pandas as pd
import tqdm
from tqdm import tqdm

from distillation.models.lr import LR
from distillation.dataset import GenotypeDataset
from distillation.wandb_logger import WandBLogger

"""
train.py

We train a model to predict cell type-specific gene expression from
genotype data (SNP dosages).
"""

def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Train model to predict cell type-specific gene expression from genotype data."
    )
    parser.add_argument(
        "-X", "--observations",
        type=Path,
        required=True,
        help="Path to input directory; should contain genotype data for each chromosome, e.g. 'chr1.bim/fam/bed'."
    )
    parser.add_argument(
        "-y", "--targets",
        type=Path,
        required=True,
        help="Path to target directory; should contain a CSV file for each cell-type."
    )
    parser.add_argument(
        "-m", "--model-name",
        type=str,
        default="elasticnet",
        choices=["elasticnet", "ridge"],
        help="Name of the model to train. List of available models will be expanded in the future."
    )
    parser.add_argument(
        "-mi", "--max_iter",
        type=int,
        default=10000,
        help="Maximum number of iterations for training."
    )
    parser.add_argument(
        "-a", "--alphas",
        type=int,
        default=5,
        help="Number of alpha values to try for CV."
    )
    parser.add_argument(
        "-v", "--verbose",
        action="count",
        default=0,
        help="Increase verbosity."
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="Name of the WandB run. Leave empty to not use WandB logging."
    )
    parser.add_argument(
        "-sg", "--select_genes",
        type=Path,
        default=None,
        help="Path to a file containing a list of genes to select. Will only train on these genes."
    )
    parser.add_argument(
        "-nt", "--norm-targets",
        type=str,
        default="log",
        choices=["log", "percentiles"],
        help="Normalization method for target labels."
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for splitting the data."
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

def main() -> None:
    args = parse_args()
    setup_logging(args.verbose)
    logging.debug("Arguments: %s", args)

    if args.run_name is not None:
        wb_logger = WandBLogger(enabled=True, run_name=args.run_name)
    else:
        wb_logger = WandBLogger(enabled=False)

    all_chromosomes = [f"chr{c}" for c in range(1, 23)] # we only use autosomes

    bims    = {}
    idx2ind = {}

    for chrom in all_chromosomes:
        if not os.path.exists(os.path.join(args.observations, f"{chrom}.bim")) or \
           not os.path.exists(os.path.join(args.observations, f"{chrom}.fam")):
            logging.warning(f"Missing genotype data for {chrom}. Aborting! :(")
            raise FileNotFoundError(f"Missing genotype data for {chrom} in {args.observations}")

        bim = pd.read_csv(
            os.path.join(args.observations, f"{chrom}.bim"),
            sep=r"\s+",
            header=None,
            names=["chrom", "snp", "cm", "bp", "a1", "a2"],
            dtype={"chrom": str, "snp": str, "bp": np.int64},
        )
        
        idx2ind_arr = pd.read_csv(
            os.path.join(args.observations, f"{chrom}.fam"),
            sep=r"\s+",
            header=None,
            usecols=[0, 1],
            names=["family_id", "individual_id"]
        )
        idx2ind_arr = idx2ind_arr["individual_id"].to_numpy()

        bims[chrom]    = bim
        idx2ind[chrom] = idx2ind_arr

        logging.info(f"Loaded genotype data for {chrom}.")

    cell_type_files = os.listdir(args.targets)
    logging.info(f"Found {len(cell_type_files)} cell types!")

    for ct_file in tqdm(cell_type_files, desc="Processing cell types"):
        ct_name = ct_file[:-4] # remove .csv extension
        logging.info(f"Processing cell type: {ct_name}")

        dataset = GenotypeDataset(bims=bims, idx2ind=idx2ind, y=os.path.join(args.targets,ct_file), bim_dir=args.observations, select_genes=args.select_genes, normalize=args.norm_targets)
        dataset = dataset.split_by_chromosome(all_chromosomes) # filter to only autosomes
        model   = LR(model_name=args.model_name, max_iter=args.max_iter, alphas=args.alphas, seed=args.seed)

        model.fit_dataset(dataset, verbose=args.verbose > 0)

        df = model.summarize_models()
        wb_logger.log_celltype_diagnostics(df, cell_type=ct_name)

    wb_logger.finish()

    logging.info("Done.")

if __name__ == "__main__":
    main()