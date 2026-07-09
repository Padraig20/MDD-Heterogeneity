from __future__ import annotations
import argparse
import logging
from pathlib import Path

import os
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from src.distillation.models.lr import LR
from src.distillation.models.ensemble_lr import EnsembleLR
from src.distillation.dataset import GenotypeDataset
from src.distillation.wandb_logger import WandBLogger

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
        "-yt", "--targets-test",
        type=Path,
        default=None,
        help=(
            "Path to an optional held-out target directory (same format/filenames as "
            "--targets, one CSV per cell-type, disjoint individuals). If provided, models "
            "are trained on *all* individuals in --targets and evaluated on the "
            "individuals found here, instead of the automatic per-gene 80:20 split."
        ),
    )
    parser.add_argument(
        "-o", "--output-dir",
        type=Path,
        default=None,
        help="Directory to save the trained models. Creates one JSON file per cell type with non-zero coefficients for each gene."
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
        default=2000,
        help="Maximum number of iterations for training."
    )
    parser.add_argument(
        "-a", "--alphas",
        type=int,
        default=15,
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
    parser.add_argument(
        "-j", "--jobs",
        type=int,
        default=os.cpu_count() or 1,
        help=(
            "Number of parallel workers used per cell type to fit genes in "
            "parallel. Defaults to the number of CPUs."
        ),
    )
    parser.add_argument(
        "-ni", "--max-individuals",
        type=int,
        default=None,
        help="Maximum number of individuals to use for training. Defaults to all individuals.",
    )
    parser.add_argument(
        "-gt", "--genotype-template",
        type=str,
        default="UKB",
        choices=["OneK1K", "UKB"],
        help="Template for genotype data files."
    )
    parser.add_argument(
        "--maf-threshold",
        type=float,
        default=None,
        help=(
            "Minimum minor allele frequency to keep a SNP (e.g. 0.05 for MAF >= 5%%). "
            "MAF is computed across the loaded cohort. Defaults to no MAF filtering."
        ),
    )
    parser.add_argument(
        "--variance",
        type=Path,
        default=None,
        help=(
            "Path to a directory of teacher target *variances* (one CSV per cell type, "
            "same names/format as --targets, e.g. the 'totvar' output of get_student_data.py). "
            "When provided, a probabilistic linear deep ensemble is distilled by matching the "
            "teacher's (mean, variance) Gaussians via the 2-Wasserstein metric, instead of a "
            "point-estimate elasticnet/ridge model."
        ),
    )
    parser.add_argument(
        "--variance-test",
        type=Path,
        default=None,
        help=(
            "Path to an optional directory of teacher target *variances* for the "
            "held-out test set (used together with --targets-test), same format/"
            "filenames as --variance. Only relevant for probabilistic distillation "
            "(--variance). If omitted while --targets-test is set, the held-out "
            "evaluation reuses --variance, matched by individual, which is only "
            "correct if that file also covers the test individuals."
        ),
    )
    parser.add_argument(
        "--ensemble-size",
        type=int,
        default=5,
        help="Number of linear ensemble members for probabilistic distillation (--variance).",
    )
    parser.add_argument(
        "--prob-epochs",
        type=int,
        default=300,
        help="Number of full-batch gradient steps per gene for probabilistic distillation.",
    )
    parser.add_argument(
        "--prob-lr",
        type=float,
        default=1e-2,
        help="Learning rate for the probabilistic (deep-ensemble LR) distillation.",
    )
    parser.add_argument(
        "--prob-weight-decay",
        type=float,
        default=0.25,
        help="Weight decay (L2) for the probabilistic (deep-ensemble LR) distillation.",
    )
    parser.add_argument(
        "--prob-l1",
        type=float,
        default=0.75,
        help=(
            "Group-lasso strength on the mean head for probabilistic distillation "
            "(--variance). Groups the ensemble members sharing each SNP and applies a "
            "proximal step so whole SNP columns are driven to exactly zero, inducing "
            "elastic-net-style sparsity. 0 disables (dense L2-only fit); increase it to "
            "reduce the nonzero-weight count."
        ),
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help=(
            "Torch device for probabilistic distillation (--variance), e.g. 'cpu', 'cuda', "
            "or 'cuda:0'. Use 'auto' to pick 'cuda' when available. Ignored for the "
            "elasticnet/ridge (sklearn, CPU) path."
        ),
    )
    return parser.parse_args()

def resolve_device(device: str) -> torch.device:
    """Resolve a --device string to a torch.device, warning + falling back to CPU
    if CUDA was requested but is unavailable."""
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dev = torch.device(device)
    if dev.type == "cuda" and not torch.cuda.is_available():
        logging.warning("CUDA requested via --device '%s' but not available; using CPU.", device)
        return torch.device("cpu")
    return dev

def setup_logging(verbosity: int) -> None:
    """Configure basic logging based on verbosity level."""
    level = logging.INFO
    if verbosity >= 1:
        level = logging.DEBUG
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

ONEK1K_BED_TEMPLATE = "OneK1K.GrCH38_chr{chrom}.biallelic"
UKB_BED_TEMPLATE = "ukb_imp_v3_chr{chrom}.unrelatedbritishqced.maf001geno9.biallelic"

def main() -> None:
    args = parse_args()
    setup_logging(args.verbose)
    logging.debug("Arguments: %s", args)

    if args.run_name is not None:
        wb_logger = WandBLogger(enabled=True, run_name=args.run_name)
    else:
        wb_logger = WandBLogger(enabled=False)

    all_chromosomes = [str(c) for c in range(1, 23)]

    bims = {}
    idx2ind = {}

    bed_template = None
    if args.genotype_template == "OneK1K":
        bed_template = ONEK1K_BED_TEMPLATE
    elif args.genotype_template == "UKB":
        bed_template = UKB_BED_TEMPLATE

    for chrom in all_chromosomes:
        chrom_name = bed_template.format(chrom=chrom)
        bim_path = os.path.join(args.observations, f"{chrom_name}.bim")
        fam_path = os.path.join(args.observations, f"{chrom_name}.fam")
        if not os.path.exists(bim_path) or not os.path.exists(fam_path):
            logging.warning(f"Missing genotype data for {chrom}. Aborting! :(")
            raise FileNotFoundError(f"Missing genotype data for {chrom} in {args.observations}")

        bim = pd.read_csv(
            bim_path,
            sep=r"\s+",
            header=None,
            names=["chrom", "snp", "cm", "bp", "a1", "a2"],
            dtype={"chrom": str, "snp": str, "bp": np.int64},
        )
        
        idx2ind_arr = pd.read_csv(
            fam_path,
            sep=r"\s+",
            header=None,
            usecols=[0, 1],
            names=["family_id", "individual_id"]
        )
        idx2ind_arr = idx2ind_arr["individual_id"].to_numpy()
        if args.genotype_template == "UKB":
            idx2ind_arr = np.array([f"{ind}_{ind}" for ind in idx2ind_arr])

        bims[chrom]    = bim
        idx2ind[chrom] = idx2ind_arr

        logging.info(f"Loaded genotype data for '{chrom_name}'.")

    cell_type_files = os.listdir(args.targets)
    logging.info(f"Found {len(cell_type_files)} cell types!")

    use_external_test = args.targets_test is not None
    if use_external_test:
        logging.info(
            "Using held-out target directory '%s' for evaluation instead of the "
            "automatic 80:20 split.", args.targets_test,
        )

    if args.output_dir is not None:
        os.makedirs(args.output_dir, exist_ok=True)

    jobs = max(1, args.jobs)
    logging.info("Fitting genes per cell type with %d parallel workers.", jobs)

    probabilistic = args.variance is not None
    prob_device = "cpu"
    if probabilistic:
        prob_device = resolve_device(args.device)
        if prob_device.type == "cuda" and jobs > 1:
            logging.info(
                "Using CUDA for probabilistic distillation; gene fits share the GPU, "
                "so the %d workers mainly overlap CPU-side data prep with GPU compute.",
                jobs,
            )
        logging.info(
            "Probabilistic distillation enabled on device '%s': fitting a linear deep "
            "ensemble against teacher (mean, variance) Gaussians via the 2-Wasserstein metric.",
            prob_device,
        )

    for ct_file in tqdm(cell_type_files, desc="Processing cell types"):
        ct_name = ct_file[:-4]  # remove .csv extension
        logging.info("Processing cell type: %s", ct_name)

        y_var = os.path.join(args.variance, ct_file) if probabilistic else None
        if probabilistic and not os.path.exists(y_var):
            logging.warning(
                "No variance file for cell type '%s' at %s; skipping.", ct_name, y_var
            )
            continue

        dataset = GenotypeDataset(
            bims=bims,
            idx2ind=idx2ind,
            y=os.path.join(args.targets, ct_file),
            bim_dir=args.observations,
            select_genes=args.select_genes,
            normalize=args.norm_targets,
            max_individuals=args.max_individuals,
            bed_template=bed_template,
            maf_threshold=args.maf_threshold,
            y_var=y_var,
        )

        test_dataset = None
        if use_external_test:
            test_path = os.path.join(args.targets_test, ct_file)
            if not os.path.exists(test_path):
                logging.warning(
                    "No held-out target file for cell type '%s' at %s; falling back "
                    "to the automatic 80:20 split for this cell type.", ct_name, test_path,
                )
            else:
                y_var_test = y_var
                if probabilistic and args.variance_test is not None:
                    candidate = os.path.join(args.variance_test, ct_file)
                    if os.path.exists(candidate):
                        y_var_test = candidate
                    else:
                        logging.warning(
                            "No held-out variance file for cell type '%s' at %s; falling "
                            "back to --variance (%s) for the held-out evaluation.",
                            ct_name, candidate, y_var,
                        )
                test_dataset = GenotypeDataset(
                    bims=bims,
                    idx2ind=idx2ind,
                    y=test_path,
                    bim_dir=args.observations,
                    select_genes=args.select_genes,
                    normalize=args.norm_targets,
                    max_individuals=None,
                    bed_template=bed_template,
                    maf_threshold=args.maf_threshold,
                    y_var=y_var_test,
                )

        if probabilistic:
            model = EnsembleLR(
                n_models=args.ensemble_size,
                epochs=args.prob_epochs,
                lr=args.prob_lr,
                weight_decay=args.prob_weight_decay,
                l1=args.prob_l1,
                seed=args.seed,
                n_jobs=jobs,
                device=prob_device,
            )
        else:
            model = LR(
                model_name=args.model_name,
                max_iter=args.max_iter,
                alphas=args.alphas,
                seed=args.seed,
                n_jobs=jobs,
            )
        model.fit_dataset(dataset, test_dataset=test_dataset, verbose=args.verbose > 0)

        if args.output_dir is not None:
            ct_path = ct_name.replace(" ", "_")
            path = os.path.join(args.output_dir, f"{ct_path}.json")
            model.save_coefficients(path)

        model_label = "Linear Deep Ensemble" if probabilistic else args.model_name.capitalize()
        df = model.summarize_models()
        wb_logger.log_celltype_diagnostics(df, cell_type=ct_name, model_label=model_label)

    wb_logger.finish()

    logging.info("Done.")

if __name__ == "__main__":
    main()