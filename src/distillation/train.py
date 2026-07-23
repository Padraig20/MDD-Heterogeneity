from __future__ import annotations
import argparse
import logging
import sys
from pathlib import Path

import os
import numpy as np
import pandas as pd
from tqdm import tqdm

from src.distillation.models.lr import LR
from src.distillation.models.probabilistic_lr import ProbabilisticLR
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
        "--cell-types",
        type=str,
        nargs="+",
        default=None,
        metavar="CELL_TYPE",
        help=(
            "Restrict training to one or more specific cell types, given by their "
            "exact name (i.e. the CSV filename in --targets without the '.csv' "
            "extension), e.g. --cell-types \"memory B cell\" \"erythrocyte\". "
            "Defaults to training on every cell type found in --targets. Exits "
            "with an error if any requested cell type cannot be found."
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
        "-l1", "--l1-ratio",
        type=float,
        default=0.5,
        help=(
            "ElasticNet L1/L2 mixing ratio for the point-estimate model (--model-name "
            "elasticnet); 1.0 is pure Lasso (sparser), 0.0 is pure Ridge (denser, more "
            "shrinkage of correlated SNPs). scPrediXcan uses 0.5. Increase towards 1.0 "
            "for sparser, more outlier-robust fits if Pearson r lags Spearman rho."
        ),
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
        "--min-detected-frac",
        type=float,
        default=None,
        help=(
            "Minimum fraction of individuals with nonzero raw expression a gene must "
            "have to be trained/evaluated (e.g. 0.2 for >=20%% detection). Filters out "
            "near-all-zero, dropout-dominated genes, computed on *all* training "
            "individuals *before* --max-individuals is applied (so the gene set is "
            "stable across training-set-size ablations). Applied only to --targets "
            "(never to --targets-test), so the held-out evaluation always covers "
            "exactly the genes the model was trained on. Defaults to no filtering."
        ),
    )
    parser.add_argument(
        "--min-expr-std",
        type=float,
        default=None,
        help=(
            "Minimum standard deviation of log1p(expression) across individuals a gene "
            "must have to be trained/evaluated. Filters out near-constant genes (no "
            "signal to predict). Same population/timing semantics as "
            "--min-detected-frac. Defaults to no filtering."
        ),
    )
    parser.add_argument(
        "--aleatoric",
        type=Path,
        default=None,
        help=(
            "Path to a directory of teacher target *aleatoric* variances (one CSV per "
            "cell type, same names/format as --targets, e.g. the 'aleatoric' output of "
            "get_student_data.py). When provided together with --epistemic, a "
            "probabilistic model is distilled: three independent elastic-net/ridge "
            "fits per gene (mean, aleatoric variance, epistemic variance) directly "
            "against the teacher's own uncertainty decomposition, instead of a single "
            "point-estimate elasticnet/ridge model."
        ),
    )
    parser.add_argument(
        "--epistemic",
        type=Path,
        default=None,
        help=(
            "Path to a directory of teacher target *epistemic* variances, same format "
            "as --aleatoric (e.g. the 'epistemic' output of get_student_data.py). Must "
            "be provided together with --aleatoric."
        ),
    )
    parser.add_argument(
        "--aleatoric-test",
        type=Path,
        default=None,
        help=(
            "Path to an optional directory of teacher *aleatoric* variances for the "
            "held-out test set (used together with --targets-test), same format/"
            "filenames as --aleatoric. Only relevant for probabilistic distillation "
            "(--aleatoric/--epistemic). If omitted while --targets-test is set, the "
            "held-out evaluation reuses --aleatoric, matched by individual, which is "
            "only correct if that file also covers the test individuals."
        ),
    )
    parser.add_argument(
        "--epistemic-test",
        type=Path,
        default=None,
        help=(
            "Path to an optional directory of teacher *epistemic* variances for the "
            "held-out test set, same format/filenames as --epistemic. Same fallback "
            "semantics as --aleatoric-test."
        ),
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

    if args.cell_types is not None:
        available_cell_types = {f[:-4]: f for f in cell_type_files}
        missing_cell_types = [ct for ct in args.cell_types if ct not in available_cell_types]
        if missing_cell_types:
            logging.error(
                "Requested cell type(s) not found in '%s': %s. Available cell types: %s",
                args.targets, missing_cell_types, sorted(available_cell_types.keys()),
            )
            sys.exit(1)
        cell_type_files = [available_cell_types[ct] for ct in args.cell_types]
        logging.info(
            "Restricting training to %d requested cell type(s): %s",
            len(cell_type_files), args.cell_types,
        )

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

    probabilistic = args.aleatoric is not None or args.epistemic is not None
    if probabilistic and (args.aleatoric is None or args.epistemic is None):
        logging.error("--aleatoric and --epistemic must both be provided together.")
        sys.exit(1)
    if probabilistic:
        logging.info(
            "Probabilistic distillation enabled: fitting three independent elastic-net/"
            "ridge regressions per gene (mean, aleatoric variance, epistemic variance) "
            "directly against the teacher's own uncertainty decomposition.",
        )
        if (args.aleatoric_test is None) != (args.epistemic_test is None):
            logging.warning(
                "Only one of --aleatoric-test/--epistemic-test was provided; ignoring "
                "it and falling back to --aleatoric/--epistemic for held-out evaluation."
            )

    for ct_file in tqdm(cell_type_files, desc="Processing cell types"):
        ct_name = ct_file[:-4]  # remove .csv extension
        logging.info("Processing cell type: %s", ct_name)

        y_aleatoric = os.path.join(args.aleatoric, ct_file) if probabilistic else None
        y_epistemic = os.path.join(args.epistemic, ct_file) if probabilistic else None
        if probabilistic and not (os.path.exists(y_aleatoric) and os.path.exists(y_epistemic)):
            logging.warning(
                "No aleatoric/epistemic file for cell type '%s' at %s / %s; skipping.",
                ct_name, y_aleatoric, y_epistemic,
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
            y_aleatoric=y_aleatoric,
            y_epistemic=y_epistemic,
            min_detected_frac=args.min_detected_frac,
            min_expr_std=args.min_expr_std,
        )
        if args.min_detected_frac is not None or args.min_expr_std is not None:
            logging.info(
                "Cell type '%s': %d genes pass expression filter "
                "(min_detected_frac=%s, min_expr_std=%s).",
                ct_name, len(dataset.genes), args.min_detected_frac, args.min_expr_std,
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
                y_aleatoric_test = y_aleatoric
                y_epistemic_test = y_epistemic
                if probabilistic and args.aleatoric_test is not None and args.epistemic_test is not None:
                    candidate_aleatoric = os.path.join(args.aleatoric_test, ct_file)
                    candidate_epistemic = os.path.join(args.epistemic_test, ct_file)
                    if os.path.exists(candidate_aleatoric) and os.path.exists(candidate_epistemic):
                        y_aleatoric_test = candidate_aleatoric
                        y_epistemic_test = candidate_epistemic
                    else:
                        logging.warning(
                            "No held-out aleatoric/epistemic file for cell type '%s' at "
                            "%s / %s; falling back to --aleatoric/--epistemic (%s / %s) "
                            "for the held-out evaluation.",
                            ct_name, candidate_aleatoric, candidate_epistemic,
                            y_aleatoric, y_epistemic,
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
                    y_aleatoric=y_aleatoric_test,
                    y_epistemic=y_epistemic_test,
                )

        if probabilistic:
            model = ProbabilisticLR(
                model_name=args.model_name,
                l1_ratio=args.l1_ratio,
                max_iter=args.max_iter,
                alphas=args.alphas,
                seed=args.seed,
                n_jobs=jobs,
            )
        else:
            model = LR(
                model_name=args.model_name,
                l1_ratio=args.l1_ratio,
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

        model_label = "Probabilistic Elastic Net" if probabilistic else args.model_name.capitalize()
        df = model.summarize_models()
        wb_logger.log_celltype_diagnostics(df, cell_type=ct_name, model_label=model_label)

    wb_logger.finish()

    logging.info("Done.")

if __name__ == "__main__":
    main()