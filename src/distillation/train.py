from __future__ import annotations
import argparse
import logging
import re
import sys
from pathlib import Path
from typing import Optional

import os
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from tqdm import tqdm

from src.distillation.models.lr import LR
from src.distillation.models.ensemble_lr import (
    EnsembleGenotypeDataset,
    EnsembleLR,
)
from src.distillation.models.probabilistic_lr import ProbabilisticLR
from src.distillation.dataset import GenotypeDataset
from src.distillation.utils import configure_convergence_warnings
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
        default=None,
        help=(
            "Path to an ordinary target directory containing one CSV per cell "
            "type. Mutually exclusive with --ensemble-members-dir."
        ),
    )
    parser.add_argument(
        "--ensemble-members-dir",
        type=Path,
        default=None,
        help=(
            "Per-member teacher output produced by get_student_data.py. Accepts "
            "either its output root or the nested members/ directory and expects "
            "member_<index>/{preds,sigmas}/<cell-type>.csv. Each member is "
            "distilled with bootstrapped, heteroskedastic elastic nets. Mutually "
            "exclusive with --targets."
        ),
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
        "--ensemble-members-test-dir",
        type=Path,
        default=None,
        help=(
            "Optional held-out per-member output root, with the same member IDs "
            "and layout as --ensemble-members-dir."
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
        help=(
            "Directory to save the trained models. Creates one JSON file per "
            "cell type; ensemble models retain SNPs meeting --pip-threshold."
        ),
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
        "--cv",
        type=int,
        default=3,
        help=(
            "Number of inner cross-validation folds used to select the penalty per "
            "gene. Cost scales with the fold count, and on a cis-window (n of a few "
            "hundred, p of tens of thousands) 3 folds pick effectively the same "
            "alpha as 5 at ~2/3 of the time."
        ),
    )
    parser.add_argument(
        "--screen-snps",
        type=int,
        default=5000,
        help=(
            "Before fitting, keep only this many SNPs per cis-window, the ones most "
            "correlated with the target. In ensemble mode, one precision-weighted "
            "screen is computed per gene and reused for every member and bootstrap. "
            "Use 0 to fit against every SNP in the window."
        ),
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
        choices=["none", "log", "percentiles"],
        help=(
            "Normalization for target means. Ensemble sigmas are never "
            "transformed; use 'log' for member files exported from a "
            "log-target teacher and 'none' for an untransformed teacher."
        ),
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
        "--celltype-batch",
        type=int,
        default=1,
        help=(
            "Number of cell types to fit together. Cell types drawn from the same "
            "cohort share their genotypes, so fitting them in a batch reads each "
            "gene's cis-window once for the whole batch instead of once per cell "
            "type. Costs one target matrix in memory per cell type in the batch, "
            "and only applies to cell types whose individuals match exactly. "
            "Defaults to 1 (no sharing)."
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
    parser.add_argument(
        "--ensemble-bootstraps",
        type=int,
        default=5,
        help=(
            "Number B of cohort bootstraps per teacher member. Ensemble "
            "distillation performs exactly B times M fixed-penalty elastic-net "
            "fits per gene after alpha selection."
        ),
    )
    parser.add_argument(
        "--ensemble-alpha-mode",
        choices=["shared", "member"],
        default="shared",
        help=(
            "How to tune the elastic-net penalty for ensemble distillation. "
            "'shared' performs one CV search per gene on the ensemble-mean "
            "target and reuses its alpha for every member/bootstrap; 'member' "
            "performs one CV search per member."
        ),
    )
    parser.add_argument(
        "--ensemble-alpha",
        type=float,
        default=None,
        help=(
            "Optional fixed elastic-net alpha for every member/bootstrap. "
            "This skips alpha CV entirely and overrides --ensemble-alpha-mode."
        ),
    )
    parser.add_argument(
        "--sigma-floor",
        type=float,
        default=1e-4,
        help=(
            "Floor applied to each teacher member's aleatoric standard "
            "deviation before computing inverse-variance sample weights."
        ),
    )
    parser.add_argument(
        "--pip-threshold",
        type=float,
        default=0.5,
        help=(
            "A SNP is written when the fraction of member-bootstrap elastic-net "
            "fits in which its coefficient is non-zero reaches this threshold."
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


def prepare_cell_type(
    ct_file: str,
    args: argparse.Namespace,
    bims: dict,
    idx2ind: dict,
    bed_template: str,
    probabilistic: bool,
    use_external_test: bool,
    screen_snps: int | None,
    jobs: int,
):
    """
    Build the dataset(s) and the (still unfitted) model for one cell type.

    Returns `(ct_name, dataset, test_dataset, model)`, or None when the cell type's
    inputs are incomplete and it has to be skipped.
    """
    ct_name = ct_file[:-4]  # remove .csv extension
    logging.info("Processing cell type: %s", ct_name)

    y_aleatoric = os.path.join(args.aleatoric, ct_file) if probabilistic else None
    y_epistemic = os.path.join(args.epistemic, ct_file) if probabilistic else None
    if probabilistic and not (os.path.exists(y_aleatoric) and os.path.exists(y_epistemic)):
        logging.warning(
            "No aleatoric/epistemic file for cell type '%s' at %s / %s; skipping.",
            ct_name, y_aleatoric, y_epistemic,
        )
        return None

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

    model_class = ProbabilisticLR if probabilistic else LR

    model = model_class(
        model_name=args.model_name,
        l1_ratio=args.l1_ratio,
        max_iter=args.max_iter,
        alphas=args.alphas,
        cv=args.cv,
        seed=args.seed,
        n_jobs=jobs,
        screen=screen_snps,
    )
    return ct_name, dataset, test_dataset, model


MEMBER_DIR_PATTERN = re.compile(r"^member_(\d+)$")


def discover_ensemble_members(
    root: Path,
) -> tuple[list[tuple[str, Path]], list[str]]:
    """Discover and validate a get_student_data per-member output tree."""
    root = Path(root)
    members_root = root / "members" if (root / "members").is_dir() else root
    discovered = []
    if members_root.is_dir():
        for child in members_root.iterdir():
            match = MEMBER_DIR_PATTERN.fullmatch(child.name)
            if match and child.is_dir():
                discovered.append((int(match.group(1)), child))
    discovered.sort(key=lambda item: item[0])
    if not discovered:
        raise ValueError(
            f"No member_<index> directories found under {members_root}."
        )

    indices = [index for index, _ in discovered]
    if indices != list(range(len(indices))):
        raise ValueError(
            "Ensemble member indices must start at 0 and be contiguous; found "
            f"{indices} under {members_root}."
        )

    reference_files = None
    members = []
    for index, member_dir in discovered:
        preds_dir = member_dir / "preds"
        sigmas_dir = member_dir / "sigmas"
        if not preds_dir.is_dir() or not sigmas_dir.is_dir():
            raise ValueError(
                f"{member_dir} must contain both preds/ and sigmas/."
            )
        pred_files = {path.name for path in preds_dir.glob("*.csv")}
        sigma_files = {path.name for path in sigmas_dir.glob("*.csv")}
        if pred_files != sigma_files:
            raise ValueError(
                f"Prediction/sigma cell-type files differ for {member_dir.name}."
            )
        if reference_files is None:
            reference_files = pred_files
        elif pred_files != reference_files:
            raise ValueError(
                "All ensemble members must contain the same cell-type CSVs."
            )
        members.append((str(index), member_dir))

    if not reference_files:
        raise ValueError(f"No cell-type CSVs found under {members_root}.")
    return members, sorted(reference_files)


def prepare_ensemble_cell_type(
    ct_file: str,
    args: argparse.Namespace,
    bims: dict,
    idx2ind: dict,
    bed_template: str,
    member_dirs: list[tuple[str, Path]],
    test_member_dirs: Optional[list[tuple[str, Path]]],
    screen_snps: int | None,
    jobs: int,
):
    """Build aligned member datasets and one EnsembleLR for a cell type."""
    ct_name = ct_file[:-4]
    logging.info(
        "Processing ensemble cell type %s with %d members.",
        ct_name,
        len(member_dirs),
    )

    member_datasets = []
    member_ids = []
    for member_id, member_dir in member_dirs:
        member_datasets.append(
            GenotypeDataset(
                bims=bims,
                idx2ind=idx2ind,
                y=member_dir / "preds" / ct_file,
                y_sigma=member_dir / "sigmas" / ct_file,
                bim_dir=args.observations,
                select_genes=args.select_genes,
                normalize=args.norm_targets,
                max_individuals=args.max_individuals,
                bed_template=bed_template,
                maf_threshold=args.maf_threshold,
                min_detected_frac=args.min_detected_frac,
                min_expr_std=args.min_expr_std,
            )
        )
        member_ids.append(member_id)
    dataset = EnsembleGenotypeDataset(member_datasets, member_ids)

    test_dataset = None
    if test_member_dirs is not None:
        test_ids = [member_id for member_id, _ in test_member_dirs]
        if test_ids != member_ids:
            raise ValueError(
                "Training and held-out ensemble member IDs must match exactly."
            )
        test_dataset = EnsembleGenotypeDataset(
            [
                GenotypeDataset(
                    bims=bims,
                    idx2ind=idx2ind,
                    y=member_dir / "preds" / ct_file,
                    y_sigma=member_dir / "sigmas" / ct_file,
                    bim_dir=args.observations,
                    select_genes=args.select_genes,
                    normalize=args.norm_targets,
                    max_individuals=None,
                    bed_template=bed_template,
                    maf_threshold=args.maf_threshold,
                )
                for _, member_dir in test_member_dirs
            ],
            test_ids,
        )

    model = EnsembleLR(
        l1_ratio=args.l1_ratio,
        cv=args.cv,
        alphas=args.alphas,
        max_iter=args.max_iter,
        n_bootstraps=args.ensemble_bootstraps,
        alpha_mode=args.ensemble_alpha_mode,
        alpha=args.ensemble_alpha,
        sigma_floor=args.sigma_floor,
        pip_threshold=args.pip_threshold,
        seed=args.seed,
        n_jobs=jobs,
        screen=screen_snps,
    )
    return ct_name, dataset, test_dataset, model


def can_share_genotypes(jobs: list) -> bool:
    """
    Whether a batch of prepared cell types can be fitted from shared design
    matrices, i.e. whether they all model exactly the same individuals in the same
    order over the same cis-windows.
    """
    reference = jobs[0][1]
    return all(reference.shares_individuals_with(dataset) for _, dataset, _, _ in jobs[1:])


def fit_batch_sharing_genotypes(jobs: list, n_jobs: int, verbose: bool) -> None:
    """
    Fit a batch of cell types gene-major: each gene's cis-window is read once and
    then handed to every cell type's model, instead of every cell type re-reading
    the same genotypes for the same gene.

    Parallelism stays at the gene level (one task fits all cell types for one gene)
    with BLAS pinned to one thread per call, as in the per-cell-type path.
    """
    genes = list(dict.fromkeys(gene for _, dataset, _, _ in jobs for gene in dataset.genes))
    n     = len(genes)

    configure_convergence_warnings(verbose)

    def fit_gene(gene: str, i: int) -> None:
        design = None
        for ct_name, dataset, test_dataset, model in jobs:
            if not dataset.has_gene(gene):
                continue
            try:
                if design is None:
                    design = dataset.gene_design(gene)
                model.fit_gene_from_design(dataset, gene, design, test_dataset=test_dataset)
            except Exception as e:
                if verbose:
                    print(f"[{i}/{n}] skip {gene} ({ct_name}): {e}")

    with threadpool_limits(limits=1):
        with ThreadPoolExecutor(max_workers=max(1, n_jobs)) as ex:
            futures = [ex.submit(fit_gene, gene, i) for i, gene in enumerate(genes, start=1)]
            iterator = as_completed(futures)
            if not verbose:
                iterator = tqdm(
                    iterator, total=len(futures), desc="Fitting genes", leave=False
                )
            for fut in iterator:
                fut.result()

def main() -> None:
    args = parse_args()
    setup_logging(args.verbose)
    logging.debug("Arguments: %s", args)

    ensemble_mode = args.ensemble_members_dir is not None
    if ensemble_mode == (args.targets is not None):
        logging.error(
            "Provide exactly one of --targets or --ensemble-members-dir."
        )
        sys.exit(1)
    aggregate_uncertainty_args = (
        args.aleatoric,
        args.epistemic,
        args.aleatoric_test,
        args.epistemic_test,
    )
    if ensemble_mode and any(
        value is not None for value in aggregate_uncertainty_args
    ):
        logging.error(
            "--aleatoric/--epistemic and their test variants are aggregate "
            "uncertainty targets and cannot be combined with "
            "--ensemble-members-dir."
        )
        sys.exit(1)
    if ensemble_mode and args.targets_test is not None:
        logging.error(
            "Use --ensemble-members-test-dir, not --targets-test, with "
            "--ensemble-members-dir."
        )
        sys.exit(1)
    if not ensemble_mode and args.ensemble_members_test_dir is not None:
        logging.error(
            "--ensemble-members-test-dir requires --ensemble-members-dir."
        )
        sys.exit(1)
    if ensemble_mode and args.norm_targets == "percentiles":
        logging.error(
            "Percentile-normalized means have no compatible transformation for "
            "the member sigmas; use --norm-targets log or none."
        )
        sys.exit(1)
    if ensemble_mode and args.model_name != "elasticnet":
        logging.error(
            "Ensemble-member bootstrap distillation requires "
            "--model-name elasticnet."
        )
        sys.exit(1)
    if ensemble_mode and args.ensemble_bootstraps < 1:
        logging.error("--ensemble-bootstraps must be at least 1.")
        sys.exit(1)
    if ensemble_mode and not 0.0 < args.l1_ratio <= 1.0:
        logging.error(
            "Ensemble-member bootstrap distillation requires --l1-ratio in "
            "(0, 1]; a nonzero L1 component is needed for empirical PIPs."
        )
        sys.exit(1)
    if ensemble_mode and (
        args.ensemble_alpha is not None
        and (not np.isfinite(args.ensemble_alpha) or args.ensemble_alpha <= 0.0)
    ):
        logging.error("--ensemble-alpha must be finite and positive.")
        sys.exit(1)
    if ensemble_mode and (
        not np.isfinite(args.sigma_floor) or args.sigma_floor <= 0.0
    ):
        logging.error("--sigma-floor must be finite and positive.")
        sys.exit(1)
    if ensemble_mode and not 0.0 <= args.pip_threshold <= 1.0:
        logging.error("--pip-threshold must be in [0, 1].")
        sys.exit(1)

    member_dirs = None
    test_member_dirs = None
    if ensemble_mode:
        member_dirs, cell_type_files = discover_ensemble_members(
            args.ensemble_members_dir
        )
        if args.ensemble_members_test_dir is not None:
            test_member_dirs, test_cell_type_files = discover_ensemble_members(
                args.ensemble_members_test_dir
            )
            if [item[0] for item in member_dirs] != [
                item[0] for item in test_member_dirs
            ]:
                raise ValueError(
                    "Training and held-out ensemble member IDs differ."
                )
            if set(cell_type_files) != set(test_cell_type_files):
                raise ValueError(
                    "Training and held-out ensemble cell-type files differ."
                )

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

    if not ensemble_mode:
        cell_type_files = [
            filename
            for filename in os.listdir(args.targets)
            if filename.endswith(".csv")
        ]
    logging.info(f"Found {len(cell_type_files)} cell types!")

    if args.cell_types is not None:
        available_cell_types = {f[:-4]: f for f in cell_type_files}
        missing_cell_types = [ct for ct in args.cell_types if ct not in available_cell_types]
        if missing_cell_types:
            logging.error(
                "Requested cell type(s) not found in '%s': %s. Available cell types: %s",
                (
                    args.ensemble_members_dir
                    if ensemble_mode
                    else args.targets
                ),
                missing_cell_types,
                sorted(available_cell_types.keys()),
            )
            sys.exit(1)
        cell_type_files = [available_cell_types[ct] for ct in args.cell_types]
        logging.info(
            "Restricting training to %d requested cell type(s): %s",
            len(cell_type_files), args.cell_types,
        )

    use_external_test = (
        args.ensemble_members_test_dir is not None
        if ensemble_mode
        else args.targets_test is not None
    )
    if use_external_test:
        logging.info(
            "Using held-out target directory '%s' for evaluation instead of the "
            "automatic 80:20 split.",
            (
                args.ensemble_members_test_dir
                if ensemble_mode
                else args.targets_test
            ),
        )

    if args.output_dir is not None:
        os.makedirs(args.output_dir, exist_ok=True)

    jobs = max(1, args.jobs)
    logging.info("Fitting genes per cell type with %d parallel workers.", jobs)

    screen_snps = args.screen_snps if args.screen_snps and args.screen_snps > 0 else None
    if screen_snps is not None and ensemble_mode:
        logging.info(
            "Screening each cis-window to the %d SNPs with the strongest "
            "precision-weighted marginal association across ensemble members.",
            screen_snps,
        )
    elif screen_snps is not None:
        logging.info(
            "Screening each cis-window down to its %d SNPs most correlated with the "
            "target (%d-fold inner CV).", screen_snps, args.cv,
        )
    else:
        if ensemble_mode:
            logging.info(
                "SNP screening disabled: fitting every bootstrap elastic net "
                "against every variable SNP in each cis-window."
            )
        else:
            logging.info(
                "SNP screening disabled: fitting against every SNP in each "
                "cis-window (%d-fold inner CV).",
                args.cv,
            )

    probabilistic = args.aleatoric is not None or args.epistemic is not None
    if probabilistic and (args.aleatoric is None or args.epistemic is None):
        logging.error("--aleatoric and --epistemic must both be provided together.")
        sys.exit(1)
    if ensemble_mode:
        logging.info(
            "Frequentist ensemble distillation enabled: %d teacher members x "
            "%d shared cohort bootstraps = %d elastic-net fits per gene after "
            "alpha selection (alpha_mode=%s, sigma_floor=%g). SNP screening "
            "is computed once per gene and reused across all fits.",
            len(member_dirs),
            args.ensemble_bootstraps,
            len(member_dirs) * args.ensemble_bootstraps,
            "fixed" if args.ensemble_alpha is not None else args.ensemble_alpha_mode,
            args.sigma_floor,
        )
    elif probabilistic:
        logging.info(
            "Probabilistic distillation enabled: fitting three independent elastic-net/"
            "ridge regressions per gene (mean, aleatoric variance, epistemic variance) "
            "directly against the teacher's own uncertainty decomposition.",
        )
    if probabilistic:
        if (args.aleatoric_test is None) != (args.epistemic_test is None):
            logging.warning(
                "Only one of --aleatoric-test/--epistemic-test was provided; ignoring "
                "it and falling back to --aleatoric/--epistemic for held-out evaluation."
            )

    batch_size = max(1, args.celltype_batch)
    batches = [
        cell_type_files[start:start + batch_size]
        for start in range(0, len(cell_type_files), batch_size)
    ]
    if batch_size > 1:
        logging.info(
            "Fitting cell types in batches of up to %d, sharing one genotype read "
            "per gene across each batch.", batch_size,
        )

    for batch in tqdm(batches, desc="Processing cell types"):
        prepared = []
        for ct_file in batch:
            if ensemble_mode:
                job = prepare_ensemble_cell_type(
                    ct_file,
                    args=args,
                    bims=bims,
                    idx2ind=idx2ind,
                    bed_template=bed_template,
                    member_dirs=member_dirs,
                    test_member_dirs=test_member_dirs,
                    screen_snps=screen_snps,
                    jobs=jobs,
                )
            else:
                job = prepare_cell_type(
                    ct_file,
                    args=args,
                    bims=bims,
                    idx2ind=idx2ind,
                    bed_template=bed_template,
                    probabilistic=probabilistic,
                    use_external_test=use_external_test,
                    screen_snps=screen_snps,
                    jobs=jobs,
                )
            if job is not None:
                prepared.append(job)

        if not prepared:
            continue

        if len(prepared) > 1 and can_share_genotypes(prepared):
            fit_batch_sharing_genotypes(prepared, jobs, verbose=args.verbose > 0)
        else:
            if len(prepared) > 1:
                logging.info(
                    "Cell types in this batch do not model the same individuals; "
                    "fitting them one at a time instead of sharing genotype reads."
                )
            for _, dataset, test_dataset, model in prepared:
                model.fit_dataset(dataset, test_dataset=test_dataset, verbose=args.verbose > 0)

        for ct_name, _, _, model in prepared:
            if args.output_dir is not None:
                ct_path = ct_name.replace(" ", "_")
                path = os.path.join(args.output_dir, f"{ct_path}.json")
                model.save_coefficients(path)

            if ensemble_mode:
                model_label = "Bootstrapped ensemble elastic net"
            elif probabilistic:
                model_label = "Probabilistic Elastic Net"
            else:
                model_label = args.model_name.capitalize()
            df = model.summarize_models()
            wb_logger.log_celltype_diagnostics(df, cell_type=ct_name, model_label=model_label)

    wb_logger.finish()

    logging.info("Done.")

if __name__ == "__main__":
    main()
