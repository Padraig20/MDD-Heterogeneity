from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from pathlib import Path

from tqdm import tqdm

from src.twas.covariance import (
    DEFAULT_MEMORY_BUDGET_MB,
    build_covariance,
    chunk_snp_budget,
    has_covariance,
    ld_paths,
)
from src.twas.reference import (
    BED_TEMPLATES,
    build_reference,
    find_target_csv,
    read_target_individuals,
)
from src.twas.weights import discover_models, load_snp_sets, read_snp_universe

"""
get_covariance_matrices.py

Precompute the reference LD covariances a model directory needs for TWAS.

`src/twas/run.py` used to build these itself, immediately before running
S-PrediXcan, which meant paying for them on every run and made the TWAS depend
on having the genotypes to hand. They are a property of the model and the
cohort it was distilled on, not of any particular GWAS, so they are built once
here and written beside the weights:

    <models dir>/<cell type>.json
    <models dir>/<cell type>_covariances.txt.gz
    <models dir>/<cell type>_covariances.snps.txt.gz
    <models dir>/<cell type>_covariances.meta.json

A model directory prepared this way is self-contained: `run.py` needs only
`--models-dir`, and never opens a .bed. Run this once per model directory,
including the one passed to `run.py --ctpred-models-dir`.

Which individuals
-----------------
Pass `--targets`, the same target directory `train.py` was given. The LD
reference should be the cohort the weights were fitted on, and the *only*
faithful record of that cohort is the target CSV: `GenotypeDataset` takes its
individuals from that file's columns, so those columns are by definition the
donors that reached the elastic net. `read_target_individuals` reproduces that
derivation, `--max-individuals` mirrors `train.py`'s truncation of it, and any
individual missing from the .fam is dropped exactly as `gene_design` drops it.

Re-sampling with `--num-individuals 1000 --sample-seed 42` does *not* reproduce
that cohort, even though `train.py` was given the same numbers.
`get_feats_from_seqs.py` draws its sample from the VCF header order, whereas
this script can only see the .fam order, and `random.Random(seed).sample`
depends on the order and contents of what it is handed. The two lists differ --
the .fam is the whole genotyped cohort, the VCF header only the sequenced
subset -- so the same seed picks different people. `--num-individuals` is
therefore a fallback for when the targets are unavailable, and it warns.

The cost is dominated by the individual count -- the dosage read and the matrix
product are both linear in it -- so if the training cohort is very large and
this is too slow, `--num-individuals` is the lever. An LD reference is only a
covariance estimate and the usual panels are in the low thousands, but it is a
deliberate departure from the training cohort, not the default.

Example
-------
    python -m src.twas.get_covariance_matrices \\
        --models-dir models/elasticnet \\
        --genotypes /data/ukb --genotype-template UKB \\
        --targets student-target --max-individuals 1000 \\
        --jobs 16
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build the reference LD covariance for every model in a directory "
            "and write it beside the weights."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "-m", "--models-dir",
        type=Path,
        required=True,
        help=(
            "Directory of weights JSONs written by src/distillation/train.py. The "
            "covariances are written here too."
        ),
    )
    parser.add_argument(
        "--cell-types",
        type=str,
        nargs="+",
        metavar="CELL_TYPE",
        default=None,
        help=(
            "Restrict the run to these cell types, named either as in the JSON "
            "filename or with spaces instead of underscores. Defaults to every "
            "model in --models-dir."
        ),
    )

    reference = parser.add_argument_group("reference panel")
    reference.add_argument(
        "-g", "--genotypes",
        type=Path,
        required=True,
        help=(
            "Directory of PLINK genotypes (chr1..chr22 .bed/.bim/.fam), the same "
            "cohort used by src/distillation/train.py."
        ),
    )
    reference.add_argument(
        "-gt", "--genotype-template",
        type=str,
        default="UKB",
        choices=sorted(BED_TEMPLATES),
        help="Naming template for the genotype files.",
    )
    reference.add_argument(
        "-y", "--targets",
        type=Path,
        default=None,
        help=(
            "The target directory (or a single CSV) that src/distillation/train.py "
            "was given. Its columns are the individuals each model was fitted on, "
            "so this makes the LD reference the training cohort exactly. Strongly "
            "recommended: the --num-individuals fallback below cannot reproduce it."
        ),
    )
    reference.add_argument(
        "-mi", "--max-individuals",
        type=int,
        default=None,
        help=(
            "Truncate each target CSV's individuals to the first this many, "
            "mirroring train.py's --max-individuals. Must match what train.py was "
            "given, or the LD reference is a different cohort from the fit."
        ),
    )
    reference.add_argument(
        "-ni", "--num-individuals",
        type=str,
        default="all",
        help=(
            "Fallback used only when --targets is absent: 'all', an integer count "
            "(randomly sampled from the .fam order with --sample-seed), or a 'K/N' "
            "contiguous split. This does NOT reproduce the training cohort, since "
            "that was sampled from the VCF header order instead."
        ),
    )
    reference.add_argument("--sample-seed", type=int, default=42)
    reference.add_argument(
        "--individual-split",
        type=str,
        default=None,
        help="Optional 'K/N' contiguous split applied after --num-individuals.",
    )
    reference.add_argument(
        "--max-snps-in-gene",
        type=int,
        default=None,
        help=(
            "Skip genes whose model has more SNPs than this. A gene's block costs "
            "the square of its SNP count. Defaults to no limit."
        ),
    )

    output = parser.add_argument_group("output")
    output.add_argument(
        "--overwrite",
        action="store_true",
        help=(
            "Rebuild covariances that already exist. Without it, a cell type whose "
            "covariance matches its weights is left alone."
        ),
    )
    output.add_argument(
        "--compression-level",
        type=int,
        default=1,
        choices=range(1, 10),
        metavar="1..9",
        help=(
            "Gzip level for the covariance. Level 1 is much faster and only "
            "modestly larger than Python's level-9 default."
        ),
    )

    runtime = parser.add_argument_group("runtime")
    runtime.add_argument(
        "-j", "--jobs",
        type=int,
        default=min(16, os.cpu_count() or 1),
        help="Worker processes computing gene blocks concurrently.",
    )
    runtime.add_argument(
        "--memory-budget-mb",
        type=int,
        default=DEFAULT_MEMORY_BUDGET_MB,
        help=(
            "Per worker, for the centred dosage chunk it holds. Larger means "
            "fewer, bigger reads; the chunk is n_individuals x SNPs of float64, "
            "so raise it when using few individuals."
        ),
    )
    runtime.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Log and skip a failing cell type instead of aborting.",
    )
    runtime.add_argument("-v", "--verbose", action="store_true")

    return parser.parse_args()


def setup_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )


def resolve_cohort(reference, cell_type: str, args: argparse.Namespace):
    """
    The reference restricted to the individuals this cell type was fitted on.

    With `--targets` that is read off the cell type's own target CSV, per cell
    type, because two cell types need not have been assayed in the same donors.
    Without it the shared `--num-individuals` sample is used unchanged.
    """
    if args.targets is None:
        return reference
    csv_path = find_target_csv(args.targets, cell_type)
    individuals = read_target_individuals(csv_path, args.max_individuals)
    cohort = reference.with_individuals(individuals, source=str(csv_path))
    logging.info(
        "'%s': %d training individual(s) from %s%s.",
        cell_type, cohort.n_individuals, csv_path.name,
        f" (first {args.max_individuals} columns)" if args.max_individuals else "",
    )
    return cohort


def main() -> None:
    args = parse_args()
    setup_logging(args.verbose)

    try:
        model_paths = discover_models(args.models_dir, args.cell_types)
    except (FileNotFoundError, ValueError) as error:
        logging.error("%s", error)
        sys.exit(1)

    pending = [
        path for path in model_paths
        if args.overwrite or not has_covariance(args.models_dir, path.stem)
    ]
    skipped = len(model_paths) - len(pending)
    if skipped:
        logging.info(
            "%d of %d cell type(s) already have a covariance; pass --overwrite to "
            "rebuild them.", skipped, len(model_paths),
        )
    if not pending:
        logging.info("Nothing to do.")
        return

    # Index the panel once for the whole directory. Walking 22 .bim files costs
    # far more than re-parsing the weights JSONs, so the SNP universe is
    # collected in a cheap first pass rather than per cell type.
    wanted_snps: set[str] = set()
    for path in tqdm(pending, desc="Collecting model SNPs", leave=False):
        wanted_snps |= read_snp_universe(path)
    logging.info(
        "%d distinct model SNP(s) across %d cell type(s).",
        len(wanted_snps), len(pending),
    )

    try:
        reference = build_reference(
            genotype_dir=args.genotypes,
            genotype_template=args.genotype_template,
            wanted_snps=wanted_snps,
            # With --targets the cohort is resolved per cell type below, so the
            # base reference just indexes everyone and `with_individuals` cuts
            # it down. The .bim pass is the expensive part and is shared either way.
            num_individuals="all" if args.targets else args.num_individuals,
            sample_seed=args.sample_seed,
            individual_split=None if args.targets else args.individual_split,
        )
    except (FileNotFoundError, ValueError) as error:
        logging.error("%s", error)
        sys.exit(1)

    if args.targets is None:
        logging.warning(
            "No --targets given, so the LD reference is a fresh sample of the .fam "
            "order (--num-individuals %s, seed %d) and is almost certainly a "
            "different set of people from the ones the models were fitted on -- "
            "the training cohort was sampled from the VCF header order, which is a "
            "different list. Pass --targets <the directory train.py used> to make "
            "the two match.", args.num_individuals, args.sample_seed,
        )

    failures: list[str] = []
    started = time.perf_counter()
    for path in tqdm(pending, desc="Cell types"):
        cell_type = path.stem
        try:
            snp_sets = load_snp_sets(path)
            cohort = resolve_cohort(reference, cell_type, args)
            chunk = chunk_snp_budget(cohort.n_individuals, args.memory_budget_mb)
            logging.info(
                "'%s': %d individual(s), %d SNP(s) per read (~%.1f GB peak across "
                "%d worker(s)).",
                cell_type, cohort.n_individuals, chunk,
                args.jobs * chunk * cohort.n_individuals * 8 / 1e9, args.jobs,
            )
            elapsed = time.perf_counter()
            build_covariance(
                cell_type=cell_type,
                snp_sets=snp_sets,
                reference=cohort,
                output_dir=args.models_dir,
                max_snps_in_gene=args.max_snps_in_gene,
                compression_level=args.compression_level,
                overwrite=True,  # the skip decision was already made above
                jobs=args.jobs,
                memory_budget_mb=args.memory_budget_mb,
            )
            cov_path = ld_paths(args.models_dir, cell_type)[0]
            logging.info(
                "'%s' done in %.1fs (%s, %.0f MB).",
                cell_type, time.perf_counter() - elapsed, cov_path.name,
                cov_path.stat().st_size / 1e6,
            )
        except Exception as error:  # noqa: BLE001 - one bad cell type must not kill the sweep
            if not args.continue_on_error:
                raise
            logging.error("Cell type '%s' failed: %s", cell_type, error)
            failures.append(cell_type)

    logging.info(
        "Built %d covariance(s) in %.1fs. %s is ready for "
        "`python -m src.twas.run --models-dir %s`.",
        len(pending) - len(failures), time.perf_counter() - started,
        args.models_dir, args.models_dir,
    )
    if failures:
        logging.warning("%d cell type(s) failed: %s", len(failures), failures)


if __name__ == "__main__":
    main()
