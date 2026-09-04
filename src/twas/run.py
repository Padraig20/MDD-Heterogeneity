from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

from src.twas import plots
from src.twas.aggregate import (
    AGREEMENT_THRESHOLDS,
    aggregate_draws,
    agreement_strata,
    agreement_summary,
    annotate_significance,
    draw_spread,
    summarize,
)
from src.twas.compare import (
    DEFAULT_COVARIANCE_SUFFIX,
    ID_STYLE_VARID,
    LctpredModel,
    comparison_metrics,
    discover_lctpred_models,
    match_model,
    matched_pvalues,
    read_db_metadata,
    run_lctpred,
)
from src.twas.covariance import build_covariance, load_ld_reference, snp_set_hash
from src.twas.ld_blocks import (
    LdBlocks,
    agreement_block_curve,
    block_metrics,
    load_ld_blocks,
    require_matching_build,
)
from src.twas.model_db import load_gene_name_map, write_model_db
from src.twas.reference import BED_TEMPLATES, build_reference
from src.twas.sprediXcan import GwasOptions, read_results, run_sprediXcan
from src.twas.wandb_logger import TwasWandBLogger
from src.twas.weights import (
    KIND_AUTO,
    KIND_MI,
    MODEL_KINDS,
    ModelSpec,
    load_model_json,
    read_snp_universe,
)

"""
run.py

Summary-based TWAS over the distilled cell-type expression models.

For each cell type the pipeline runs the same three steps:

  1. Read the weights JSON written by `src/distillation/train.py` and expand it
     into the draws to run (one for a point-estimate or pooled-mean model, one
     per member-bootstrap fit under --model-kind mi).
  2. Build the reference LD covariance from the UKB genotypes, or reuse a
     previously built one from --ld-dir. One covariance serves every draw.
  3. Run S-PrediXcan per draw, then aggregate, plot and log the results.

Example
-------
    python -m src.twas.run \\
        --models-dir models/elasticnet \\
        --gwas-file gwas/mdd.txt.gz \\
        --snp-column SNP --effect-allele-column A1 --non-effect-allele-column A2 \\
        --beta-column BETA --se-column SE \\
        --genotypes /data/ukb --genotype-template UKB \\
        --ld-dir ld/ukb --output-dir results/mdd --wandb-project mdd-twas
"""

DEFAULT_METAXCAN_DIR = Path("metaxcan/software")
DEFAULT_GENE_NAME_MAP = Path("data/mdd_genes.tsv")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run S-PrediXcan summary TWAS for every distilled cell-type model."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    models = parser.add_argument_group("models")
    models.add_argument(
        "-m", "--models-dir",
        type=Path,
        required=True,
        help=(
            "Directory of weights JSONs written by src/distillation/train.py, one "
            "per cell type (e.g. 'memory_B_cell.json')."
        ),
    )
    models.add_argument(
        "--cell-types",
        type=str,
        nargs="+",
        default=None,
        metavar="CELL_TYPE",
        help=(
            "Restrict the run to these cell types, named either as in the JSON "
            "filename or with spaces instead of underscores (train.py writes "
            "'<cell type>.json' with spaces replaced by underscores). Defaults to "
            "every JSON in --models-dir."
        ),
    )
    models.add_argument(
        "--model-kind",
        choices=MODEL_KINDS,
        default=KIND_AUTO,
        help=(
            "How to turn each JSON into TWAS runs. 'single' uses the point "
            "estimate, 'mean' an ensemble's pooled mean weights, and 'mi' runs one "
            "TWAS per member-bootstrap fit and aggregates them. 'auto' picks "
            "'mean' for ensemble JSONs and 'single' otherwise."
        ),
    )
    models.add_argument(
        "--mi-draws",
        type=int,
        default=None,
        help=(
            "Subsample this many of the member-bootstrap fits under --model-kind "
            "mi. Defaults to using every fit."
        ),
    )
    models.add_argument(
        "--gene-name-map",
        type=Path,
        default=DEFAULT_GENE_NAME_MAP,
        help=(
            "Two-column TSV mapping ensembl gene id to symbol, used to fill the "
            "gene_name output column. Pass an absent path to skip."
        ),
    )

    gwas = parser.add_argument_group("GWAS summary statistics")
    gwas.add_argument("--gwas-file", type=str, default=None, help="Single GWAS file.")
    gwas.add_argument(
        "--gwas-folder",
        type=str,
        default=None,
        help="Folder of GWAS files belonging to one study (alternative to --gwas-file).",
    )
    gwas.add_argument(
        "--gwas-file-pattern",
        type=str,
        default=None,
        help="Regex selecting the GWAS files inside --gwas-folder.",
    )
    gwas.add_argument("--snp-column", type=str, default="SNP")
    gwas.add_argument("--effect-allele-column", type=str, default="A1")
    gwas.add_argument("--non-effect-allele-column", type=str, default="A2")
    gwas.add_argument("--chromosome-column", type=str, default=None)
    gwas.add_argument("--position-column", type=str, default=None)
    gwas.add_argument("--freq-column", type=str, default=None)
    gwas.add_argument("--beta-column", type=str, default=None)
    gwas.add_argument("--beta-sign-column", type=str, default=None)
    gwas.add_argument("--or-column", type=str, default=None)
    gwas.add_argument("--se-column", type=str, default=None)
    gwas.add_argument("--zscore-column", type=str, default=None)
    gwas.add_argument("--pvalue-column", type=str, default=None)
    gwas.add_argument("--separator", type=str, default=None)
    gwas.add_argument("--skip-until-header", type=str, default=None)
    gwas.add_argument(
        "--snp-map-file",
        type=str,
        default=None,
        help="Table converting the GWAS' SNP ids to the reference's, if they differ.",
    )
    gwas.add_argument("--keep-non-rsid", action="store_true")
    gwas.add_argument("--handle-empty-columns", action="store_true")

    reference = parser.add_argument_group("LD reference")
    reference.add_argument(
        "-g", "--genotypes",
        type=Path,
        default=None,
        help=(
            "Directory of PLINK genotypes (chr1..chr22 .bed/.bim/.fam), the same "
            "cohort used by src/distillation/train.py. Omit to reuse a previously "
            "built --ld-dir."
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
        "--ld-dir",
        type=Path,
        required=True,
        help=(
            "Where the per-cell-type covariances live. Written here when "
            "--genotypes is given, read from here otherwise."
        ),
    )
    reference.add_argument(
        "--rebuild-ld",
        action="store_true",
        help="Recompute the covariances even if a matching cached one exists.",
    )
    reference.add_argument(
        "-ni", "--num-individuals",
        type=str,
        default="all",
        help=(
            "Reference individuals to use: 'all', an integer count (randomly "
            "sampled with --sample-seed), or a 'K/N' contiguous split of the .fam "
            "order. Same syntax as get_feats_from_seqs.py."
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
        help="Skip genes whose model has more SNPs than this. Defaults to no limit.",
    )
    reference.add_argument(
        "--ld-compression-level",
        type=int,
        choices=range(1, 10),
        default=1,
        metavar="1..9",
        help=(
            "Gzip compression level for newly built LD files. Level 1 is much "
            "faster and only modestly larger than Python's level-9 default."
        ),
    )

    ld_blocks = parser.add_argument_group("LD blocks")
    ld_blocks.add_argument(
        "--ld-blocks",
        type=Path,
        default=None,
        help=(
            "BED of approximately independent LD blocks (`chr start stop`), used "
            "to count how many distinct blocks the significant genes implicate. "
            "The Berisa-Pickrell definitions live at "
            "bitbucket.org/nygcresearch/ldetect-data (EUR/fourier_ls-all.bed has "
            "the 1,703 blocks scPrediXcan reports against)."
        ),
    )
    ld_blocks.add_argument(
        "--ld-blocks-build",
        type=str,
        default=None,
        choices=["hg19", "hg38", "GRCh37", "GRCh38"],
        help=(
            "Genome build of --ld-blocks. The ldetect files are hg19; lift them "
            "over first if the reference panel is hg38."
        ),
    )
    ld_blocks.add_argument(
        "--genotype-build",
        type=str,
        default=None,
        choices=["hg19", "hg38", "GRCh37", "GRCh38"],
        help=(
            "Genome build of the reference panel's .bim positions, which is what "
            "genes are placed into blocks with. Required with --ld-blocks. UK "
            "Biobank imputation v3 is hg19; OneK1K here is hg38."
        ),
    )
    ld_blocks.add_argument(
        "--ld-blocks-name",
        type=str,
        default=None,
        help="Label for the block set in the outputs, e.g. 'Berisa-Pickrell EUR'.",
    )
    ld_blocks.add_argument(
        "--agreement-criterion",
        type=str,
        default="bonferroni",
        choices=["bonferroni", "fdr"],
        help=(
            "Per-draw significance rule counted for MI model agreement. Each draw "
            "is corrected against its own gene count."
        ),
    )
    ld_blocks.add_argument(
        "--agreement-thresholds",
        type=float,
        nargs="+",
        default=list(AGREEMENT_THRESHOLDS),
        help=(
            "Agreement fractions at which to report gene and LD-block counts. "
            "0 means 'significant in at least one draw'."
        ),
    )

    comparison = parser.add_argument_group("l-ctPred comparison")
    comparison.add_argument(
        "--lctpred-model-dir",
        type=Path,
        default=None,
        help=(
            "Directory of scPrediXcan l-ctPred models, holding a <cell type>.db "
            "beside its <cell type>_covariances.txt.gz. Supplying it runs a second "
            "S-PrediXcan on the same GWAS with those models and reports the two "
            "side by side. Omit to skip the comparison entirely."
        ),
    )
    comparison.add_argument(
        "--lctpred-covariance-suffix",
        type=str,
        default=DEFAULT_COVARIANCE_SUFFIX,
        help="Suffix pairing a covariance file with its .db.",
    )
    comparison.add_argument(
        "--lctpred-snp-column",
        type=str,
        default=None,
        help=(
            "GWAS column holding the variant ids the l-ctPred DB uses, when it "
            "differs from --snp-column. Their models are keyed on "
            "chr1_12345_A_G_b38 style ids rather than rs numbers, so an rs-id "
            "GWAS needs either this or --lctpred-snp-map-file to match anything."
        ),
    )
    comparison.add_argument(
        "--lctpred-snp-map-file",
        type=str,
        default=None,
        help="S-PrediXcan --snp_map_file translating GWAS ids to l-ctPred model ids.",
    )

    output = parser.add_argument_group("output")
    output.add_argument(
        "-o", "--output-dir",
        type=Path,
        required=True,
        help="Directory for the per-cell-type results, figures and statistics.",
    )
    output.add_argument(
        "--wandb-project",
        type=str,
        default=None,
        help="WandB project. Each cell type becomes one run named after it. Omit to disable.",
    )
    output.add_argument("--wandb-entity", type=str, default=None)
    output.add_argument(
        "--fdr",
        type=float,
        default=0.05,
        help="Benjamini-Hochberg level used for the significance flag.",
    )
    output.add_argument(
        "--top-n",
        type=int,
        default=25,
        help="How many of the strongest genes to write to the top-genes table.",
    )
    output.add_argument(
        "--keep-intermediate",
        action="store_true",
        help="Keep the per-draw model DBs and raw S-PrediXcan output.",
    )

    runtime = parser.add_argument_group("runtime")
    runtime.add_argument(
        "--metaxcan-dir",
        type=Path,
        default=DEFAULT_METAXCAN_DIR,
        help="Directory containing SPrediXcan.py.",
    )
    runtime.add_argument(
        "-j", "--jobs",
        type=int,
        default=min(8, os.cpu_count() or 1),
        help="Number of S-PrediXcan runs to execute concurrently.",
    )
    runtime.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Log and skip a failing cell type instead of aborting the whole run.",
    )
    runtime.add_argument("-v", "--verbose", action="count", default=0)
    return parser.parse_args()


def setup_logging(verbosity: int) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbosity >= 1 else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )


def gwas_options(args: argparse.Namespace) -> GwasOptions:
    # MetaXcan splits GWAS lines with a literal `str.split(separator)`, so the
    # flag has to carry the real character. Decode the escape a shell leaves
    # intact, letting `--separator '\t'` mean a tab rather than a backslash.
    separator = args.separator
    if separator:
        separator = separator.encode().decode("unicode_escape")

    options = GwasOptions(
        gwas_file=args.gwas_file,
        gwas_folder=args.gwas_folder,
        gwas_file_pattern=args.gwas_file_pattern,
        snp_column=args.snp_column,
        effect_allele_column=args.effect_allele_column,
        non_effect_allele_column=args.non_effect_allele_column,
        chromosome_column=args.chromosome_column,
        position_column=args.position_column,
        freq_column=args.freq_column,
        beta_column=args.beta_column,
        beta_sign_column=args.beta_sign_column,
        or_column=args.or_column,
        se_column=args.se_column,
        zscore_column=args.zscore_column,
        pvalue_column=args.pvalue_column,
        separator=separator,
        skip_until_header=args.skip_until_header,
        snp_map_file=args.snp_map_file,
        keep_non_rsid=args.keep_non_rsid,
        handle_empty_columns=args.handle_empty_columns,
    )
    options.validate()
    return options


def discover_models(
    models_dir: Path, requested: Optional[list[str]]
) -> list[Path]:
    """
    The weights JSONs to process.

    `train.py` writes `<cell type with spaces replaced by underscores>.json`, so
    a requested cell type is accepted in either spelling.
    """
    available = sorted(Path(models_dir).glob("*.json"))
    if not available:
        raise FileNotFoundError(f"No *.json weights files found in {models_dir}.")
    if requested is None:
        return available

    by_name: dict[str, Path] = {}
    for path in available:
        by_name[path.stem] = path
        by_name[path.stem.replace("_", " ")] = path

    selected, missing = [], []
    for cell_type in requested:
        path = by_name.get(cell_type) or by_name.get(cell_type.replace(" ", "_"))
        if path is None:
            missing.append(cell_type)
        elif path not in selected:
            selected.append(path)
    if missing:
        raise ValueError(
            f"Requested cell type(s) not found in {models_dir}: {missing}. "
            f"Available: {sorted({p.stem for p in available})}"
        )
    return selected


def run_draws(
    spec: ModelSpec,
    ld,
    snp_table,
    gene_names: dict[str, str],
    gwas: GwasOptions,
    metaxcan_dir: Path,
    work_dir: Path,
    jobs: int,
) -> tuple[dict[str, pd.DataFrame], dict]:
    """Build a model DB per draw, run S-PrediXcan on each, and collect the tables."""
    work_dir.mkdir(parents=True, exist_ok=True)

    def one_draw(draw):
        db_path = work_dir / f"{draw.draw_id}.db"
        stats = write_model_db(
            path=db_path,
            draw=draw,
            snp_sets=spec.snp_sets,
            snp_table=snp_table,
            standardized=spec.standardized,
            gene_names=gene_names,
        )
        if stats.n_genes == 0:
            raise RuntimeError(
                f"Draw {draw.draw_id} produced an empty model DB: none of its genes "
                "kept a usable weight after matching the LD reference."
            )
        result = run_sprediXcan(
            metaxcan_dir=metaxcan_dir,
            model_db_path=db_path,
            covariance_path=ld.cov_path,
            output_path=work_dir / f"{draw.draw_id}.csv",
            gwas=gwas,
        )
        return draw.draw_id, read_results(result.output_path), stats

    results: dict[str, pd.DataFrame] = {}
    db_stats: list = []
    with ThreadPoolExecutor(max_workers=max(1, jobs)) as executor:
        iterator = executor.map(one_draw, spec.draws)
        if len(spec.draws) > 1:
            iterator = tqdm(
                iterator, total=len(spec.draws), desc="S-PrediXcan draws", leave=False
            )
        for draw_id, frame, stats in iterator:
            results[draw_id] = frame
            db_stats.append(stats)

    model_stats = {
        "n_draws": len(results),
        "n_model_genes": len(spec.snp_sets),
        "n_model_weights": spec.n_model_snps(),
        "mean_genes_in_db": (
            sum(s.n_genes for s in db_stats) / len(db_stats) if db_stats else 0
        ),
        "mean_weights_in_db": (
            sum(s.n_weights for s in db_stats) / len(db_stats) if db_stats else 0
        ),
        "n_model_snps_absent_from_reference": (
            db_stats[0].n_snps_missing_from_reference if db_stats else 0
        ),
    }
    return results, model_stats


def compare_to_lctpred(
    final: pd.DataFrame,
    model: LctpredModel,
    args: argparse.Namespace,
    gwas: GwasOptions,
    blocks: Optional[LdBlocks],
    cell_dir: Path,
) -> tuple[dict, dict[str, Optional[plt.Figure]], dict[str, pd.DataFrame]]:
    """
    Run S-PrediXcan against one l-ctPred model and put the two results together.

    Their DB is used exactly as shipped, with their covariance, so the only
    thing that differs between the two arms is the prediction model itself.
    """
    logging.info(
        "Comparing '%s' against the l-ctPred model %s.", cell_dir.name, model.db_path.name
    )
    metadata = read_db_metadata(model.db_path)

    if blocks is not None and metadata.build and metadata.build != blocks.build:
        # Their coordinates come from the varIDs, so a build clash here is just
        # as fatal to the block count as it is on our own side.
        raise ValueError(
            f"The l-ctPred model {model.db_path.name} is on {metadata.build} but the "
            f"LD blocks are {blocks.build}; its LD-block count would be meaningless."
        )

    # Their models live in a different identifier space, so the GWAS may have to
    # be read through a different column or mapping for this arm. MetaXcan also
    # discards any variant whose id does not start with 'rs' unless told not to,
    # which would throw away every varID before it ever reached the model.
    their_gwas = replace(
        gwas,
        snp_column=args.lctpred_snp_column or gwas.snp_column,
        snp_map_file=args.lctpred_snp_map_file or gwas.snp_map_file,
        keep_non_rsid=gwas.keep_non_rsid or metadata.id_style == ID_STYLE_VARID,
    )

    comparison_dir = cell_dir / "lctpred"
    comparison_dir.mkdir(parents=True, exist_ok=True)
    theirs = run_lctpred(
        model=model,
        gwas=their_gwas,
        metaxcan_dir=args.metaxcan_dir,
        output_path=comparison_dir / "results.csv",
        fdr=args.fdr,
        metadata=metadata,
    )
    if "gene_name" not in theirs.columns or theirs["gene_name"].isna().all():
        theirs["gene_name"] = (
            theirs["gene"].astype(str).str.split(".").str[0].map(metadata.gene_names)
        )

    matched = matched_pvalues(final, theirs)
    statistics = comparison_metrics(
        final, theirs, matched, blocks=blocks, their_positions=metadata.positions
    )
    statistics["lctpred_model"] = model.name
    statistics["lctpred_build"] = metadata.build
    statistics["lctpred_n_genes_in_model"] = metadata.n_genes

    theirs.to_csv(comparison_dir / "results.csv", index=False)
    matched.to_csv(comparison_dir / "matched_genes.csv", index=False)

    figures = {
        "lctpred_qq": plots.qq_comparison(final, theirs, cell_dir.name),
        "lctpred_scatter": plots.pvalue_scatter(matched, cell_dir.name),
    }
    tables = {"lctpred_top_genes": plots.top_genes(theirs, n=args.top_n)}
    return statistics, figures, tables


def analyse_blocks_and_agreement(
    final: pd.DataFrame,
    ld,
    blocks: Optional[LdBlocks],
    args: argparse.Namespace,
    is_mi: bool,
) -> tuple[pd.DataFrame, dict, Optional[pd.DataFrame], Optional[pd.DataFrame]]:
    """
    Count how many independent LD blocks the hits span, and for an MI run how
    that count survives demanding agreement between the member-bootstrap fits.

    Genes are located by the midpoint of their model's cis window, taken from
    the covariance metadata, which means the coordinates are the reference
    panel's -- hence the build check against the block file.

    Either half works alone: without `--ld-blocks` the agreement strata are
    still produced, just without their block columns.
    """
    statistics: dict = {}
    if blocks is not None:
        assignment = blocks.assign_frame(ld.gene_positions(), final["gene"].tolist())
        final = final.merge(assignment, on="gene", how="left")
        final["block_index"] = final["block_index"].fillna(-1).astype(int)

        statistics = {"ld_blocks": blocks.name, "ld_blocks_build": blocks.build}
        for criterion in ("bonferroni", "fdr"):
            column = f"significant_{criterion}"
            if column in final.columns:
                statistics.update(
                    block_metrics(
                        final,
                        blocks,
                        mask=final[column].fillna(False),
                        prefix=f"{criterion}_",
                    )
                )

    curve = strata = None
    agreement_column = f"agreement_{args.agreement_criterion}"
    if is_mi and agreement_column in final.columns:
        # The strata table stands on its own; it only gains its block columns
        # when a block file was supplied.
        strata = agreement_strata(final, agreement_column=agreement_column)
    if blocks is not None and strata is not None:
        curve = agreement_block_curve(
            final,
            blocks,
            agreement_column=agreement_column,
            thresholds=args.agreement_thresholds,
        )
        at_80 = curve.loc[np.isclose(curve["threshold"], 0.8)]
        if not at_80.empty:
            statistics["n_ld_blocks_at_80pct_agreement"] = int(at_80["n_ld_blocks"].iloc[0])
            statistics["n_genes_at_80pct_agreement"] = int(at_80["n_genes"].iloc[0])
            statistics["frac_ld_blocks_retained_at_80pct"] = float(
                at_80["frac_ld_blocks_retained"].iloc[0]
            )
            statistics["frac_genes_retained_at_80pct"] = float(
                at_80["frac_genes_retained"].iloc[0]
            )
    return final, statistics, curve, strata


def build_figures(
    final: pd.DataFrame,
    long: Optional[pd.DataFrame],
    ld,
    cell_type: str,
    fdr: float,
    curve: Optional[pd.DataFrame] = None,
    agreement_column: str = "agreement_bonferroni",
) -> dict[str, Optional[plt.Figure]]:
    figures = {
        "manhattan": plots.manhattan(final, ld.gene_positions(), cell_type, fdr=fdr),
        "qq": plots.qq(final, cell_type),
        "volcano": plots.volcano(final, cell_type, fdr=fdr),
        "zscore_histogram": plots.zscore_histogram(final, cell_type),
    }
    if long is not None:
        figures["mi_stability"] = plots.mi_stability(final, cell_type)
        figures["mi_draw_spread"] = plots.mi_draw_spread(long, cell_type)
        figures["mi_draw_summary"] = plots.mi_draw_summary(
            draw_spread(long), cell_type
        )
        figures["mi_agreement"] = plots.agreement_histogram(
            final, cell_type, agreement_column=agreement_column
        )
        figures["mi_agreement_vs_strength"] = plots.agreement_vs_strength(
            final, cell_type, agreement_column=agreement_column
        )
    if "block_index" in final.columns:
        figures["ld_block_gene_counts"] = plots.ld_block_gene_counts(final, cell_type)
    if curve is not None:
        figures["mi_agreement_ld_blocks"] = plots.agreement_ld_block_curve(
            curve, cell_type
        )
    return figures


def process_cell_type(
    model_path: Path,
    args: argparse.Namespace,
    gwas: GwasOptions,
    gene_names: dict[str, str],
    logger: TwasWandBLogger,
    reference=None,
    blocks: Optional[LdBlocks] = None,
    lctpred: Optional[LctpredModel] = None,
) -> dict:
    """Run the full three-step pipeline for one cell type."""
    spec = load_model_json(
        model_path, kind=args.model_kind, mi_draws=args.mi_draws, seed=args.sample_seed
    )
    cell_type = spec.cell_type
    logging.info(
        "Cell type '%s': %s model, kind=%s, %d gene(s), %d draw(s).",
        cell_type, spec.source, spec.kind, len(spec.snp_sets), len(spec.draws),
    )

    # Step 2: the LD reference, shared by every draw of this cell type.
    if reference is not None:
        ld = build_covariance(
            cell_type=cell_type,
            snp_sets=spec.snp_sets,
            reference=reference,
            ld_dir=args.ld_dir,
            max_snps_in_gene=args.max_snps_in_gene,
            compression_level=args.ld_compression_level,
            overwrite=args.rebuild_ld,
        )
    else:
        ld = load_ld_reference(
            args.ld_dir, cell_type, expected_hash=snp_set_hash(spec.snp_sets)
        )
        logging.info("Using the pre-built LD reference at %s.", ld.cov_path)
    snp_table = ld.load_snp_table()

    # Step 1 + 3: one model DB and one S-PrediXcan run per draw.
    cell_dir = Path(args.output_dir) / cell_type
    cell_dir.mkdir(parents=True, exist_ok=True)
    # The scratch dir lives under the output dir rather than /tmp: an MI sweep
    # writes one model DB per draw, which for a full transcriptome adds up to
    # more than a small /tmp can take.
    work_dir = (
        cell_dir / "draws"
        if args.keep_intermediate
        else Path(tempfile.mkdtemp(prefix="draws_", dir=cell_dir))
    )
    try:
        per_draw, model_stats = run_draws(
            spec=spec,
            ld=ld,
            snp_table=snp_table,
            gene_names=gene_names,
            gwas=gwas,
            metaxcan_dir=args.metaxcan_dir,
            work_dir=work_dir,
            jobs=args.jobs,
        )
    finally:
        if not args.keep_intermediate and work_dir.exists():
            shutil.rmtree(work_dir, ignore_errors=True)

    long = None
    if spec.kind == KIND_MI:
        final, long = aggregate_draws(per_draw, fdr=args.fdr)
    else:
        final = annotate_significance(next(iter(per_draw.values())), fdr=args.fdr)

    final, block_stats, curve, strata = analyse_blocks_and_agreement(
        final, ld, blocks, args, is_mi=spec.kind == KIND_MI
    )

    summary = summarize(
        final,
        fdr=args.fdr,
        extra={
            **model_stats,
            **block_stats,
            "model_source": spec.source,
            "model_kind": spec.kind,
            "standardized_weights_rescaled": spec.standardized,
            "n_reference_individuals": ld.meta["reference"]["n_individuals"],
            "n_genes_in_ld_reference": ld.meta["n_genes_written"],
        },
    )
    if spec.kind == KIND_MI:
        summary.update(
            agreement_summary(final, agreement_column=f"agreement_{args.agreement_criterion}")
        )

    comparison_figures: dict[str, Optional[plt.Figure]] = {}
    comparison_tables: dict[str, pd.DataFrame] = {}
    if lctpred is not None:
        try:
            stats, comparison_figures, comparison_tables = compare_to_lctpred(
                final, lctpred, args, gwas, blocks, cell_dir
            )
            summary.update(stats)
        except Exception as error:  # noqa: BLE001 - the comparison is an extra
            if not args.continue_on_error:
                raise
            logging.error(
                "The l-ctPred comparison for '%s' failed: %s", cell_type, error
            )

    # Persist everything before touching WandB, so a logging failure cannot lose
    # a run that took hours of S-PrediXcan.
    final.to_csv(cell_dir / "results.csv", index=False)
    top = plots.top_genes(final, n=args.top_n)
    top.to_csv(cell_dir / "top_genes.csv", index=False)
    if long is not None:
        long.to_csv(cell_dir / "per_draw_zscores.csv", index=False)
        draw_spread(long).to_csv(cell_dir / "draw_spread.csv", index=False)
    if curve is not None:
        curve.to_csv(cell_dir / "agreement_ld_block_curve.csv", index=False)
    if strata is not None and not strata.empty:
        strata.to_csv(cell_dir / "agreement_strata.csv", index=False)
    with (cell_dir / "summary.json").open("w") as handle:
        json.dump(summary, handle, indent=2)

    figures = build_figures(
        final, long, ld, cell_type, args.fdr,
        curve=curve,
        agreement_column=f"agreement_{args.agreement_criterion}",
    )
    figures.update(comparison_figures)
    figure_dir = cell_dir / "figures"
    figure_dir.mkdir(exist_ok=True)
    for name, figure in figures.items():
        if figure is not None:
            figure.savefig(figure_dir / f"{name}.png", dpi=150)

    tables = {"top_genes": top}
    if long is not None:
        tables["draw_spread"] = draw_spread(long)
    if curve is not None:
        tables["agreement_ld_block_curve"] = curve
    if strata is not None and not strata.empty:
        tables["agreement_strata"] = strata
    tables.update(comparison_tables)

    logger.start(cell_type, config=_wandb_config(args, spec))
    try:
        logger.log_results(final, summary, figures, tables=tables)
    finally:
        logger.finish()
        for figure in figures.values():
            if figure is not None:
                plt.close(figure)

    logging.info(
        "Cell type '%s': %d gene(s) tested, %d significant at BH FDR %g, "
        "lambda_GC = %.3f.",
        cell_type, summary["n_genes_tested"], summary["n_significant_fdr"],
        args.fdr, summary["lambda_gc"],
    )
    if "bonferroni_n_ld_blocks" in summary:
        logging.info(
            "Cell type '%s': %d Bonferroni-significant gene(s) from %d different "
            "LD block(s) among %d pre-defined blocks (%.2f genes per block).",
            cell_type, summary["n_significant_bonferroni"],
            summary["bonferroni_n_ld_blocks"], summary["bonferroni_n_ld_blocks_total"],
            summary["bonferroni_genes_per_ld_block"],
        )
    return {"cell_type": cell_type, **summary}


def _wandb_config(args: argparse.Namespace, spec: ModelSpec) -> dict:
    return {
        "model_path": str(spec.path),
        "model_source": spec.source,
        "model_kind": spec.kind,
        "n_draws": len(spec.draws),
        "n_model_genes": len(spec.snp_sets),
        "gwas_file": args.gwas_file or args.gwas_folder,
        "genotype_template": args.genotype_template,
        "num_individuals": args.num_individuals,
        "individual_split": args.individual_split,
        "fdr": args.fdr,
        "ld_blocks": str(args.ld_blocks) if args.ld_blocks else None,
        "ld_blocks_build": args.ld_blocks_build,
        "genotype_build": args.genotype_build,
        "agreement_criterion": args.agreement_criterion,
    }


def main() -> None:
    args = parse_args()
    setup_logging(args.verbose)

    try:
        gwas = gwas_options(args)
    except ValueError as error:
        logging.error("%s", error)
        sys.exit(1)

    if args.genotypes is None and not Path(args.ld_dir).exists():
        logging.error(
            "Neither --genotypes nor an existing --ld-dir (%s) was provided; there "
            "is nothing to compute the LD covariance from.", args.ld_dir,
        )
        sys.exit(1)

    try:
        model_paths = discover_models(args.models_dir, args.cell_types)
    except (FileNotFoundError, ValueError) as error:
        logging.error("%s", error)
        sys.exit(1)

    lctpred_models: dict = {}
    if args.lctpred_model_dir is not None:
        try:
            lctpred_models = discover_lctpred_models(
                args.lctpred_model_dir,
                covariance_suffix=args.lctpred_covariance_suffix,
            )
        except (FileNotFoundError, NotADirectoryError) as error:
            logging.error("%s", error)
            sys.exit(1)

    blocks = None
    if args.ld_blocks is not None:
        try:
            blocks = load_ld_blocks(
                args.ld_blocks, build=args.ld_blocks_build, name=args.ld_blocks_name
            )
            require_matching_build(blocks, args.genotype_build)
        except (FileNotFoundError, ValueError) as error:
            logging.error("%s", error)
            sys.exit(1)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    gene_names = load_gene_name_map(args.gene_name_map)
    logger = TwasWandBLogger(project=args.wandb_project, entity=args.wandb_entity)

    # Index the reference once for the whole sweep. Walking 22 UKB .bim files
    # costs far more than re-parsing the weights JSONs, so the SNP universe is
    # collected in a cheap first pass rather than per cell type.
    reference = None
    if args.genotypes is not None:
        wanted_snps: set[str] = set()
        for model_path in tqdm(model_paths, desc="Collecting model SNPs", leave=False):
            wanted_snps |= read_snp_universe(model_path)
        logging.info(
            "%d distinct model SNP(s) across %d cell type(s).",
            len(wanted_snps), len(model_paths),
        )
        reference = build_reference(
            genotype_dir=args.genotypes,
            genotype_template=args.genotype_template,
            wanted_snps=wanted_snps,
            num_individuals=args.num_individuals,
            sample_seed=args.sample_seed,
            individual_split=args.individual_split,
        )

    logging.info("Running TWAS for %d cell type(s).", len(model_paths))
    summaries: list[dict] = []
    failures: list[str] = []
    unmatched: list[str] = []
    for model_path in tqdm(model_paths, desc="Cell types"):
        lctpred = match_model(lctpred_models, model_path.stem) if lctpred_models else None
        if lctpred_models and lctpred is None:
            unmatched.append(model_path.stem)
        try:
            summaries.append(
                process_cell_type(
                    model_path, args, gwas, gene_names, logger,
                    reference=reference, blocks=blocks, lctpred=lctpred,
                )
            )
        except Exception as error:  # noqa: BLE001 - one bad cell type must not kill the sweep
            if not args.continue_on_error:
                raise
            logging.error("Cell type '%s' failed: %s", model_path.stem, error)
            failures.append(model_path.stem)

    if summaries:
        overview = pd.DataFrame(summaries).sort_values("n_significant_fdr", ascending=False)
        overview.to_csv(output_dir / "summary.csv", index=False)
        logging.info("Wrote the cross-cell-type overview to %s.", output_dir / "summary.csv")
    if unmatched:
        logging.warning(
            "%d cell type(s) had no l-ctPred counterpart in %s and were run "
            "without a comparison: %s",
            len(unmatched), args.lctpred_model_dir, unmatched,
        )
    if failures:
        logging.warning("%d cell type(s) failed: %s", len(failures), failures)
    logging.info("Done.")


if __name__ == "__main__":
    main()
