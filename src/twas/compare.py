from __future__ import annotations

import logging
import re
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from src.twas.aggregate import annotate_significance
from src.twas.ld_blocks import LdBlocks, block_metrics
from src.twas.sprediXcan import GwasOptions, read_results, run_sprediXcan

"""
compare.py

Head-to-head against the l-ctPred models scPrediXcan ships.

scPrediXcan distributes a finished S-PrediXcan prediction model DB and its
matching covariance per cell type, so the comparison is simply a second
S-PrediXcan run on the same GWAS with their DB in place of ours. Everything
downstream -- multiple-testing correction, LD-block counting -- is then the
same machinery both sides go through, which is the only way the numbers mean
anything next to each other.

SNP identifier spaces
---------------------
Their `weights.rsid` column does not hold rs numbers. It repeats `varID`, which
looks like

    chr21_17276203_G_T_b38

while our models are keyed on the reference panel's rs ids. S-PrediXcan matches
the GWAS to a model purely on that string, so pointing it at their DB with an
rs-id GWAS silently matches nothing and returns a table of NAs rather than an
error. `check_snp_overlap` therefore inspects the result and fails loudly.

To bridge the two, either point `--lctpred-snp-column` at a GWAS column already
in `chr_pos_ref_alt_b38` form, or supply `--lctpred-snp-map-file` (S-PrediXcan's
own `--snp_map_file`, a table translating GWAS ids to model ids).

That identifier format also carries the genome build, so `DbMetadata` reads the
build straight off the varIDs and the caller can check it against the LD blocks
instead of trusting a flag.
"""

DEFAULT_COVARIANCE_SUFFIX = "_covariances.txt.gz"

# chr21_17276203_G_T_b38
VARID_PATTERN = re.compile(
    r"^chr([0-9XYM]+|[0-9]+)_(\d+)_([ACGTN]+)_([ACGTN]+)_b(\d+)$", re.IGNORECASE
)
RSID_PATTERN = re.compile(r"^rs\d+$", re.IGNORECASE)

ID_STYLE_VARID = "varid"
ID_STYLE_RSID = "rsid"
ID_STYLE_UNKNOWN = "unknown"


def normalize_cell_type(name: str) -> str:
    """
    Fold a cell-type name to a comparable key.

    Their file names and ours are the same labels through different pipelines,
    so 'CD14-low_CD16-positive_monocyte' and 'CD14 low CD16 positive monocyte'
    have to land on the same key.
    """
    return re.sub(r"[^a-z0-9]+", "", str(name).lower())


@dataclass
class LctpredModel:
    """One cell type's prediction model DB and its covariance."""

    name: str
    db_path: Path
    covariance_path: Path


@dataclass
class DbMetadata:
    """What a prediction model DB can tell us without running anything."""

    n_genes: int
    id_style: str
    build: Optional[str]
    genes: set[str] = field(default_factory=set)
    gene_names: dict[str, str] = field(default_factory=dict)
    positions: dict[str, tuple[str, int]] = field(default_factory=dict)


def discover_lctpred_models(
    directory: Path, covariance_suffix: str = DEFAULT_COVARIANCE_SUFFIX
) -> dict[str, LctpredModel]:
    """
    Pair every `<name>.db` in a directory with its `<name><suffix>` covariance.

    Returns a dict keyed on the normalized cell-type name. A DB without a
    covariance beside it is skipped with a warning rather than failing the run;
    the comparison is an extra, not the point of the pipeline.
    """
    directory = Path(directory)
    if not directory.is_dir():
        raise NotADirectoryError(f"--lctpred-model-dir is not a directory: {directory}")

    models: dict[str, LctpredModel] = {}
    orphans: list[str] = []
    for db_path in sorted(directory.glob("*.db")):
        name = db_path.stem
        covariance = directory / f"{name}{covariance_suffix}"
        if not covariance.exists():
            # Tolerate the other common spelling before giving up.
            alternatives = sorted(directory.glob(f"{name}*cov*.txt.gz"))
            if not alternatives:
                orphans.append(name)
                continue
            covariance = alternatives[0]
        models[normalize_cell_type(name)] = LctpredModel(
            name=name, db_path=db_path, covariance_path=covariance
        )

    if orphans:
        logging.warning(
            "%d l-ctPred model(s) in %s have no '%s' covariance beside them and "
            "will be skipped: %s",
            len(orphans), directory, covariance_suffix, ", ".join(orphans[:5]),
        )
    if not models:
        raise FileNotFoundError(
            f"No usable <name>.db + <name>{covariance_suffix} pair found in {directory}."
        )
    logging.info("Found %d l-ctPred model(s) in %s.", len(models), directory)
    return models


def match_model(
    models: dict[str, LctpredModel], cell_type: str
) -> Optional[LctpredModel]:
    """The l-ctPred model for one of our cell types, matched on the folded name."""
    return models.get(normalize_cell_type(cell_type))


def read_db_metadata(db_path: Path, chunk_size: int = 500_000) -> DbMetadata:
    """
    Gene list, symbols, cis-window midpoints and genome build from a model DB.

    Positions come from parsing the varIDs, which makes the gene coordinates
    directly comparable to the ones we take from our own covariance metadata:
    both are the midpoint of the window the model actually puts weight in.

    The weights table runs to millions of rows, so it is streamed and reduced to
    a per-gene min/max as it goes rather than materialised.
    """
    db_path = Path(db_path)
    with sqlite3.connect(f"file:{db_path}?mode=ro", uri=True) as connection:
        gene_names: dict[str, str] = {}
        try:
            extra = pd.read_sql("SELECT gene, genename FROM extra", connection)
            gene_names = {
                str(gene).split(".")[0]: str(name)
                for gene, name in zip(extra["gene"], extra["genename"])
                if name is not None
            }
        except Exception as error:  # noqa: BLE001 - `extra` is optional here
            logging.debug("Could not read `extra` from %s: %s", db_path.name, error)

        columns = {
            row[1] for row in connection.execute("PRAGMA table_info(weights)")
        }
        id_column = "varID" if "varID" in columns else "rsid"

        partials: list[pd.DataFrame] = []
        id_style = ID_STYLE_UNKNOWN
        build: Optional[str] = None
        unparsed = 0

        query = f"SELECT gene, {id_column} AS snp FROM weights"
        for chunk in pd.read_sql(query, connection, chunksize=chunk_size):
            parsed = chunk["snp"].astype(str).str.extract(VARID_PATTERN)
            usable = parsed[1].notna()
            if id_style == ID_STYLE_UNKNOWN:
                if usable.any():
                    id_style = ID_STYLE_VARID
                elif chunk["snp"].astype(str).head(50).str.match(RSID_PATTERN).any():
                    id_style = ID_STYLE_RSID
            if not usable.any():
                unparsed += len(chunk)
                continue
            if build is None:
                build = f"hg{parsed.loc[usable, 4].iloc[0]}"

            block = pd.DataFrame({
                "gene": chunk.loc[usable, "gene"].astype(str).str.split(".").str[0],
                "chrom": parsed.loc[usable, 0].astype(str),
                "bp": parsed.loc[usable, 1].astype(np.int64),
            })
            partials.append(
                block.groupby("gene", sort=False).agg(
                    chrom=("chrom", "first"), lo=("bp", "min"), hi=("bp", "max")
                )
            )

        genes = {
            str(row[0]).split(".")[0]
            for row in connection.execute("SELECT DISTINCT gene FROM weights")
        }
        n_genes = len(genes)

    positions: dict[str, tuple[str, int]] = {}
    if partials:
        # Chunk boundaries can split a gene, so reduce the per-chunk extremes.
        combined = pd.concat(partials).groupby(level=0).agg(
            chrom=("chrom", "first"), lo=("lo", "min"), hi=("hi", "max")
        )
        midpoints = ((combined["lo"] + combined["hi"]) // 2).astype(np.int64)
        positions = {
            gene: (chrom, int(bp))
            for gene, chrom, bp in zip(combined.index, combined["chrom"], midpoints)
        }
    if unparsed:
        logging.debug(
            "%s: %d weight row(s) had an unparseable variant id; those genes have "
            "no position and are left out of the LD-block count.",
            db_path.name, unparsed,
        )
    logging.info(
        "%s: %d gene(s), %s-style variant ids, build %s, %d gene(s) positioned.",
        db_path.name, n_genes, id_style, build or "unknown", len(positions),
    )
    return DbMetadata(
        n_genes=n_genes,
        id_style=id_style,
        build=build,
        genes=genes,
        gene_names=gene_names,
        positions=positions,
    )


def check_snp_overlap(results: pd.DataFrame, model: LctpredModel, metadata: DbMetadata) -> None:
    """
    Fail loudly when the GWAS and the model turn out to speak different id
    dialects.

    S-PrediXcan does not treat "nothing matched" as an error -- it writes a
    table of NAs -- so without this check a comparison would quietly report that
    l-ctPred finds no genes at all, which looks like a result rather than a
    misconfiguration.
    """
    if results.empty or "n_snps_used" not in results.columns:
        used = 0
    else:
        used = int(pd.to_numeric(results["n_snps_used"], errors="coerce").fillna(0).sum())
    if used > 0:
        return

    hint = ""
    if metadata.id_style == ID_STYLE_VARID:
        hint = (
            f" {model.db_path.name} is keyed on '{ID_STYLE_VARID}' identifiers such as "
            f"chr1_12345_A_G_b38, not rs numbers. Point --lctpred-snp-column at a GWAS "
            "column in that format, or pass --lctpred-snp-map-file to translate."
        )
    raise RuntimeError(
        f"The l-ctPred run for '{model.name}' matched zero GWAS variants, so its "
        f"results are all NA.{hint}"
    )


def run_lctpred(
    model: LctpredModel,
    gwas: GwasOptions,
    metaxcan_dir: Path,
    output_path: Path,
    fdr: float = 0.05,
    metadata: Optional[DbMetadata] = None,
) -> pd.DataFrame:
    """Run S-PrediXcan against one l-ctPred model and correct its p-values."""
    run_sprediXcan(
        metaxcan_dir=metaxcan_dir,
        model_db_path=model.db_path,
        covariance_path=model.covariance_path,
        output_path=output_path,
        gwas=gwas,
    )
    results = read_results(output_path)
    if metadata is not None:
        check_snp_overlap(results, model, metadata)
    return annotate_significance(results, fdr=fdr)


def two_sample_quantiles(
    left: np.ndarray, right: np.ndarray, n_points: Optional[int] = None
) -> tuple[np.ndarray, np.ndarray]:
    """
    Matched quantiles of two samples, for a quantile-quantile comparison.

    This deliberately does not pair genes. A Q-Q plot asks whether one method's
    *distribution* of evidence is shifted relative to the other's, which is the
    question "does it have more power" -- a gene-level pairing answers the
    different question of whether the two agree case by case, and
    `matched_pvalues` covers that.
    """
    left = np.asarray(left, dtype=float)
    right = np.asarray(right, dtype=float)
    left = left[np.isfinite(left)]
    right = right[np.isfinite(right)]
    if left.size == 0 or right.size == 0:
        return np.array([]), np.array([])

    n = n_points or min(left.size, right.size)
    probabilities = (np.arange(1, n + 1) - 0.5) / n
    return np.quantile(left, probabilities), np.quantile(right, probabilities)


def matched_pvalues(
    ours: pd.DataFrame, theirs: pd.DataFrame, suffixes: tuple[str, str] = ("_ours", "_lctpred")
) -> pd.DataFrame:
    """
    Inner join of the two result tables on the versionless Ensembl id.

    Both sides carry their own significance flags, so the join is enough to read
    off which genes each method finds and which they share.
    """
    def prepare(frame: pd.DataFrame) -> pd.DataFrame:
        columns = [
            c for c in
            ("gene", "gene_name", "zscore", "pvalue", "qvalue", "effect_size",
             "n_snps_used", "significant_fdr", "significant_bonferroni")
            if c in frame.columns
        ]
        subset = frame[columns].copy()
        subset["gene_key"] = subset["gene"].astype(str).str.split(".").str[0]
        return subset.drop_duplicates("gene_key")

    merged = prepare(ours).merge(
        prepare(theirs), on="gene_key", how="inner", suffixes=suffixes
    )
    return merged


def gene_keys(values) -> set[str]:
    """Versionless, case-folded Ensembl ids, for comparing gene sets."""
    return {
        str(value).split(".")[0].strip().upper()
        for value in values
        if value is not None and str(value) != "nan"
    }


def gene_overlap_report(
    our_model_genes,
    their_model_genes,
    our_results: pd.DataFrame,
    their_results: pd.DataFrame,
) -> dict:
    """
    Where genes are lost between the two model definitions and the two results.

    A small shared-gene count has two very different explanations and this
    separates them. If the two *models* already barely overlap, the identifier
    spaces or the trained transcriptomes differ. If the models overlap well but
    the *results* do not, genes are being dropped downstream instead -- by the
    LD reference, by `--max-snps-in-gene`, or by S-PrediXcan finding no usable
    variant -- and `*_lost_from_model` says which side is doing the dropping.
    """
    our_model, their_model = gene_keys(our_model_genes), gene_keys(their_model_genes)
    our_tested = gene_keys(our_results["gene"]) if len(our_results) else set()
    their_tested = gene_keys(their_results["gene"]) if len(their_results) else set()

    report = {
        "n_genes_in_our_model": len(our_model),
        "n_genes_in_lctpred_model": len(their_model),
        "n_genes_shared_by_models": len(our_model & their_model),
        "n_genes_in_our_results": len(our_tested),
        "n_genes_in_lctpred_results": len(their_tested),
        "n_genes_shared_by_results": len(our_tested & their_tested),
        "n_genes_ours_lost_from_model": len(our_model - our_tested),
        "n_genes_lctpred_lost_from_model": len(their_model - their_tested),
    }
    shared_models = report["n_genes_shared_by_models"]
    if shared_models:
        report["frac_shared_models_reaching_both_results"] = (
            report["n_genes_shared_by_results"] / shared_models
        )
    return report


def log_overlap_report(
    report: dict, id_style: Optional[str] = None, frac_snps_used: Optional[float] = None
) -> None:
    """Explain the attrition at INFO, since a low overlap is easy to misread."""
    logging.info(
        "Gene overlap: %d in our model, %d in the l-ctPred model, %d shared. "
        "After the association: %d ours, %d theirs, %d shared.",
        report["n_genes_in_our_model"], report["n_genes_in_lctpred_model"],
        report["n_genes_shared_by_models"], report["n_genes_in_our_results"],
        report["n_genes_in_lctpred_results"], report["n_genes_shared_by_results"],
    )
    if report["n_genes_shared_by_models"] == 0:
        logging.warning(
            "The two models share no gene at all. The identifier spaces differ "
            "-- check whether one side uses gene symbols or versioned Ensembl ids."
        )
        return

    fraction = report.get("frac_shared_models_reaching_both_results", 1.0)
    if fraction >= 0.5:
        return
    logging.warning(
        "Only %.0f%% of the %d gene(s) both models define survive into both "
        "result tables (%d ours, %d theirs dropped between model and result). "
        "The gene identifiers line up, so the loss is downstream.",
        100 * fraction, report["n_genes_shared_by_models"],
        report["n_genes_ours_lost_from_model"],
        report["n_genes_lctpred_lost_from_model"],
    )
    if frac_snps_used is not None and frac_snps_used < 0.5:
        logging.warning(
            "The l-ctPred arm used only %.0f%% of the variants its models "
            "contain, so most of its genes were dropped for lack of a usable "
            "variant rather than by any gene filter.%s",
            100 * frac_snps_used,
            (
                " Its models are keyed on chr_pos_ref_alt_b38 ids: check that "
                "--lctpred-snp-column names a GWAS column in exactly that "
                "format, including the same reference/alternate order and the "
                "'_b38' suffix."
                if id_style == ID_STYLE_VARID else ""
            ),
        )
    else:
        logging.warning(
            "Genes with no usable variant in the LD reference or the GWAS, or "
            "genes skipped by --max-snps-in-gene, are the usual cause."
        )


def comparison_metrics(
    ours: pd.DataFrame,
    theirs: pd.DataFrame,
    matched: pd.DataFrame,
    suffixes: tuple[str, str] = ("_ours", "_lctpred"),
) -> dict:
    """
    How the two arms' hit lists relate on the genes they both test.

    Each arm's own significance counts and LD-block coverage are produced by the
    per-arm analysis; what is left here is strictly the overlap. Note that each
    side is corrected against its own gene count, which is right -- the two
    models test different transcriptomes -- but does mean the two Bonferroni
    thresholds differ.
    """
    ours_suffix, theirs_suffix = suffixes
    statistics: dict = {"n_genes_shared": int(len(matched))}

    for criterion in ("fdr", "bonferroni"):
        ours_column = f"significant_{criterion}{ours_suffix}"
        theirs_column = f"significant_{criterion}{theirs_suffix}"
        if ours_column not in matched.columns or theirs_column not in matched.columns:
            continue
        mine = matched[ours_column].fillna(False)
        yours = matched[theirs_column].fillna(False)
        statistics[f"{criterion}_both"] = int((mine & yours).sum())
        statistics[f"{criterion}_ours_only"] = int((mine & ~yours).sum())
        statistics[f"{criterion}_lctpred_only"] = int((~mine & yours).sum())
        union = int((mine | yours).sum())
        statistics[f"{criterion}_jaccard"] = (
            statistics[f"{criterion}_both"] / union if union else float("nan")
        )

    for suffix, side in ((ours_suffix, "ours"), (theirs_suffix, "lctpred")):
        column = f"zscore{suffix}"
        if column in matched.columns:
            statistics[f"mean_abs_zscore_{side}"] = float(
                matched[column].abs().mean()
            )
    left, right = f"zscore{ours_suffix}", f"zscore{theirs_suffix}"
    if left in matched.columns and right in matched.columns:
        pair = matched[[left, right]].dropna()
        if len(pair) > 2:
            statistics["zscore_correlation"] = float(pair[left].corr(pair[right]))
    return statistics


__all__ = [
    "DEFAULT_COVARIANCE_SUFFIX",
    "ID_STYLE_RSID",
    "ID_STYLE_UNKNOWN",
    "ID_STYLE_VARID",
    "DbMetadata",
    "LctpredModel",
    "check_snp_overlap",
    "comparison_metrics",
    "discover_lctpred_models",
    "gene_keys",
    "gene_overlap_report",
    "log_overlap_report",
    "match_model",
    "matched_pvalues",
    "normalize_cell_type",
    "read_db_metadata",
    "run_lctpred",
    "two_sample_quantiles",
]
