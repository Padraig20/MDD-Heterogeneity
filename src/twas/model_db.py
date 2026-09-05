from __future__ import annotations

import logging
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from src.twas.covariance import GeneSnpInfo
from src.twas.weights import Draw, GeneSnps

"""
model_db.py

Turn one draw's weights into the SQLite prediction model S-PrediXcan reads.

`metax.PredictionModel.ModelDB` issues exactly two queries, and the schema here
is the minimum that satisfies both:

    SELECT rsid, gene, weight, ref_allele, eff_allele FROM weights
    SELECT gene, genename, `n.snps.in.model`, `pred.perf.R2`,
           `pred.perf.pval`, `pred.perf.qval` FROM extra ORDER BY gene

Note the naming: `WDBQF` maps the `ref_allele` column to *non-effect* allele
and `eff_allele` to the effect allele. The effect allele is whichever allele the
dosage counts, which for bed_reader is .bim column 5.

Weight scale
------------
`AssociationCalculation.association` computes

    z = sum_l  w_l * z_l * sigma_l  /  sqrt(w' COV w)

with `sigma_l = sqrt(diag(COV))` taken from the reference. Rescaling every `w`
by one constant cancels, so the target standardization `LR` applies is
harmless. A *per-SNP* rescale does not cancel, and `LR`/`ProbabilisticLR`
persist `enet.coef_` fitted on a standardized design, i.e. `w_std = w_raw *
sigma_l`. Those models therefore need `w_raw = w_std / sigma_l` before they can
be written here, using the reference panel's `sigma_l`.
"""

SCHEMA = """
CREATE TABLE weights (
    rsid TEXT,
    gene TEXT,
    weight REAL,
    ref_allele TEXT,
    eff_allele TEXT
);
CREATE TABLE extra (
    gene TEXT,
    genename TEXT,
    "n.snps.in.model" INTEGER,
    "pred.perf.R2" REAL,
    "pred.perf.pval" REAL,
    "pred.perf.qval" REAL
);
"""

INDICES = """
CREATE INDEX weights_gene ON weights (gene);
CREATE INDEX weights_rsid ON weights (rsid);
CREATE INDEX extra_gene ON extra (gene);
"""


@dataclass
class ModelDbStats:
    n_genes: int
    n_weights: int
    n_snps_missing_from_reference: int
    n_zero_weights: int


def load_gene_name_map(path: Optional[Path]) -> dict[str, str]:
    """
    `ensembl id -> gene symbol`, from a two-column TSV such as
    `data/mdd_genes.tsv` (header `ENSID`/`Gene`).

    The symbol is cosmetic: S-PrediXcan copies it into the `gene_name` output
    column and never uses it in the association.
    """
    if path is None:
        return {}
    path = Path(path)
    if not path.exists():
        logging.warning("Gene name map %s does not exist; gene_name will be blank.", path)
        return {}
    table = pd.read_csv(path, sep="\t", dtype=str).fillna("")
    if table.shape[1] < 2:
        logging.warning("Gene name map %s needs at least two columns; ignoring.", path)
        return {}
    keys = table.iloc[:, 0].astype(str).str.split(".").str[0]
    return dict(zip(keys, table.iloc[:, 1].astype(str)))


def attach_gene_names(frame: pd.DataFrame, names: dict[str, str]) -> pd.DataFrame:
    """
    Fill `gene_name` from `{versionless Ensembl id -> symbol}`.

    Existing symbols are kept when they already look like a gene name.
    Ensembl ids, blanks, and missing values are replaced so the plots can
    label every gene with the GTF name that `get_shared_genes.py` wrote.
    """
    if frame is None or frame.empty or "gene" not in frame.columns or not names:
        return frame
    frame = frame.copy()
    keys = frame["gene"].astype(str).str.split(".").str[0].str.strip().str.upper()
    mapped = keys.map(names)
    if "gene_name" not in frame.columns:
        frame["gene_name"] = mapped
        return frame

    current = frame["gene_name"].astype(str)
    blank = (
        frame["gene_name"].isna()
        | current.str.strip().isin({"", "nan", "none"})
        | current.str.upper().eq(keys)
        | current.str.upper().str.startswith("ENSG")
    )
    replace = blank & mapped.notna()
    frame.loc[replace, "gene_name"] = mapped[replace]
    still_missing = frame["gene_name"].isna() & mapped.notna()
    frame.loc[still_missing, "gene_name"] = mapped[still_missing]
    return frame


def write_model_db(
    path: Path,
    draw: Draw,
    snp_sets: dict[str, GeneSnps],
    snp_table: dict[str, dict[str, GeneSnpInfo]],
    standardized: bool,
    gene_names: Optional[dict[str, str]] = None,
) -> ModelDbStats:
    """
    Write one draw as a S-PrediXcan model DB.

    Only genes present in `snp_table` (i.e. that made it into the covariance)
    and only nonzero weights are written; a zero weight contributes nothing to
    the association but would inflate `n_snps_in_model`.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        path.unlink()
    gene_names = gene_names or {}

    weight_rows: list[tuple[str, str, float, str, str]] = []
    extra_rows: list[tuple[str, str, int, None, None, None]] = []
    n_missing = 0
    n_zero = 0

    for gene in sorted(draw.coefs):
        gene_snps = snp_table.get(gene)
        if gene_snps is None:
            continue
        values = draw.coefs[gene]
        snp_ids = snp_sets[gene].snp_ids

        rows_for_gene: list[tuple[str, str, float, str, str]] = []
        for snp, coef in zip(snp_ids, values):
            if coef == 0.0:
                n_zero += 1
                continue
            info = gene_snps.get(snp)
            if info is None:
                n_missing += 1
                continue
            weight = float(coef) / info.sd if standardized else float(coef)
            if not np.isfinite(weight):
                n_missing += 1
                continue
            rows_for_gene.append(
                (snp, gene, weight, info.non_effect_allele, info.effect_allele)
            )

        if not rows_for_gene:
            continue
        weight_rows.extend(rows_for_gene)
        extra_rows.append(
            (
                gene,
                gene_names.get(gene.split(".")[0], gene),
                len(rows_for_gene),
                # train.py's save_coefficients does not persist the per-gene
                # metrics summarize_models() computes, so there is nothing to
                # report here. S-PrediXcan only echoes these columns.
                None,
                None,
                None,
            )
        )

    connection = sqlite3.connect(str(path))
    try:
        connection.executescript(SCHEMA)
        connection.executemany(
            "INSERT INTO weights (rsid, gene, weight, ref_allele, eff_allele) "
            "VALUES (?, ?, ?, ?, ?)",
            weight_rows,
        )
        connection.executemany(
            'INSERT INTO extra (gene, genename, "n.snps.in.model", "pred.perf.R2", '
            '"pred.perf.pval", "pred.perf.qval") VALUES (?, ?, ?, ?, ?, ?)',
            extra_rows,
        )
        connection.executescript(INDICES)
        connection.commit()
    finally:
        connection.close()

    stats = ModelDbStats(
        n_genes=len(extra_rows),
        n_weights=len(weight_rows),
        n_snps_missing_from_reference=n_missing,
        n_zero_weights=n_zero,
    )
    logging.debug(
        "Wrote %s: %d gene(s), %d weight(s).", path.name, stats.n_genes, stats.n_weights
    )
    return stats


__all__ = ["ModelDbStats", "attach_gene_names", "load_gene_name_map", "write_model_db"]
