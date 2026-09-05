from __future__ import annotations

import json
import logging
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Optional

import numpy as np

"""
weights.py

Read the weights JSONs written by `src/distillation/train.py` and expand them
into the list of *draws* a TWAS has to be run for.

The three model families each save a different layout:

  * `LR.save_coefficients`            -> {"snp_ids", "coefs", "chr", "intercept"}
  * `ProbabilisticLR.save_coefficients` -> {"chr", "mean": {...}, "aleatoric": {...},
                                            "epistemic": {...}}
  * `EnsembleLR.save_coefficients`    -> the above plus "members" and "metadata"

`LR` and `ProbabilisticLR` fit on a standardized design matrix and persist
`enet.coef_` directly, so their weights live in standardized-X units and have to
be divided by each SNP's dosage standard deviation before S-PrediXcan sees them
(`model_db.py` does that). `EnsembleLR` already records
`metadata.coefficient_scale == "raw_dosage"` and needs no conversion.
"""

# How a JSON is turned into draws.
KIND_SINGLE = "single"  # one TWAS from the single point-estimate weight vector
KIND_MEAN = "mean"      # one TWAS from an ensemble's pooled mean weights
KIND_MI = "mi"          # one TWAS per member-bootstrap fit, aggregated afterwards
KIND_AUTO = "auto"

MODEL_KINDS = (KIND_AUTO, KIND_SINGLE, KIND_MEAN, KIND_MI)

# Which model family wrote the JSON.
SOURCE_LR = "lr"
SOURCE_PROBABILISTIC = "probabilistic"
SOURCE_ENSEMBLE = "ensemble"

RAW_DOSAGE_SCALE = "raw_dosage"


@dataclass(frozen=True)
class GeneSnps:
    """The SNP set of one gene, shared by every draw of a model."""

    gene: str
    chrom: str
    snp_ids: tuple[str, ...]


@dataclass(frozen=True)
class Draw:
    """
    One weight vector per gene, i.e. exactly one S-PrediXcan run.

    `coefs[gene]` is aligned with `ModelSpec.snp_sets[gene].snp_ids`.
    """

    draw_id: str
    coefs: dict[str, np.ndarray]


@dataclass
class ModelSpec:
    cell_type: str
    path: Path
    source: str
    kind: str
    # True when the coefficients are in standardized-X units and need to be
    # divided by the per-SNP dosage standard deviation.
    standardized: bool
    snp_sets: dict[str, GeneSnps]
    draws: list[Draw]

    @property
    def genes(self) -> list[str]:
        return list(self.snp_sets)

    def all_snps(self) -> set[str]:
        snps: set[str] = set()
        for entry in self.snp_sets.values():
            snps.update(entry.snp_ids)
        return snps

    def n_model_snps(self) -> int:
        return sum(len(entry.snp_ids) for entry in self.snp_sets.values())


def detect_source(payload: dict) -> str:
    """Which model family wrote this JSON, from the layout of its gene entries."""
    for entry in payload.values():
        if not isinstance(entry, dict):
            continue
        if "members" in entry or "metadata" in entry:
            return SOURCE_ENSEMBLE
        if "mean" in entry and "aleatoric" in entry and "epistemic" in entry:
            return SOURCE_PROBABILISTIC
        if "snp_ids" in entry and "coefs" in entry:
            return SOURCE_LR
        break
    raise ValueError(
        "Could not recognise the weights JSON layout; expected the output of "
        "LR, ProbabilisticLR or EnsembleLR save_coefficients()."
    )


def default_kind(source: str) -> str:
    """The draw expansion `--model-kind auto` picks for a given model family."""
    return KIND_MEAN if source == SOURCE_ENSEMBLE else KIND_SINGLE


def _point_estimate(entry: dict, source: str) -> tuple[list, list]:
    """`(snp_ids, coefs)` of the entry's single point-estimate weight vector."""
    if source == SOURCE_PROBABILISTIC:
        head = entry["mean"]
        return head["snp_ids"], head["coefs"]
    return entry["snp_ids"], entry["coefs"]


def _entry_chrom(entry: dict) -> str:
    return str(entry["chr"])


def _bootstrap_replicates(entry: dict) -> list[tuple[str, list]]:
    """
    Every member-bootstrap elastic-net fit of one ensemble gene entry, as
    `(draw_id, coefs)`. Each replicate is already restricted to the same
    PIP-selected SNPs as the pooled `coefs`, so all draws stay aligned.
    """
    replicates = []
    for member in entry.get("members", []):
        member_id = str(member.get("member_id", len(replicates)))
        for index, coefs in enumerate(member.get("bootstrap_coefs", [])):
            replicates.append((f"member{member_id}_boot{index}", coefs))
    return replicates


def _draw_ids_for_mi(payload: dict) -> list[str]:
    """
    The union of member-bootstrap draw ids across genes, in first-seen order.

    Genes are fitted independently and a gene can drop out of a member, so the
    id list is built as a union rather than read off a single gene.
    """
    ids: list[str] = []
    seen: set[str] = set()
    for entry in payload.values():
        for draw_id, _ in _bootstrap_replicates(entry):
            if draw_id not in seen:
                seen.add(draw_id)
                ids.append(draw_id)
    return ids


def load_model_json(
    path: Path,
    kind: str = KIND_AUTO,
    mi_draws: Optional[int] = None,
    seed: int = 42,
) -> ModelSpec:
    """
    Load one cell type's weights JSON and expand it into the draws to run.

    Genes with no SNPs (intercept-only fits, which `train.py` does produce) are
    dropped: they carry no genotype signal and S-PrediXcan has nothing to
    associate for them.
    """
    path = Path(path)
    with path.open() as handle:
        payload = json.load(handle)
    if not payload:
        raise ValueError(f"{path} contains no genes.")

    source = detect_source(payload)
    if kind == KIND_AUTO:
        kind = default_kind(source)
    if kind == KIND_MI and source != SOURCE_ENSEMBLE:
        raise ValueError(
            f"{path} was written by the '{source}' model family, which has no "
            "member-bootstrap fits; --model-kind mi needs an EnsembleLR JSON."
        )
    if kind == KIND_MEAN and source != SOURCE_ENSEMBLE:
        # The pooled mean of a one-model family is just its point estimate.
        logging.debug(
            "%s is a '%s' JSON; --model-kind mean is equivalent to single here.",
            path, source,
        )

    standardized = source in (SOURCE_LR, SOURCE_PROBABILISTIC)
    if source == SOURCE_ENSEMBLE:
        scale = _ensemble_coefficient_scale(payload)
        if scale != RAW_DOSAGE_SCALE:
            raise ValueError(
                f"{path} declares metadata.coefficient_scale='{scale}'; only "
                f"'{RAW_DOSAGE_SCALE}' is supported for ensemble models."
            )

    snp_sets = _build_snp_sets(payload, source, path)

    if kind == KIND_MI:
        draws = _mi_draws(payload, snp_sets, mi_draws=mi_draws, seed=seed)
    else:
        draws = [_point_estimate_draw(payload, snp_sets, source)]

    return ModelSpec(
        cell_type=path.stem,
        path=path,
        source=source,
        kind=kind,
        standardized=standardized,
        snp_sets=snp_sets,
        draws=draws,
    )


def _build_snp_sets(payload: dict, source: str, path: Path) -> dict[str, GeneSnps]:
    """
    Each gene's SNP set, in the order the coefficients are stored in.

    Genes with no SNPs (intercept-only fits, which `train.py` does produce) are
    dropped: they carry no genotype signal, so there is nothing for
    S-PrediXcan to associate and nothing to put in a covariance.
    """
    snp_sets: dict[str, GeneSnps] = {}
    n_empty = 0
    for gene, entry in payload.items():
        snp_ids, _ = _point_estimate(entry, source)
        if not snp_ids:
            n_empty += 1
            continue
        snp_sets[gene] = GeneSnps(
            gene=gene, chrom=_entry_chrom(entry), snp_ids=tuple(str(s) for s in snp_ids)
        )
    if n_empty:
        logging.info(
            "%s: skipping %d gene(s) whose model has no SNPs.", path.name, n_empty
        )
    if not snp_sets:
        raise ValueError(f"{path}: every gene's model is empty.")
    return snp_sets


def load_snp_sets(path: Path) -> dict[str, GeneSnps]:
    """
    Just the gene -> SNP mapping of a weights JSON.

    This is all a covariance depends on, and it is identical to
    `load_model_json(path).snp_sets` -- including the `snp_set_hash` computed
    from it -- while skipping the draw expansion, which for an ensemble JSON
    means not materialising every member-bootstrap coefficient vector.
    """
    path = Path(path)
    with path.open() as handle:
        payload = json.load(handle)
    if not payload:
        raise ValueError(f"{path} contains no genes.")
    return _build_snp_sets(payload, detect_source(payload), path)


def discover_models(models_dir: Path, requested: Optional[list[str]]) -> list[Path]:
    """
    The weights JSONs of a model directory.

    `train.py` writes `<cell type with spaces replaced by underscores>.json`, so
    a requested cell type is accepted in either spelling. Covariance metadata
    sidecars live in the same directory and are also `.json`, so they are
    filtered out rather than mistaken for a twenty-third cell type.
    """
    from src.twas.covariance import META_SUFFIX

    available = sorted(
        path for path in Path(models_dir).glob("*.json")
        if not path.name.endswith(META_SUFFIX)
    )
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


def _ensemble_coefficient_scale(payload: dict) -> str:
    for entry in payload.values():
        metadata = entry.get("metadata")
        if isinstance(metadata, dict) and "coefficient_scale" in metadata:
            return str(metadata["coefficient_scale"])
    # Older ensemble JSONs predate the field but were always raw-dosage.
    return RAW_DOSAGE_SCALE


def _point_estimate_draw(
    payload: dict, snp_sets: dict[str, GeneSnps], source: str
) -> Draw:
    coefs = {}
    for gene in snp_sets:
        _, values = _point_estimate(payload[gene], source)
        coefs[gene] = np.asarray(values, dtype=np.float64)
    return Draw(draw_id="point", coefs=coefs)


def _mi_draws(
    payload: dict,
    snp_sets: dict[str, GeneSnps],
    mi_draws: Optional[int],
    seed: int,
) -> list[Draw]:
    draw_ids = _draw_ids_for_mi(payload)
    if not draw_ids:
        raise ValueError(
            "No member-bootstrap fits found; the JSON has no "
            "members[].bootstrap_coefs to run multiple imputation over."
        )
    if mi_draws is not None and 0 < mi_draws < len(draw_ids):
        draw_ids = sorted(random.Random(seed).sample(draw_ids, mi_draws))
        logging.info("Subsampled %d of the available member-bootstrap fits.", mi_draws)

    by_gene: dict[str, dict[str, list]] = {}
    for gene in snp_sets:
        by_gene[gene] = dict(_bootstrap_replicates(payload[gene]))

    draws = []
    for draw_id in draw_ids:
        coefs = {}
        for gene, entry in snp_sets.items():
            values = by_gene[gene].get(draw_id)
            if values is None:
                continue
            array = np.asarray(values, dtype=np.float64)
            if array.size != len(entry.snp_ids):
                raise ValueError(
                    f"Gene {gene}: bootstrap draw {draw_id} has {array.size} "
                    f"coefficients but the model has {len(entry.snp_ids)} SNPs."
                )
            coefs[gene] = array
        if coefs:
            draws.append(Draw(draw_id=draw_id, coefs=coefs))

    logging.info("Expanded ensemble into %d member-bootstrap draw(s).", len(draws))
    return draws


def read_snp_universe(path: Path) -> set[str]:
    """
    Every SNP any gene of one weights JSON refers to.

    Used to index the reference panel once for a whole sweep instead of once
    per cell type: a UKB .bim pass costs far more than re-parsing the JSON.
    """
    with Path(path).open() as handle:
        payload = json.load(handle)
    source = detect_source(payload)
    snps: set[str] = set()
    for entry in payload.values():
        snp_ids, _ = _point_estimate(entry, source)
        snps.update(str(snp) for snp in snp_ids)
    return snps


def iter_nonzero(
    draw: Draw, snp_sets: dict[str, GeneSnps]
) -> Iterator[tuple[str, str, float]]:
    """Yield `(gene, snp_id, coef)` for every nonzero weight of a draw."""
    for gene, values in draw.coefs.items():
        snp_ids = snp_sets[gene].snp_ids
        nonzero = np.flatnonzero(values)
        for position in nonzero:
            yield gene, snp_ids[position], float(values[position])
