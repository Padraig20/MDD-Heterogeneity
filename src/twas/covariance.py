from __future__ import annotations

import gzip
import hashlib
import json
import logging
import shutil
import tempfile
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Optional, Sequence

import numpy as np
from tqdm import tqdm

from src.twas.reference import Reference, bed_path, impute_to_float
from src.twas.weights import GeneSnps

"""
covariance.py

Build the reference LD covariance S-PrediXcan consumes, straight from PLINK.

`metaxcan/software/M01_covariances_correlations.py` produces the same artifact
but can only read PrediXcan-format dosage *text* files
(`PrediXcanFormatUtilities.PrediXcanFormatDosageLoader`); it has no PLINK path,
and materialising a UKB-sized cohort as dosage text is not practical. This
module writes the identical format directly from the .bed files.

`metax.MatrixManager` is strict about that format:

  * `_validate` rejects duplicated rows and requires each gene's rows to be
    contiguous, so genes are written one complete block at a time.
  * `_to_matrix` indexes `entries[id_i][id_j]` for every pair of retained SNPs,
    so the diagonal plus one triangle must be complete. `_rows_to_entries`
    mirrors each pair, and only ever adds an id it has seen as RSID1, which the
    diagonal guarantees.

Where the files live
--------------------
A covariance belongs to the model it was built for, so all three files are
written beside the weights JSON in the model directory:

    <cell type>.json                        the weights
    <cell type>_covariances.txt.gz          the covariance MetaXcan reads
    <cell type>_covariances.snps.txt.gz     per-SNP alleles, position, SD
    <cell type>_covariances.meta.json       provenance and the SNP-set hash

`src/twas/get_covariance_matrices.py` writes them once; `src/twas/run.py` only
ever reads them, and so never needs the genotypes at all.

The sidecar exists because the covariance alone is not enough to build a model
DB. Each SNP's dosage standard deviation is recoverable as `sqrt(diag(cov))`,
but its alleles are not, and the effect/non-effect pair is what S-PrediXcan
aligns the GWAS against. Carrying them here is what makes a model directory
self-sufficient.

Making it fast
--------------
A transcriptome-wide model is ~18k genes over ~1.1M gene-SNP pairs and emits
~47M covariance rows, so the constant factors decide whether this takes minutes
or hours. Four things matter, in descending order:

  * `open_bed.read` defaults to `num_threads=0`, i.e. every core. For the ~60
    columns one gene needs, the thread hand-off costs several times the read
    itself -- at 2k individuals a 61-column read is 5.6x slower on all cores
    than on one. Every read here is `num_threads=1`, and parallelism comes from
    running whole genes concurrently instead.
  * Neighbouring genes select overlapping SNPs, and centring is a full pass
    over the dosages. Genes are therefore processed in chunks: one batched read
    of the chunk's SNP union, centred once, and every gene in the chunk takes
    its block out of that shared centred matrix.
  * `np.cov` re-centres and copies internally, so the centred matrix is formed
    once and multiplied directly.
  * Formatting 47M rows one `handle.write` at a time is slower than building a
    gene's block as a single string, which is what `_gene_block_text` does.

Together those are ~5x on one core; the process pool then scales that out.
Chunks are sized against a memory budget, since the centred matrix is
`n_individuals x chunk_snps` float64 and the individual count is the term that
grows. The output is byte-identical to computing each gene on its own.
"""

COV_HEADER = "GENE\tRSID1\tRSID2\tVALUE\n"
SNP_HEADER = "GENE\tRSID\tCHR\tBP\tEFFECT_ALLELE\tNON_EFFECT_ALLELE\tSD\n"

COV_SUFFIX = "_covariances.txt.gz"
SNP_SUFFIX = "_covariances.snps.txt.gz"
META_SUFFIX = "_covariances.meta.json"

# Per worker, for the centred float64 chunk it holds in memory.
DEFAULT_MEMORY_BUDGET_MB = 512
MIN_CHUNK_SNPS = 64
MAX_CHUNK_SNPS = 16_384

# Several tasks per worker, so a chromosome that finishes early does not leave
# a core idle while chromosome 1 is still going.
TASKS_PER_JOB = 4


@dataclass(frozen=True)
class GeneSnpInfo:
    """Per-SNP reference facts needed to build a model DB."""

    snp: str
    chrom: str
    bp: int
    effect_allele: str
    non_effect_allele: str
    sd: float


@dataclass
class LdReference:
    """A built (or previously built) covariance for one cell type."""

    cell_type: str
    cov_path: Path
    snp_path: Path
    meta_path: Path
    meta: dict

    def gene_positions(self) -> dict[str, tuple[str, int]]:
        """`gene -> (chrom, midpoint bp)`, for Manhattan plotting."""
        positions = {}
        for gene, info in self.meta["genes"].items():
            positions[gene] = (str(info["chrom"]), int(info["mid_bp"]))
        return positions

    def load_snp_table(self) -> dict[str, dict[str, GeneSnpInfo]]:
        """`gene -> {snp -> GeneSnpInfo}` read back from the sidecar."""
        table: dict[str, dict[str, GeneSnpInfo]] = {}
        with gzip.open(self.snp_path, "rt") as handle:
            header = handle.readline()
            if header != SNP_HEADER:
                raise ValueError(f"Unexpected header in {self.snp_path}: {header!r}")
            for line in handle:
                gene, snp, chrom, bp, effect, non_effect, sd = line.rstrip("\n").split("\t")
                table.setdefault(gene, {})[snp] = GeneSnpInfo(
                    snp=snp,
                    chrom=chrom,
                    bp=int(bp),
                    effect_allele=effect,
                    non_effect_allele=non_effect,
                    sd=float(sd),
                )
        return table


def snp_set_hash(snp_sets: dict[str, GeneSnps]) -> str:
    """
    Fingerprint of the gene -> SNP mapping a covariance was built for.

    A covariance is only usable for a model whose SNP sets are identical, so
    this is what pairs one up with its weights JSON. It is checked on load,
    which is how a covariance left behind by an earlier version of a model gets
    caught instead of silently producing a truncated TWAS.
    """
    digest = hashlib.sha256()
    for gene in sorted(snp_sets):
        digest.update(gene.encode())
        digest.update(b"\t")
        digest.update(",".join(snp_sets[gene].snp_ids).encode())
        digest.update(b"\n")
    return digest.hexdigest()


def ld_paths(directory: Path, cell_type: str) -> tuple[Path, Path, Path]:
    """The covariance, SNP sidecar and metadata paths for one cell type."""
    directory = Path(directory)
    return (
        directory / f"{cell_type}{COV_SUFFIX}",
        directory / f"{cell_type}{SNP_SUFFIX}",
        directory / f"{cell_type}{META_SUFFIX}",
    )


def has_covariance(directory: Path, cell_type: str) -> bool:
    """Whether all three covariance files exist for a cell type."""
    return all(path.exists() for path in ld_paths(directory, cell_type))


def load_ld_reference(
    directory: Path,
    cell_type: str,
    expected_hash: Optional[str] = None,
) -> LdReference:
    """Load a covariance from a model directory, validating its SNP fingerprint."""
    cov_path, snp_path, meta_path = ld_paths(directory, cell_type)
    for path in (cov_path, snp_path, meta_path):
        if not path.exists():
            raise FileNotFoundError(
                f"No covariance for cell type '{cell_type}': {path} is missing. "
                "Build it with `python -m src.twas.get_covariance_matrices "
                f"--models-dir {directory} --genotypes ...`."
            )
    with meta_path.open() as handle:
        meta = json.load(handle)
    if expected_hash is not None and meta.get("snp_set_hash") != expected_hash:
        raise ValueError(
            f"{cov_path.name} was built for a different set of model SNPs than "
            f"{cell_type}.json currently specifies. The weights were retrained "
            "since; rebuild the covariance with "
            "`python -m src.twas.get_covariance_matrices --overwrite`."
        )
    return LdReference(
        cell_type=cell_type,
        cov_path=cov_path,
        snp_path=snp_path,
        meta_path=meta_path,
        meta=meta,
    )


@dataclass(frozen=True)
class _GenePlan:
    """One gene's model SNPs, already resolved against the reference panel."""

    gene: str
    chrom: str
    snps: tuple[str, ...]
    var_indices: np.ndarray
    bps: tuple[int, ...]
    effect_alleles: tuple[str, ...]
    non_effect_alleles: tuple[str, ...]
    n_model_snps: int


@dataclass(frozen=True)
class _Task:
    """A contiguous run of same-chromosome genes, handled by one worker."""

    index: int
    plans: list[_GenePlan]


@dataclass(frozen=True)
class _ShardResult:
    index: int
    cov_path: Path
    snp_path: Path
    genes: dict[str, dict]
    n_monomorphic_snps: int
    n_dropped_genes: int


@dataclass(frozen=True)
class _WorkerConfig:
    """Everything a worker needs that does not vary between tasks."""

    genotype_dir: Path
    bed_template: str
    counts: dict[str, tuple[int, int]]
    row_index: dict[str, np.ndarray]
    shared_row_index: Optional[np.ndarray]
    compression_level: int
    chunk_snps: int
    shard_dir: Path

    def rows_for(self, chrom: str) -> np.ndarray:
        if self.shared_row_index is not None:
            return self.shared_row_index
        return self.row_index[chrom]


# Worker-process state. Set once per process rather than shipped per task: the
# row index alone is `n_individuals` int64 per chromosome, which is far larger
# than any single task's payload.
_CONFIG: Optional[_WorkerConfig] = None
_HANDLES: dict[str, object] = {}
_THREAD_LIMITS = None


def _limit_blas_threads():
    """
    Hold every BLAS pool to one thread, or None if that is not possible.

    Each worker's gemms are small and there are `jobs` of them running, so
    letting each fan out over all cores only makes them contend. Returned as a
    context manager so the single-worker path, which runs in the parent, can
    put the limits back afterwards.
    """
    try:
        from threadpoolctl import threadpool_limits

        return threadpool_limits(limits=1)
    except Exception:  # noqa: BLE001 - an optional speedup, never a hard failure
        return None


def _worker_init(config: _WorkerConfig) -> None:
    global _CONFIG, _THREAD_LIMITS
    _CONFIG = config
    _HANDLES.clear()
    # The worker process exists only to run tasks, so this is never unwound.
    _THREAD_LIMITS = _limit_blas_threads()


def _handle(chrom: str):
    """A cached `open_bed` handle, so a task opens each .bed at most once."""
    cached = _HANDLES.get(chrom)
    if cached is None:
        from bed_reader import open_bed

        assert _CONFIG is not None
        iid_count, sid_count = _CONFIG.counts[chrom]
        cached = open_bed(
            str(bed_path(_CONFIG.genotype_dir, _CONFIG.bed_template, chrom)),
            # Given these, open_bed skips counting the .bim's lines, which on a
            # UKB chromosome costs far more than the reads that follow.
            iid_count=iid_count,
            sid_count=sid_count,
        )
        _HANDLES[chrom] = cached
    return cached


def _gene_block_text(gene: str, snps: np.ndarray, cov: np.ndarray) -> str:
    """
    One gene's complete covariance block as a single string.

    The diagonal plus the upper triangle in position order, with no duplicated
    (RSID1, RSID2) pair. `np.triu_indices` walks the triangle row by row, which
    is the order `MatrixManager` expects: (0,0), (0,1) ... (1,1), (1,2) ...

    Returning one string rather than writing row by row is worth ~1.5x, since
    the row count is quadratic in the gene's SNP count. The indices are *not*
    cached across genes: at the real distribution of gene sizes that saves
    around 1% of this function, and would hold tens of MB of index arrays in
    every worker.
    """
    rows, cols = np.triu_indices(len(snps))
    return "".join([
        f"{gene}\t{left}\t{right}\t{value:.10g}\n"
        for left, right, value in zip(snps[rows], snps[cols], cov[rows, cols])
    ])


def _chunks(plans: Sequence[_GenePlan], budget: int) -> Iterator[list[_GenePlan]]:
    """
    Consecutive genes grouped so their SNP union stays under `budget`.

    Genes arrive in position order, so a chunk is a contiguous stretch of the
    chromosome and its union is exactly what one batched read should fetch.
    """
    chunk: list[_GenePlan] = []
    seen: set[int] = set()
    for plan in plans:
        wanted = set(plan.var_indices.tolist())
        if chunk and len(seen | wanted) > budget:
            yield chunk
            chunk, seen = [], set()
        chunk.append(plan)
        seen |= wanted
    if chunk:
        yield chunk


def _build_shard(task: _Task) -> _ShardResult:
    """Compute and write one task's slice of the covariance and the sidecar."""
    config = _CONFIG
    assert config is not None
    chrom = task.plans[0].chrom
    handle = _handle(chrom)
    rows = config.rows_for(chrom)
    n_individuals = int(rows.size)
    if n_individuals < 3:
        raise ValueError(
            f"Only {n_individuals} reference individual(s); a covariance needs at least 3."
        )

    cov_path = config.shard_dir / f"cov_{task.index:05d}.gz"
    snp_path = config.shard_dir / f"snp_{task.index:05d}.gz"
    genes: dict[str, dict] = {}
    n_monomorphic = 0
    n_dropped = 0

    with gzip.open(
        cov_path, "wt", newline="", compresslevel=config.compression_level
    ) as cov_out, gzip.open(
        snp_path, "wt", newline="", compresslevel=config.compression_level
    ) as snp_out:
        for chunk in _chunks(task.plans, config.chunk_snps):
            columns = np.unique(np.concatenate([plan.var_indices for plan in chunk]))
            centred = impute_to_float(
                handle.read(index=np.s_[rows, columns], dtype="int8", num_threads=1)
            )
            centred -= centred.mean(axis=0, keepdims=True)

            for plan in chunk:
                local = np.searchsorted(columns, plan.var_indices)
                block = np.ascontiguousarray(centred[:, local])
                cov = (block.T @ block) / (n_individuals - 1)
                sds = np.sqrt(np.clip(np.diag(cov), 0.0, None))

                # A monomorphic SNP has zero variance, so it carries no LD
                # information and its standardized-to-raw rescale would divide
                # by zero. Drop it from the block entirely.
                keep = sds > 0.0
                snps = np.asarray(plan.snps, dtype=object)
                bps, effect = plan.bps, plan.effect_alleles
                non_effect = plan.non_effect_alleles
                if not keep.all():
                    n_monomorphic += int((~keep).sum())
                    if not keep.any():
                        n_dropped += 1
                        continue
                    kept = np.flatnonzero(keep)
                    cov = cov[np.ix_(kept, kept)]
                    sds = sds[kept]
                    snps = snps[kept]
                    bps = tuple(bps[i] for i in kept)
                    effect = tuple(effect[i] for i in kept)
                    non_effect = tuple(non_effect[i] for i in kept)

                cov_out.write(_gene_block_text(plan.gene, snps, cov))
                snp_out.write("".join([
                    f"{plan.gene}\t{snp}\t{chrom}\t{bp}\t{ea}\t{nea}\t{sd:.10g}\n"
                    for snp, bp, ea, nea, sd in zip(snps, bps, effect, non_effect, sds)
                ]))
                genes[plan.gene] = {
                    "chrom": chrom,
                    "start_bp": int(min(bps)),
                    "end_bp": int(max(bps)),
                    "mid_bp": int((min(bps) + max(bps)) // 2),
                    "n_snps_in_model": plan.n_model_snps,
                    "n_snps_in_cov": len(snps),
                }

    return _ShardResult(
        index=task.index,
        cov_path=cov_path,
        snp_path=snp_path,
        genes=genes,
        n_monomorphic_snps=n_monomorphic,
        n_dropped_genes=n_dropped,
    )


def _plan_genes(
    snp_sets: dict[str, GeneSnps],
    reference: Reference,
    max_snps_in_gene: Optional[int],
) -> tuple[list[_GenePlan], dict]:
    """
    Resolve every gene's model SNPs against the reference, in output order.

    Genes are ordered by chromosome then name, and each gene's SNPs by
    position, which is how `M01` lays its triangle out and what gives the
    covariance a stable, reproducible layout.
    """
    counters = {"missing_snps": 0, "dropped_genes": 0, "skipped_large": 0}
    plans: list[_GenePlan] = []

    ordered = sorted(
        snp_sets, key=lambda gene: (_chrom_sort_key(snp_sets[gene].chrom), gene)
    )
    for gene in ordered:
        entry = snp_sets[gene]
        records = []
        for snp in entry.snp_ids:
            record = reference.snp_index.get(snp)
            if record is not None:
                records.append((snp, record))
        counters["missing_snps"] += len(entry.snp_ids) - len(records)
        if not records:
            counters["dropped_genes"] += 1
            continue

        chroms = {record.chrom for _, record in records}
        if len(chroms) > 1:
            raise ValueError(
                f"Gene {gene} has model SNPs on several chromosomes: {sorted(chroms)}."
            )
        if max_snps_in_gene is not None and len(records) > max_snps_in_gene:
            logging.debug(
                "Skipping gene %s: %d SNPs exceeds --max-snps-in-gene.",
                gene, len(records),
            )
            counters["skipped_large"] += 1
            counters["dropped_genes"] += 1
            continue

        records.sort(key=lambda item: item[1].bp)
        plans.append(_GenePlan(
            gene=gene,
            chrom=records[0][1].chrom,
            snps=tuple(snp for snp, _ in records),
            var_indices=np.array(
                [record.var_index for _, record in records], dtype=np.int64
            ),
            bps=tuple(record.bp for _, record in records),
            effect_alleles=tuple(record.effect_allele for _, record in records),
            non_effect_alleles=tuple(
                record.non_effect_allele for _, record in records
            ),
            n_model_snps=len(entry.snp_ids),
        ))
    return plans, counters


def _partition(plans: Sequence[_GenePlan], n_tasks: int) -> list[_Task]:
    """
    Split the genes into contiguous, same-chromosome tasks of similar cost.

    Both the matrix product and the row count scale with the square of a gene's
    SNP count, so that is what the tasks are balanced on rather than gene
    count: a few hundred-SNP genes outweigh a great many small ones.
    """
    if not plans:
        return []
    weights = np.array([len(plan.snps) ** 2 for plan in plans], dtype=np.float64)
    target = weights.sum() / max(n_tasks, 1)

    tasks: list[_Task] = []
    current: list[_GenePlan] = []
    accumulated = 0.0
    for plan, weight in zip(plans, weights):
        crosses_chromosome = bool(current) and plan.chrom != current[-1].chrom
        if current and (crosses_chromosome or accumulated >= target):
            tasks.append(_Task(index=len(tasks), plans=current))
            current, accumulated = [], 0.0
        current.append(plan)
        accumulated += weight
    if current:
        tasks.append(_Task(index=len(tasks), plans=current))
    return tasks


def _shared_row_index(
    row_index: dict[str, np.ndarray]
) -> tuple[dict[str, np.ndarray], Optional[np.ndarray]]:
    """
    Collapse the per-chromosome row indices when they all agree.

    They normally do -- the .fam files of one cohort are identical -- and the
    index is one int64 per individual per chromosome, so collapsing it cuts
    what has to be shipped to each worker by a factor of 22.
    """
    values = list(row_index.values())
    if not values:
        return {}, None
    first = values[0]
    if all(np.array_equal(first, other) for other in values[1:]):
        return {}, first
    return dict(row_index), None


def _concatenate(header: str, shards: Sequence[Path], destination: Path) -> None:
    """
    Join the workers' gzip shards into one file, header first.

    Concatenated gzip members are themselves a valid gzip stream, which both
    `gzip.open` and the `pandas.read_table` inside `metax.MatrixManager` read
    transparently. Shards therefore need no recompression to be joined.
    """
    with destination.open("wb") as out:
        out.write(gzip.compress(header.encode(), compresslevel=1))
        for shard in shards:
            with shard.open("rb") as source:
                shutil.copyfileobj(source, out, 4 << 20)


def chunk_snp_budget(n_individuals: int, memory_budget_mb: int) -> int:
    """How many SNPs one chunk's centred float64 matrix may span."""
    per_snp = 8 * max(n_individuals, 1)
    return int(np.clip(
        (memory_budget_mb << 20) // per_snp, MIN_CHUNK_SNPS, MAX_CHUNK_SNPS
    ))


def build_covariance(
    cell_type: str,
    snp_sets: dict[str, GeneSnps],
    reference: Reference,
    output_dir: Path,
    max_snps_in_gene: Optional[int] = None,
    compression_level: int = 1,
    overwrite: bool = False,
    jobs: int = 1,
    memory_budget_mb: int = DEFAULT_MEMORY_BUDGET_MB,
    show_progress: bool = True,
) -> LdReference:
    """
    Compute and write one cell type's covariance into its model directory.

    One covariance serves every draw of a model: `EnsembleLR.save_coefficients`
    applies the same PIP mask to the pooled weights and to every
    member-bootstrap replicate, so all draws share a gene's SNP set.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cov_path, snp_path, meta_path = ld_paths(output_dir, cell_type)
    fingerprint = snp_set_hash(snp_sets)

    if not overwrite and meta_path.exists():
        try:
            existing = load_ld_reference(output_dir, cell_type, expected_hash=fingerprint)
        except (FileNotFoundError, ValueError) as error:
            logging.info("Rebuilding the covariance for '%s': %s", cell_type, error)
        else:
            logging.info(
                "Reusing the existing covariance for '%s' (%s).",
                cell_type, cov_path.name,
            )
            return existing

    plans, counters = _plan_genes(snp_sets, reference, max_snps_in_gene)
    if not plans:
        raise ValueError(
            f"No gene of '{cell_type}' kept a single SNP present in the reference "
            "panel; the models and the genotypes do not match."
        )

    n_individuals = reference.n_individuals
    chunk_snps = chunk_snp_budget(n_individuals, memory_budget_mb)
    jobs = max(1, jobs)
    tasks = _partition(plans, jobs * TASKS_PER_JOB)
    logging.info(
        "Covariance for '%s': %d gene(s) over %d individual(s), %d task(s) on %d "
        "worker(s), %d SNP(s) per read.",
        cell_type, len(plans), n_individuals, len(tasks), jobs, chunk_snps,
    )

    per_chrom_rows, shared_rows = _shared_row_index(reference.row_index)
    shard_dir = Path(tempfile.mkdtemp(prefix=f".{cell_type}_shards_", dir=output_dir))
    config = _WorkerConfig(
        genotype_dir=reference.genotype_dir,
        bed_template=reference.bed_template,
        counts=reference.counts,
        row_index=per_chrom_rows,
        shared_row_index=shared_rows,
        compression_level=compression_level,
        chunk_snps=chunk_snps,
        shard_dir=shard_dir,
    )

    genes_meta: dict[str, dict] = {}
    n_monomorphic = 0
    tmp_cov = cov_path.with_suffix(cov_path.suffix + ".partial")
    tmp_snp = snp_path.with_suffix(snp_path.suffix + ".partial")
    try:
        results = _run_tasks(tasks, config, jobs, cell_type, show_progress)
        for result in results:
            genes_meta.update(result.genes)
            n_monomorphic += result.n_monomorphic_snps
            counters["dropped_genes"] += result.n_dropped_genes

        _concatenate(COV_HEADER, [r.cov_path for r in results], tmp_cov)
        _concatenate(SNP_HEADER, [r.snp_path for r in results], tmp_snp)

        meta = {
            "cell_type": cell_type,
            "snp_set_hash": fingerprint,
            "n_genes_requested": len(snp_sets),
            "n_genes_written": len(genes_meta),
            "n_genes_dropped": counters["dropped_genes"],
            "n_model_snps_missing_from_reference": counters["missing_snps"],
            "n_monomorphic_snps_dropped": n_monomorphic,
            "n_genes_skipped_too_large": counters["skipped_large"],
            "max_snps_in_gene": max_snps_in_gene,
            "compression_level": compression_level,
            "reference": reference.selection,
            "genes": genes_meta,
        }
        with meta_path.open("w") as handle:
            json.dump(meta, handle, indent=2)
        tmp_cov.replace(cov_path)
        tmp_snp.replace(snp_path)
    finally:
        for path in (tmp_cov, tmp_snp):
            if path.exists():
                path.unlink()
        shutil.rmtree(shard_dir, ignore_errors=True)

    logging.info(
        "Built the covariance for '%s': %d gene(s) written, %d dropped, %d model "
        "SNP(s) absent from the reference, %d monomorphic SNP(s) dropped.",
        cell_type, len(genes_meta), counters["dropped_genes"],
        counters["missing_snps"], n_monomorphic,
    )
    return LdReference(
        cell_type=cell_type,
        cov_path=cov_path,
        snp_path=snp_path,
        meta_path=meta_path,
        meta=meta,
    )


def _run_tasks(
    tasks: Sequence[_Task],
    config: _WorkerConfig,
    jobs: int,
    cell_type: str,
    show_progress: bool,
) -> list[_ShardResult]:
    """
    Run the tasks, in-process when there is only one worker.

    Shards are returned in task order, not completion order: the covariance has
    to come back out in the order the genes were planned in.
    """
    global _CONFIG
    results: list[Optional[_ShardResult]] = [None] * len(tasks)
    progress = tqdm(
        total=len(tasks),
        desc=f"Covariance ({cell_type})",
        leave=False,
        disable=not show_progress,
    )
    try:
        if jobs == 1:
            # In-process, so the thread limit and the module globals have to be
            # unwound afterwards rather than dying with the worker.
            _CONFIG = config
            _HANDLES.clear()
            limits = _limit_blas_threads()
            try:
                for task in tasks:
                    results[task.index] = _build_shard(task)
                    progress.update(1)
            finally:
                _CONFIG = None
                _HANDLES.clear()
                if limits is not None:
                    limits.restore_original_limits()
        else:
            with ProcessPoolExecutor(
                max_workers=jobs, initializer=_worker_init, initargs=(config,)
            ) as executor:
                for result in executor.map(_build_shard, tasks):
                    results[result.index] = result
                    progress.update(1)
    finally:
        progress.close()
    return [result for result in results if result is not None]


def _chrom_sort_key(chrom: str) -> tuple[int, str]:
    text = str(chrom).removeprefix("chr")
    return (int(text), "") if text.isdigit() else (99, text)


__all__ = [
    "COV_HEADER",
    "COV_SUFFIX",
    "DEFAULT_MEMORY_BUDGET_MB",
    "GeneSnpInfo",
    "LdReference",
    "META_SUFFIX",
    "SNP_HEADER",
    "SNP_SUFFIX",
    "build_covariance",
    "chunk_snp_budget",
    "has_covariance",
    "ld_paths",
    "load_ld_reference",
    "snp_set_hash",
]
