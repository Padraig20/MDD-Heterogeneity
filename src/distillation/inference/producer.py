from __future__ import annotations

import logging
import multiprocessing as mp
from multiprocessing.shared_memory import SharedMemory
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from bed_reader import open_bed

from distillation.inference.inference import UKB_BED_TEMPLATE, ONEK1K_BED_TEMPLATE


LOGGER = logging.getLogger(__name__)


def setup_worker_logging(log_path: Path, level: int = logging.INFO) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)

    root = logging.getLogger()
    root.handlers.clear()
    root.setLevel(level)

    formatter = logging.Formatter("%(asctime)s %(levelname)s %(processName)s %(name)s: %(message)s")

    file_handler = logging.FileHandler(log_path)
    file_handler.setLevel(level)
    file_handler.setFormatter(formatter)
    root.addHandler(file_handler)


class GenotypeResourceManager:
    """ Lazily opens BED/BIM resources per chromosome inside a producer process. """

    def __init__(self, input_dir: Path, bed_template: str = UKB_BED_TEMPLATE) -> None:
        self.input_dir = Path(input_dir)
        self.bed_template = bed_template
        self._beds: dict[str, Any] = {}
        self._snp_to_col: dict[str, dict[str, int]] = {}
        self._sample_ids: np.ndarray | None = None

    def _load_chromosome(self, chrom: str) -> None:
        # enables lazy loading!
        if chrom in self._beds:
            return

        chrom_prefix = self.bed_template.format(chrom=chrom)
        bed_path     = self.input_dir / f"{chrom_prefix}.bed"
        bim_path     = self.input_dir / f"{chrom_prefix}.bim"

        bim = pd.read_csv(
            bim_path,
            sep=r"\s+",
            header=None,
            names=["chrom", "snp", "cm", "bp", "a1", "a2"],
            dtype={"chrom": str, "snp": str, "bp": np.int64},
        )

        bed     = open_bed(str(bed_path))
        snp_ids = bim["snp"].astype(str).to_numpy()
        lookup  = {snp_id: i for i, snp_id in enumerate(snp_ids)}

        current_sample_ids = np.asarray(bed.iid, dtype=str)
        if self._sample_ids is None:
            self._sample_ids = current_sample_ids
        else:
            if (len(self._sample_ids) != len(current_sample_ids)
                or not np.array_equal(self._sample_ids, current_sample_ids)):
                raise ValueError(f"Sample IDs differ across chromosomes; detected mismatch at {chrom}")

        self._beds[chrom]       = bed
        self._snp_to_col[chrom] = lookup

    def fetch_gene_arrays(self, gene: str, spec: dict[str, Any], n_samples: int):
        chrom     = str(spec["chr"])
        intercept = float(spec["intercept"])
        snps      = list(spec["snp_ids"])
        coefs     = list(spec["coefs"])

        if not snps: # no SNPs in model (happens), return intercept only
            return gene, intercept, None, None

        self._load_chromosome(chrom)

        bed        = self._beds[chrom]
        snp_lookup = self._snp_to_col[chrom]

        found_cols: list[int]    = []
        found_coefs: list[float] = []

        missing_count = 0

        for snp_id, coef in zip(snps, coefs):
            col = snp_lookup.get(str(snp_id))
            if col is None:
                missing_count += 1
                continue
            found_cols.append(col)
            found_coefs.append(float(coef))

        if not found_cols:
            raise ValueError(f"Gene {gene}: none of the SNPs were found in the genotype data for chromosome {chrom}")

        if missing_count:
            LOGGER.debug(
                "Gene %s: found %d/%d SNPs; missing %d",
                gene,
                len(found_cols),
                len(snps),
                missing_count,
            )

        cols = np.asarray(found_cols, dtype=np.int64)
        G    = bed.read(index=np.s_[:, cols], dtype=np.float32)
        G    = np.nan_to_num(G, nan=0.0) # impute zeros for missing genotypes

        return (
            gene,
            intercept,
            G.astype(np.float32, copy=False),
            np.asarray(found_coefs, dtype=np.float32),
        )

    def close(self) -> None:
        for bed in self._beds.values():
            try:
                bed.close()
            except Exception:
                pass
        self._beds.clear()
        self._snp_to_col.clear()


def _open_shared_slot_buffers(
    geno_shm_names: list[str],
    coef_shm_names: list[str],
    geno_capacity_floats: int,
    coef_capacity_floats: int,
) -> tuple[list[SharedMemory], list[np.ndarray], list[SharedMemory], list[np.ndarray]]:
    geno_shms: list[SharedMemory] = []
    geno_views: list[np.ndarray]  = []
    coef_shms: list[SharedMemory] = []
    coef_views: list[np.ndarray]  = []

    for name in geno_shm_names:
        shm = SharedMemory(name=name)
        geno_shms.append(shm)
        geno_views.append(np.ndarray((geno_capacity_floats,), dtype=np.float32, buffer=shm.buf))

    for name in coef_shm_names:
        shm = SharedMemory(name=name)
        coef_shms.append(shm)
        coef_views.append(np.ndarray((coef_capacity_floats,), dtype=np.float32, buffer=shm.buf))

    return geno_shms, geno_views, coef_shms, coef_views


def producer_main(
    producer_id: int,
    genes: list[tuple[str, dict[str, Any]]],
    input_dir: str,
    ready_queue: mp.Queue,
    free_slot_queue: mp.Queue,
    n_samples: int,
    geno_shm_names: list[str],
    coef_shm_names: list[str],
    geno_capacity_floats: int,
    coef_capacity_floats: int,
    max_genes_per_slot: int,
    log_dir: str | None = None,
    log_level: int = logging.INFO,
    bed_template: str = UKB_BED_TEMPLATE,
) -> None:
    if log_dir is not None:
        setup_worker_logging(Path(log_dir) / f"producer-{producer_id}.log", level=log_level)

    LOGGER.info("Producer %d started with %d genes", producer_id, len(genes))

    manager = GenotypeResourceManager(Path(input_dir), bed_template=bed_template)
    geno_shms: list[SharedMemory] = []
    coef_shms: list[SharedMemory] = []

    try:
        geno_shms, geno_views, coef_shms, coef_views = _open_shared_slot_buffers(
            geno_shm_names=geno_shm_names,
            coef_shm_names=coef_shm_names,
            geno_capacity_floats=geno_capacity_floats,
            coef_capacity_floats=coef_capacity_floats,
        )

        slot_id: int | None = None
        geno_view: np.ndarray | None = None
        coef_view: np.ndarray | None = None

        gene_names: list[str]   = []
        intercepts: list[float] = []
        lengths: list[int]      = []
        geno_offsets: list[int] = []
        coef_offsets: list[int] = []

        geno_used = 0
        coef_used = 0

        # --------------------------------------------------------------------------

        def start_new_slot() -> None:
            nonlocal slot_id, geno_view, coef_view
            nonlocal gene_names, intercepts, lengths, geno_offsets, coef_offsets
            nonlocal geno_used, coef_used

            slot_id   = free_slot_queue.get()
            geno_view = geno_views[slot_id]
            coef_view = coef_views[slot_id]

            gene_names   = []
            intercepts   = []
            lengths      = []
            geno_offsets = []
            coef_offsets = []

            geno_used = 0
            coef_used = 0

        def flush_slot() -> None:
            nonlocal slot_id, geno_view, coef_view
            if slot_id is None or geno_view is None or coef_view is None:
                return
            if not gene_names:
                free_slot_queue.put(slot_id)
                slot_id   = None
                geno_view = None
                coef_view = None
                return

            ready_queue.put(
                {
                    "type": "slot_batch",
                    "producer_id": producer_id,
                    "slot_id": slot_id,
                    "gene_names": list(gene_names),
                    "intercepts": list(intercepts),
                    "lengths": list(lengths),
                    "geno_offsets": list(geno_offsets),
                    "coef_offsets": list(coef_offsets),
                    "geno_used": int(geno_used),
                    "coef_used": int(coef_used),
                }
            )
            LOGGER.debug(
                "Producer %d flushed slot %d with %d genes, %d geno floats, %d coef floats",
                producer_id,
                slot_id,
                len(gene_names),
                geno_used,
                coef_used,
            )
            slot_id   = None
            geno_view = None
            coef_view = None

        # --------------------------------------------------------------------------

        start_new_slot()

        for i, (gene, spec) in enumerate(genes):
            gene_name, intercept, G, coefs = manager.fetch_gene_arrays(
                gene=gene,
                spec=spec,
                n_samples=n_samples,
            )

            if G is None or coefs is None:
                required_geno = 0
                required_coef = 0
                snp_count     = 0
            else:
                required_geno = int(G.size)
                required_coef = int(coefs.size)
                snp_count     = int(coefs.size)

            if required_geno > geno_capacity_floats or required_coef > coef_capacity_floats:
                raise ValueError(
                    f"Gene {gene_name} does not fit into one shared slot. USE MORE MEMORY! "
                    f"required_geno={required_geno}, geno_capacity={geno_capacity_floats}, "
                    f"required_coef={required_coef}, coef_capacity={coef_capacity_floats}"
                )

            slot_full = (
                len(gene_names) >= max_genes_per_slot
                or geno_used + required_geno > geno_capacity_floats
                or coef_used + required_coef > coef_capacity_floats
            )

            if slot_full and gene_names:
                flush_slot()
                start_new_slot()

            gene_names.append(gene_name)
            intercepts.append(intercept)
            lengths.append(snp_count)
            geno_offsets.append(geno_used)
            coef_offsets.append(coef_used)

            if required_geno > 0:
                assert geno_view is not None
                geno_view[geno_used:geno_used + required_geno] = G.reshape(-1)
                geno_used += required_geno

            if required_coef > 0:
                assert coef_view is not None
                coef_view[coef_used:coef_used + required_coef] = coefs.reshape(-1)
                coef_used += required_coef
            
            LOGGER.info(
                "Producer %d processed gene %d/%d: %s (snps=%d, geno_floats=%d, coef_floats=%d, slot_id=%s)",
                producer_id,
                i + 1,
                len(genes),
                gene_name,
                snp_count,
                required_geno,
                required_coef,
                slot_id,
            )

        flush_slot()

        LOGGER.info("Producer %d finished successfully", producer_id)

    except Exception as exc:
        LOGGER.exception("Producer %d failed", producer_id)
        ready_queue.put(
            {
                "type": "error",
                "producer_id": producer_id,
                "message": repr(exc),
            }
        )
        raise

    finally:
        manager.close()
        for shm in geno_shms:
            try:
                shm.close()
            except Exception:
                pass
        for shm in coef_shms:
            try:
                shm.close()
            except Exception:
                pass
        ready_queue.put({"type": "producer_done", "producer_id": producer_id})