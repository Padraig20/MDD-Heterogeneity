from __future__ import annotations

import argparse
import json
import logging
import time
import multiprocessing as mp
import os
from collections import defaultdict
from multiprocessing.shared_memory import SharedMemory
from pathlib import Path
from typing import Any

import numpy as np
from bed_reader import open_bed
from tqdm import tqdm

from src.distillation.inference.consumer import consumer_main
from src.distillation.inference.producer import producer_main


LOGGER = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Use trained model to predict cell type-specific gene expression from genotype data."
    )
    parser.add_argument(
        "-m", "--model-dir",
        type=Path,
        required=True,
        help="Directory containing trained models. One JSON file per cell type."
    )
    parser.add_argument(
        "-i", "--input-dir",
        type=Path,
        required=True,
        help="Directory containing genotype files for each chromosome, e.g. chr1.bed/.bim/.fam."
    )
    parser.add_argument(
        "-o", "--output-dir",
        type=Path,
        required=True,
        help="Directory to save predictions."
    )
    parser.add_argument(
        "--log-dir",
        type=Path,
        default=None,
        help="Directory where worker log files are written."
    )
    parser.add_argument(
        "-p", "--n-producers",
        type=int,
        default=max(1, (os.cpu_count() or 1) - 1),
        help="Number of producer processes."
    )
    parser.add_argument(
        "-b", "--batch-size",
        type=int,
        default=4,
        help="Number of shared-memory slot descriptors the consumer pulls together."
    )
    parser.add_argument(
        "--num-slots",
        type=int,
        default=None,
        help="Number of reusable shared-memory slots. Default: max(2, n_producers + 1)."
    )
    parser.add_argument(
        "--max-genes-per-slot",
        type=int,
        default=8,
        help="Maximum number of genes a producer packs into one shared-memory slot."
    )
    parser.add_argument(
        "--slot-geno-capacity-mb",
        type=int,
        default=512,
        help="Per-slot genotype shared-memory capacity in MB."
    )
    parser.add_argument(
        "--slot-coef-capacity-mb",
        type=int,
        default=16,
        help="Per-slot coefficient shared-memory capacity in MB."
    )
    parser.add_argument(
        "--gpu",
        action="store_true",
        help="Use GPU in the consumer if torch+CUDA are available."
    )
    parser.add_argument(
        "-v", "--verbose",
        action="count",
        default=0,
        help="Increase verbosity."
    )
    return parser.parse_args()


def setup_logging(verbosity: int) -> int:
    level = logging.INFO if verbosity == 0 else logging.DEBUG
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    return level


def load_sample_ids_and_validate(input_dir: Path, chromosomes: list[str]) -> np.ndarray:
    sample_ids: np.ndarray | None = None
    beds = []

    try:
        for chrom in chromosomes:
            chrom_name = f"ukb_imp_v3_chr{chrom}.unrelatedbritishqced.maf001geno9.biallelic"
            bed_path   = input_dir / f"{chrom_name}.bed"

            bed = open_bed(str(bed_path))
            beds.append(bed)

            current_sample_ids = np.asarray(bed.iid, dtype=str)
            if sample_ids is None:
                sample_ids = current_sample_ids
            else:
                if (len(sample_ids) != len(current_sample_ids)
                    or not np.array_equal(sample_ids, current_sample_ids)):
                    raise ValueError(f"Sample IDs differ across chromosomes; detected mismatch at {chrom}")

        if sample_ids is None:
            raise ValueError("No usable BED files found in input_dir.")
        return sample_ids

    finally: # make sure all BEDs are closed
        for bed in beds:
            try:
                bed.close()
            except Exception:
                pass


def group_genes_by_chromosome(model: dict[str, Any]) -> dict[str, list[tuple[str, dict[str, Any]]]]:
    genes_by_chrom: dict[str, list[tuple[str, dict[str, Any]]]] = defaultdict(list)
    for gene, spec in model.items():
        chrom = str(spec["chr"])
        genes_by_chrom[chrom].append((gene, spec))
    return dict(genes_by_chrom)


def assign_chromosomes_to_producers(genes_by_chrom: dict[str, list[tuple[str, dict[str, Any]]]], n_producers: int) -> list[list[tuple[str, dict[str, Any]]]]:
    """ Greedy scheduling of chromosome to producers (not optimal but good enough?). """
    if n_producers <= 1: # just put everything in one producer
        all_genes: list[tuple[str, dict[str, Any]]] = []
        for chrom in sorted(genes_by_chrom.keys(), key=lambda c: (len(genes_by_chrom[c]), c), reverse=True):
            all_genes.extend(genes_by_chrom[chrom])
        return [all_genes]

    producer_gene_lists = [[] for _ in range(n_producers)]
    producer_loads      = [0 for _ in range(n_producers)]

    chrom_items = sorted(
        genes_by_chrom.items(),
        key=lambda kv: (len(kv[1]), kv[0]),
        reverse=True
    )

    LOGGER.debug("chrom \t -> \t prod \t #genes")

    for chrom, genes in chrom_items:
        producer_idx = min(range(n_producers), key=lambda i: producer_loads[i])
        producer_gene_lists[producer_idx].extend(genes)
        producer_loads[producer_idx] += len(genes)

        LOGGER.debug(f"{chrom} \t -> \t {producer_idx} \t {len(genes)}")

    return [genes for genes in producer_gene_lists if genes]


def mb_to_float32_count(size_mb: int) -> int: # only roughly...
    return (size_mb * 1024 * 1024) // np.dtype(np.float32).itemsize


def create_shared_slot_pool(num_slots: int, geno_capacity_floats: int, coef_capacity_floats: int) -> tuple[list[SharedMemory], list[SharedMemory]]:
    geno_shms: list[SharedMemory] = []
    coef_shms: list[SharedMemory] = []

    geno_bytes = geno_capacity_floats * np.dtype(np.float32).itemsize
    coef_bytes = coef_capacity_floats * np.dtype(np.float32).itemsize

    for _ in range(num_slots):
        geno_shms.append(SharedMemory(create=True, size=geno_bytes))
        coef_shms.append(SharedMemory(create=True, size=coef_bytes))

    return geno_shms, coef_shms


def cleanup_shared_slot_pool(geno_shms: list[SharedMemory], coef_shms: list[SharedMemory]) -> None:
    for shm in geno_shms:
        try:
            shm.close()
        except Exception:
            pass
        try:
            shm.unlink()
        except Exception:
            pass

    for shm in coef_shms:
        try:
            shm.close()
        except Exception:
            pass
        try:
            shm.unlink()
        except Exception:
            pass


def predict_cell_type_distributed(
    model: dict[str, Any],
    input_dir: Path,
    sample_ids: np.ndarray,
    out_path: Path,
    n_producers: int,
    batch_size: int,
    num_slots: int,
    max_genes_per_slot: int,
    geno_capacity_floats: int,
    coef_capacity_floats: int,
    use_gpu: bool,
    log_dir: Path | None,
    log_level: int,
) -> None:
    n_samples  = len(sample_ids)
    gene_items = list(model.items())

    if not gene_items:
        raise ValueError("Model contains no genes.")

    genes_by_chrom = group_genes_by_chromosome(model)
    n_producers    = max(1, min(n_producers, len(genes_by_chrom)))
    gene_chunks    = assign_chromosomes_to_producers(
        genes_by_chrom=genes_by_chrom,
        n_producers=n_producers,
    )

    LOGGER.info(
        "Partitioned %d genes across %d producers using %d chromosomes",
        len(gene_items),
        len(gene_chunks),
        len(genes_by_chrom),
    )

    # start parallelism:
    # - producers read gene specs, load genotypes, pack shared-memory slots, and send descriptors to consumer via ready_queue
    # - consumer pulls batches of descriptors, reads from shared-memory slots, runs predictions, and sends results back via result_queue
    # we have two separate queues, one for ready descriptors, one for free slot IDs, 
    # to avoid head-of-line blocking of producers when consumer is busy and all slots are in use

    ctx = mp.get_context("spawn")

    ready_queue: mp.Queue     = ctx.Queue(maxsize=max(1, num_slots))
    free_slot_queue: mp.Queue = ctx.Queue()
    result_queue: mp.Queue    = ctx.Queue()

    # create shared-memory slot pool before starting workers to
    # ensure they can open the shared memory segments

    geno_shms, coef_shms = create_shared_slot_pool(
        num_slots=num_slots,
        geno_capacity_floats=geno_capacity_floats,
        coef_capacity_floats=coef_capacity_floats,
    )

    try:
        for slot_id in range(num_slots):
            free_slot_queue.put(slot_id)

        geno_shm_names = [shm.name for shm in geno_shms]
        coef_shm_names = [shm.name for shm in coef_shms]

        # start multiple producer processes
        producers: list[mp.Process] = []
        for producer_id, gene_chunk in enumerate(gene_chunks):
            proc = ctx.Process(
                target=producer_main,
                kwargs={
                    "producer_id": producer_id,
                    "genes": gene_chunk,
                    "input_dir": str(input_dir),
                    "ready_queue": ready_queue,
                    "free_slot_queue": free_slot_queue,
                    "n_samples": n_samples,
                    "geno_shm_names": geno_shm_names,
                    "coef_shm_names": coef_shm_names,
                    "geno_capacity_floats": geno_capacity_floats,
                    "coef_capacity_floats": coef_capacity_floats,
                    "max_genes_per_slot": max_genes_per_slot,
                    "log_dir": str(log_dir) if log_dir is not None else None,
                    "log_level": log_level,
                },
                name=f"producer-{producer_id}",
            )
            producers.append(proc)

        # start one consumer process
        consumer = ctx.Process(
            target=consumer_main,
            kwargs={
                "ready_queue": ready_queue,
                "free_slot_queue": free_slot_queue,
                "result_queue": result_queue,
                "n_producers": len(producers),
                "batch_size": batch_size,
                "n_samples": n_samples,
                "n_genes": len(gene_items),
                "sample_ids": sample_ids.tolist(),
                "out_path": str(out_path),
                "use_gpu": use_gpu,
                "geno_shm_names": geno_shm_names,
                "coef_shm_names": coef_shm_names,
                "geno_capacity_floats": geno_capacity_floats,
                "coef_capacity_floats": coef_capacity_floats,
                "log_dir": str(log_dir) if log_dir is not None else None,
                "log_level": log_level,
            },
            name="consumer",
        )

        for proc in producers:
            proc.start()
        consumer.start()

        result = result_queue.get()

        for proc in producers:
            proc.join()
        consumer.join()

        # if failure (hopefully not hehe)
        failed = [proc for proc in producers + [consumer] if proc.exitcode not in (0, None)]
        if failed:
            names = ", ".join(f"{proc.name}(exitcode={proc.exitcode})" for proc in failed)
            raise RuntimeError(f"One or more worker processes failed: {names}")
        if result.get("type") == "error":
            raise RuntimeError(f"Consumer failed: {result.get('message')}")

    finally:
        cleanup_shared_slot_pool(geno_shms, coef_shms)


def main() -> None:
    args      = parse_args()
    log_level = setup_logging(args.verbose)
    LOGGER.debug("Arguments: %s", args)

    os.makedirs(args.output_dir, exist_ok=True)
    if args.log_dir is not None:
        args.log_dir.mkdir(parents=True, exist_ok=True)

    chromosomes = [str(c) for c in range(1, 23)]
    sample_ids  = load_sample_ids_and_validate(
        input_dir=args.input_dir,
        chromosomes=chromosomes,
    )

    cell_type_files = sorted([f for f in os.listdir(args.model_dir) if f.endswith(".json")])
    LOGGER.info("Found %d cell type model files", len(cell_type_files))

    num_slots            = args.num_slots if args.num_slots is not None else max(2, args.n_producers + 1)
    geno_capacity_floats = mb_to_float32_count(args.slot_geno_capacity_mb)
    coef_capacity_floats = mb_to_float32_count(args.slot_coef_capacity_mb)

    LOGGER.info(
        "Shared-memory configuration: num_slots=%d, geno_capacity=%d floats (~%d MB/slot), coef_capacity=%d floats (~%d MB/slot)",
        num_slots,
        geno_capacity_floats,
        args.slot_geno_capacity_mb,
        coef_capacity_floats,
        args.slot_coef_capacity_mb,
    )

    for ct_file in tqdm(cell_type_files, desc="Processing cell types"):
        ct_name = Path(ct_file).stem
        LOGGER.info("Processing cell type: %s", ct_name)

        with open(args.model_dir / ct_file, "r") as f:
            model = json.load(f)
        
        out_path = args.output_dir / f"{ct_name}.predictions.tsv.gz"
        LOGGER.info("Streaming predictions to %s", out_path)

        start = time.time()

        predict_cell_type_distributed(
            model=model,
            input_dir=args.input_dir,
            sample_ids=sample_ids,
            out_path=out_path,
            n_producers=args.n_producers,
            batch_size=args.batch_size,
            num_slots=num_slots,
            max_genes_per_slot=args.max_genes_per_slot,
            geno_capacity_floats=geno_capacity_floats,
            coef_capacity_floats=coef_capacity_floats,
            use_gpu=args.gpu,
            log_dir=args.log_dir,
            log_level=log_level,
        )

        end = time.time()
        elapsed = end - start
        LOGGER.info("Finished predictions for %s in %.2f seconds", ct_name, elapsed)
        LOGGER.info("Saved predictions for %s to %s", ct_name, out_path)

    LOGGER.info("Done.")


if __name__ == "__main__":
    main()