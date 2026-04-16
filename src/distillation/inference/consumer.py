from __future__ import annotations

import logging
import multiprocessing as mp
from multiprocessing.shared_memory import SharedMemory
from pathlib import Path
from queue import Empty
from typing import Any

import numpy as np
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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


def _predict_active_batch_cpu(active_items: list[dict[str, Any]]) -> dict[str, np.ndarray]:
    results: dict[str, np.ndarray] = {}
    for item in active_items:
        pred = float(item["intercept"]) + (item["G"] @ item["coefs"])
        results[item["gene"]] = np.asarray(pred, dtype=np.float32)
    return results


def _predict_active_batch_gpu(active_items: list[dict[str, Any]], n_samples: int) -> dict[str, np.ndarray]:
    if not active_items:
        raise ValueError("No active items to predict")

    max_snps   = max(int(item["coefs"].shape[0]) for item in active_items)
    batch_size = len(active_items)

    geno_tensor = torch.zeros(
        (batch_size, n_samples, max_snps),
        dtype=torch.float32,
        device=device,
    )
    coef_tensor = torch.zeros(
        (batch_size, max_snps),
        dtype=torch.float32,
        device=device,
    )
    intercepts = torch.zeros(
        (batch_size, 1),
        dtype=torch.float32,
        device=device,
    )

    genes: list[str] = []

    for i, item in enumerate(active_items):
        G     = item["G"]
        coefs = item["coefs"]
        k     = int(coefs.shape[0])

        geno_tensor[i, :, :k] = torch.as_tensor(G, dtype=torch.float32, device=device)
        coef_tensor[i, :k]    = torch.as_tensor(coefs, dtype=torch.float32, device=device)
        intercepts[i, 0]      = float(item["intercept"])
        genes.append(item["gene"])

    preds    = intercepts + torch.einsum("bnk,bk->bn", geno_tensor, coef_tensor)
    preds_np = preds.detach().cpu().numpy().astype(np.float32, copy=False)

    results: dict[str, np.ndarray] = {}
    for gene, pred in zip(genes, preds_np, strict=True):
        results[gene] = pred

    return results


def _build_gene_views_from_descriptor(descriptor: dict[str, Any], geno_views, coef_views, n_samples):
    """ Return active items (i.e. gene payloads with G/coefs views) + ready results (intercept-only predictions). """
    slot_id   = int(descriptor["slot_id"])
    geno_flat = geno_views[slot_id]
    coef_flat = coef_views[slot_id]

    gene_names: list[str]   = descriptor["gene_names"]
    intercepts: list[float] = descriptor["intercepts"]
    lengths: list[int]      = descriptor["lengths"]
    geno_offsets: list[int] = descriptor["geno_offsets"]
    coef_offsets: list[int] = descriptor["coef_offsets"]

    active_items: list[dict[str, Any]]   = []
    ready_results: dict[str, np.ndarray] = {}

    for gene, intercept, length, geno_offset, coef_offset in zip(
        gene_names,
        intercepts,
        lengths,
        geno_offsets,
        coef_offsets,
        strict=True,
    ):
        length = int(length)
        if length == 0: # no SNPs for this gene, so intercept-only prediction ready to go
            ready_results[gene] = np.full(n_samples, float(intercept), dtype=np.float32)
            continue

        geno_start = int(geno_offset)
        geno_end   = geno_start + (n_samples * length)
        coef_start = int(coef_offset)
        coef_end   = coef_start + length

        G     = geno_flat[geno_start:geno_end].reshape(n_samples, length)
        coefs = coef_flat[coef_start:coef_end]

        active_items.append(
            {
                "gene": gene,
                "intercept": float(intercept),
                "G": G,
                "coefs": coefs,
            }
        )

    return active_items, ready_results


def consumer_main(
    ready_queue: mp.Queue,
    free_slot_queue: mp.Queue,
    result_queue: mp.Queue,
    n_producers: int,
    batch_size: int,
    n_samples: int,
    n_genes: int,
    use_gpu: bool,
    geno_shm_names: list[str],
    coef_shm_names: list[str],
    geno_capacity_floats: int,
    coef_capacity_floats: int,
    log_dir: str | None = None,
    log_level: int = logging.INFO,
) -> None:
    if log_dir is not None:
        setup_worker_logging(Path(log_dir) / "consumer.log", level=log_level)

    producer_done = 0
    predictions: dict[str, np.ndarray] = {}

    geno_shms: list[SharedMemory] = []
    coef_shms: list[SharedMemory] = []

    gpu_enabled = bool(use_gpu and torch.cuda.is_available())
    LOGGER.info("Consumer started")
    LOGGER.info("Consumer using %s", f"GPU device {device}" if gpu_enabled else "CPU")

    total_genes_processed = 0

    try:
        geno_shms, geno_views, coef_shms, coef_views = _open_shared_slot_buffers(
            geno_shm_names=geno_shm_names,
            coef_shm_names=coef_shm_names,
            geno_capacity_floats=geno_capacity_floats,
            coef_capacity_floats=coef_capacity_floats,
        )

        while True:
            descriptors: list[dict[str, Any]] = []
            num_collected_genes = 0

            while num_collected_genes < batch_size:
                if producer_done >= n_producers:
                    try:
                        item = ready_queue.get_nowait()
                    except Empty:
                        break
                else:
                    try:
                        item = ready_queue.get(timeout=1.0)
                    except Empty:
                        continue

                if item.get("type") == "producer_done":
                    producer_done += 1
                    LOGGER.info(
                        "Received completion from producer %s (%d/%d)",
                        item.get("producer_id"),
                        producer_done,
                        n_producers,
                    )
                    continue

                if item.get("type") == "error":
                    raise RuntimeError(f"Producer {item.get('producer_id')} failed: {item.get('message')}")
                
                descriptors.append(item)
                num_collected_genes += len(item.get("gene_names", []))

                LOGGER.debug(
                    "Received descriptor for slot %s with %d genes (producer %s)",
                    item.get("slot_id"),
                    len(item.get("gene_names", [])),
                    item.get("producer_id"),
                )

            if descriptors:
                active_items: list[dict[str, Any]]       = []
                immediate_results: dict[str, np.ndarray] = {}

                for descriptor in descriptors:
                    slot_active, slot_immediate = _build_gene_views_from_descriptor(
                        descriptor=descriptor,
                        geno_views=geno_views,
                        coef_views=coef_views,
                        n_samples=n_samples,
                    )
                    active_items.extend(slot_active)
                    immediate_results.update(slot_immediate)

                predictions.update(immediate_results)

                if active_items:
                    LOGGER.debug(
                        "Consumer processing %d shared-slot descriptors containing %d active genes",
                        len(descriptors),
                        len(active_items),
                    )
                    if gpu_enabled:
                        batch_preds = _predict_active_batch_gpu(
                            active_items=active_items,
                            n_samples=n_samples
                        )
                    else:
                        batch_preds = _predict_active_batch_cpu(active_items=active_items)
                    predictions.update(batch_preds)

                    total_genes_processed += len(active_items)
                    LOGGER.info(
                        "UPDATE: Total genes processed: %d / %d)",
                        total_genes_processed,
                        n_genes
                    )

                for descriptor in descriptors:
                    free_slot_queue.put(int(descriptor["slot_id"]))

            if producer_done >= n_producers and not descriptors:
                break

        LOGGER.info("Consumer finished successfully with %d predicted genes", len(predictions))
        result_queue.put({"type": "predictions", "predictions": predictions})

    except Exception as exc:
        LOGGER.exception("Consumer failed")
        result_queue.put({"type": "error", "message": repr(exc)})
        raise

    finally:
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