from __future__ import annotations
import argparse
import logging
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import numpy as np
from numpy.lib.format import open_memmap

import torch
from enformer_pytorch import Enformer

"""
get_feats_from_seqs.py

Script that takes input from the human reference genome sequences we extracted
earlier (around the TSS) and then extracts features from these sequences using
some sort of pLM/gLM model. Here, we use e.g. Enformer via Hugging Face

https://huggingface.co/EleutherAI/enformer-official-rough
"""

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
mapping = {'A': 0, 'a': 0, 'C': 1, 'c': 1, 'G': 2, 'g': 2, 'T': 3, 't': 3, 'N': 4, 'n': 4} # for ACGTN, in that order (-1 for padding)

def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Prepare target variables for downstream analysis."
    )
    parser.add_argument(
        "-i", "--input",
        type=Path,
        required=True,
        help="Path to input file (*.csv)."
    )
    parser.add_argument(
        "-o", "--output",
        type=Path,
        required=True,
        help="Path to write output file (just name, *.npy suffix will be added)."
    )
    parser.add_argument(
        "-m", "--model-name",
        type=str,
        default="enformer",
        choices=["enformer"],
        help="Name of the model to use for feature extraction. List of available models will be expanded in the future."
    )
    parser.add_argument(
        "-b", "--batch-size",
        type=int,
        default=1,
        help="Batch size for feature extraction."
    )
    parser.add_argument(
        "-w", "--window-size",
        type=int,
        default=4,
        help="Window size for feature extraction, i.e. number of bins."
    )
    parser.add_argument(
        "-v", "--verbose",
        action="count",
        default=0,
        help="Increase verbosity."
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

def load_data(input_path: Path) -> pd.DataFrame:
    """Load data from `input_path`."""
    logging.info("Loading data from %s", input_path)
    return pd.read_csv(input_path)

def dna_seq_to_tensor(seq: str) -> torch.Tensor:
    """Convert a DNA sequence string to an integer-encoded tensor (1D)."""
    ids = [mapping[nuc] for nuc in seq]
    return torch.tensor(ids, dtype=torch.long, device=device)  # (L,)

def get_features(data: pd.DataFrame, model_name: str, batch_size: int, window_size: int, output_path: Path) -> pd.DataFrame:
    """Extract features from sequences using specified model."""
    logging.info("Extracting features using model: %s", model_name)
    if model_name == "enformer":
        model = Enformer.from_pretrained("EleutherAI/enformer-official-rough", dtype="auto").to(device)
        model.eval()

        feats_mm_path = output_path.with_suffix(".features.npy")
        ensids_mm_path = output_path.with_suffix(".ensids.npy")

        num_rows = len(data)
        logging.info("Number of sequences to process: %d", num_rows)

        # create memmaps to immediately flush to disk
        logging.debug("Creating memmap arrays at %s and %s", feats_mm_path, ensids_mm_path)

        # pre-allocate
        feats_mm = open_memmap(
            feats_mm_path,
            mode="w+",
            dtype=np.float32,
            shape=(num_rows, 5313)
        )

        # don't need a memmap, small enough to fit in RAM
        ensids = np.empty(num_rows, dtype=object)
        chroms = np.empty(num_rows, dtype=object)

        for start_idx in tqdm(range(0, num_rows, batch_size),
                              total=(num_rows + batch_size - 1) // batch_size,
                              desc="Extracting features", unit="batch"):

            end_idx = min(start_idx + batch_size, num_rows)
            batch   = data.iloc[start_idx:end_idx]

            batch_seqs   = [dna_seq_to_tensor(seq) for seq in batch["sequence"]]
            batch_tensor = torch.stack(batch_seqs, dim=0)  # (B, L)

            with torch.no_grad():
                output = model(batch_tensor)['human']            # (B, 896, 5313)
            features = output.cpu().numpy().astype(np.float32)   # (B, 896, 5313)

            # select central bin, average over window size
            central_bin = features.shape[1] // 2
            window_size_half = window_size // 2
            features = features[:, central_bin-window_size_half:central_bin+window_size_half, :]  # (B, W, 5313)
            features = features.mean(axis=1)  # (B, 5313)

            feats_mm[start_idx:end_idx, :] = features
            ensids[start_idx:end_idx] = batch["ensid"].to_numpy()
            chroms[start_idx:end_idx] = batch["chrom"].to_numpy()

        logging.info("Features successfully extracted.")

        logging.debug("Sample of extracted features:\n%s", feats_mm[0])
        logging.debug("Features saved to %s, now flushing...", feats_mm_path)

        del feats_mm
        
        np.save(output_path.with_suffix(".ensids.npy"), ensids)
        np.save(output_path.with_suffix(".chroms.npy"), chroms)
 
    else:
        raise ValueError(f"Model {model_name} not supported.")

def main() -> None:
    args = parse_args()
    setup_logging(args.verbose)
    logging.debug("Arguments: %s", args)

    data = load_data(args.input)
    get_features(data, args.model_name, args.batch_size, args.window_size, args.output)

    logging.info("Done.")

if __name__ == "__main__":
    main()