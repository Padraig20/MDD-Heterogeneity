import os
import argparse
import numpy as np
from tqdm import tqdm
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare per-sample teacher-model input files."
    )
    parser.add_argument(
        "-i", "--input-prefix",
        type=Path,
        required=True,
        help="Prefix of input files, i.e. X.chroms.npy, X.ensids.npy, etc."
    )
    parser.add_argument(
        "-o", "--output-dir",
        type=Path,
        required=True,
        help="Path to write output directory."
    )
    return parser.parse_args()


def preprocess(chroms, ensids, tss, sample_ids, features, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Sort rows by sample_id so rows for the same sample become contiguous
    order = np.argsort(sample_ids, kind="stable")
    sorted_sample_ids = sample_ids[order]

    # Find boundaries between groups
    unique_sample_ids, start_idx, counts = np.unique(
        sorted_sample_ids, return_index=True, return_counts=True
    )

    for sample_id, start, count in tqdm(
        zip(unique_sample_ids, start_idx, counts),
        total=len(unique_sample_ids),
        desc="Processing samples"
    ):
        idx = order[start:start + count]

        sample_dir = output_dir / sample_id
        if sample_dir.exists():
            print(f"Directory {sample_dir} already exists. Skipping sample {sample_id}.")
            continue
        sample_dir.mkdir(parents=True, exist_ok=False)

        np.save(sample_dir / "features.npy", features[idx])
        np.save(sample_dir / "ensids.npy", ensids[idx])
        np.save(sample_dir / "chroms.npy", chroms[idx])
        np.save(sample_dir / "tss.npy", tss[idx])


def main() -> None:
    args = parse_args()

    chroms = np.load(f"{args.input_prefix}.chroms.npy", allow_pickle=True)
    ensids = np.load(f"{args.input_prefix}.ensids.npy", allow_pickle=True)
    sample_ids = np.load(f"{args.input_prefix}.sample_ids.npy", allow_pickle=True)
    tss = np.load(f"{args.input_prefix}.tss.npy", allow_pickle=True)

    # Memory-mapped read for large array
    features = np.load(
        f"{args.input_prefix}.features.npy",
        allow_pickle=True,
        mmap_mode="r"
    )

    preprocess(chroms, ensids, tss, sample_ids, features, args.output_dir)
    print("Done!")


if __name__ == "__main__":
    main()