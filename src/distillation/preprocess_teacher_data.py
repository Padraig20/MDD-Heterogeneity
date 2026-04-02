import os
import h5py
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path

"""
preprocess_teacher_data.py

We are taking Geuvadis Enformer embeddings prodived by scPrediXcan. We process
these embeddings to get them in a format suitable for our pipeline to generate
a dataset for the student model. The original data can be found here:

https://github.com/hakyimlab/scPrediXcan/blob/master/Scripts/Enformer_epigenomic_features/Geuvadis_individuals_epigenome.txt
"""

def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Prepare target variables for downstream analysis."
    )
    parser.add_argument(
        "-i", "--input-dir",
        type=Path,
        required=True,
        help="Path to input directory."
    )
    parser.add_argument(
        "-o", "--output-dir",
        type=Path,
        required=True,
        help="Path to write output directory."
    )
    return parser.parse_args()

def preprocess(input_dir, person_name, md, output_dir):
    target_dir = os.path.join(output_dir, person_name)
    # already exists? skip
    if os.path.exists(target_dir):
        print(f"Directory {target_dir} already exists, skipping {person_name}...")
        return

    os.makedirs(target_dir, exist_ok=True)

    with h5py.File(os.path.join(input_dir, f"{person_name}.h5"), "r") as f:
        # for one person large enough to fit in memory
        ensids   = []
        chroms   = []
        features = []
        tss      = []
        for key in tqdm(f.keys(), total=len(f.keys()), desc=f"{person_name}", unit="row"):
            chr = key.split("_")[0]
            # we could have multiple TSSs for the same gene!
            ensid_list = md["ensembl_gene_id"][md["TSS_enformer_input"] == key[:-len("_predictions")]]
            tss_list = md["transcription_start_site"][md["TSS_enformer_input"] == key[:-len("_predictions")]]
            feat = f[key][:]          # (4, 5313) -> (5313,)
            feat = feat.mean(axis=0)  # average over 4 bins

            for idx, ensid in enumerate(ensid_list):
                ensids.append(ensid)
                chroms.append(chr)
                features.append(feat)
                tss.append(tss_list.iloc[idx])

        np.save(os.path.join(target_dir, "features.npy"), features)
        np.save(os.path.join(target_dir, "ensids.npy"), ensids)
        np.save(os.path.join(target_dir, "chroms.npy"), chroms)
        np.save(os.path.join(target_dir, "tss.npy"), tss)

def main() -> None:
    args = parse_args()

    md = pd.read_csv(os.path.join(args.input_dir, "metadata.csv"))

    for file in os.listdir(args.input_dir):
        if file.endswith(".h5"):
            person_name = file[:-3]  # remove .h5 extension
            preprocess(args.input_dir, person_name, md, args.output_dir)

    print("Done!")

if __name__ == "__main__":
    main()