import os
import torch
import argparse
import numpy as np
from tqdm import tqdm
from pathlib import Path

from training.models.mlp import MLPPredictor

"""
get_student_data.py

We are taking Geuvadis Enformer embeddings prodived by scPrediXcan, which we
preproecessed via 'preprocess_teacher_data.py', as well as a mapping from
the teacher model output to the cell-type index. We perform a forward pass 
through the teacher model to get the target variables for the student model.
The output is a directory with one csv file per cell-type, which look like this:

gene (ensid) | chrom | tss | individual 1 | individual 2 | ... | individual N
"""

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare target variables for downstream analysis."
    )
    parser.add_argument("-i", "--input-dir",     type=Path, required=True)
    parser.add_argument("-o", "--output-dir",    type=Path, required=True)
    parser.add_argument("-t", "--teacher-model", type=Path, required=True)
    parser.add_argument("-c", "--ct-mapping",    type=Path, required=True)
    parser.add_argument("-b", "--batch-size",    type=int, default=1)
    return parser.parse_args()


def load_model(teacher_model_path: Path):
    checkpoint = torch.load(teacher_model_path, map_location=device)

    model = MLPPredictor(
        input_dim=checkpoint["input_dim"],
        n_layers=checkpoint["n_layers"],
        output_dim=checkpoint["output_dim"],
        layer_norm=checkpoint["layer_norm"]
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    return model


# we need a uniquely identifying key for each gene: ensid + chrom + tss
def make_keys(ensids, chroms, tss):
    ens  = ensids.astype(str)
    chr_ = chroms.astype(str)
    t    = tss.astype(np.int64)
    return list(zip(ens, chr_, t)) # list of tuples is easiest to reason about


def main() -> None:
    args = parse_args()

    # load some data first...
    idx2ct  = np.load(args.ct_mapping)
    model   = load_model(args.teacher_model)
    persons = os.listdir(args.input_dir)

    # first we build a master index of genes (key triple) across all persons
    all_keys = []
    key_set = set()

    for person in tqdm(persons, desc="Indexing genes (metadata)"):
        pdir   = args.input_dir / person
        chroms = np.load(pdir / "chroms.npy")
        ensids = np.load(pdir / "ensids.npy")
        tss    = np.load(pdir / "tss.npy")

        keys = make_keys(ensids, chroms, tss)
        for k in keys:
            if k not in key_set:
                key_set.add(k)
                all_keys.append(k)

    num_rows = len(all_keys)
    key_to_row = {k: i for i, k in enumerate(all_keys)}

    # master keys -> column vectors for output CSVs
    master_ensids = np.array([k[0] for k in all_keys], dtype=object)
    master_chroms = np.array([k[1] for k in all_keys], dtype=object)
    master_tss    = np.array([k[2] for k in all_keys], dtype=np.int64)

    # preallocate gene x person matrices for each cell-type
    ct_to_matrix = {
        ct: np.full((num_rows, len(persons)), np.nan, dtype=np.float32)
        for ct in idx2ct
    }

    # now perform inference for each person, place into correct row in each matrix based on key
    for person_idx, person in enumerate(persons):
        pdir   = args.input_dir / person
        chroms = np.load(pdir / "chroms.npy")
        ensids = np.load(pdir / "ensids.npy")
        tss    = np.load(pdir / "tss.npy")
        feats  = np.load(pdir / "features.npy")  # (n_person_genes, D)

        keys = make_keys(ensids, chroms, tss)
        n_person_genes = feats.shape[0]

        # batched inference
        preds = np.empty((n_person_genes, len(idx2ct)), dtype=np.float32)
        with torch.no_grad():
            for start in tqdm(range(0, n_person_genes, args.batch_size), desc=f"Predicting {person}"):
                end = min(start + args.batch_size, n_person_genes)
                x = torch.from_numpy(feats[start:end]).float().to(device)
                y = model(x)  # (B, num_ct)
                preds[start:end] = y.detach().cpu().numpy()

        # place into matrices using master row index
        for i, k in enumerate(keys):
            row = key_to_row.get(k)
            for ct_idx, ct in enumerate(idx2ct): # fill all cell-types for this gene and person
                ct_to_matrix[ct][row, person_idx] = preds[i, ct_idx]

    # now write one CSV per cell-type
    print(f"Writing output CSVs to {args.output_dir}...")
    os.makedirs(args.output_dir, exist_ok=True)
    header = "gene,chrom,tss," + ",".join(persons) + "\n"

    for ct in idx2ct:
        out_path = args.output_dir / f"{ct}.csv"
        M = ct_to_matrix[ct]  # (num_rows, num_persons)
        with open(out_path, "w") as f:
            f.write(header)
            for r in range(num_rows):
                values = ",".join(map(str, M[r, :].tolist()))
                f.write(f"{master_ensids[r]},{master_chroms[r]},{master_tss[r]},{values}\n")

    print("Done!")

if __name__ == "__main__":
    main()