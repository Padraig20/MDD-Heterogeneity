import os
import torch
import argparse
import numpy as np
from tqdm import tqdm
from pathlib import Path

from src.training.models.mlp import MLPPredictor
from src.training.models.mlp_deep_ensemble import MLPEnsemble

"""
get_student_data.py

We are taking Geuvadis Enformer embeddings prodived by scPrediXcan, which we
preproecessed via 'preprocess_teacher_data.py', as well as a mapping from
the teacher model output to the cell-type index. We perform a forward pass 
through the teacher model to get the target variables for the student model.

For the single MLP teacher, the output is a directory with one csv file per
cell-type, which look like this:

gene (ensid) | chrom | tss | individual 1 | individual 2 | ... | individual N

For the deep ensemble teacher, we additionally obtain aleatoric and epistemic
uncertainty estimates, so we write three such csv files per cell-type, split
into subdirectories:

output_dir/preds/*.csv       (predicted mean expression)
output_dir/aleatoric/*.csv   (aleatoric uncertainty / data noise)
output_dir/epistemic/*.csv   (epistemic uncertainty / model uncertainty)
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
    parser.add_argument("-m", "--model-name",
                        type=str, default="mlp",
                        choices=["mlp", "deep-ensemble"],
                        help="Type of teacher model to load.")
    parser.add_argument("-nt", "--norm-targets", 
                        type=str, default="none", 
                        choices=["none", "log", "percentiles"], 
                        help="Normalization method used for target labels. Will undo normalization.")
    return parser.parse_args()


def infer_n_models(state_dict) -> int:
    """Infer the number of ensemble members from the saved state dict keys
    (e.g. 'models.0.mlp...', 'models.1.mlp...')."""
    indices = set()
    for key in state_dict:
        if key.startswith("models."):
            indices.add(int(key.split(".")[1]))
    if not indices:
        raise ValueError("Could not infer number of ensemble members from checkpoint.")
    return max(indices) + 1


def load_model(teacher_model_path: Path, model_name: str):
    checkpoint = torch.load(teacher_model_path, map_location=device)

    if model_name == "mlp":
        model = MLPPredictor(
            input_dim=checkpoint["input_dim"],
            n_layers=checkpoint["n_layers"],
            output_dim=checkpoint["output_dim"],
            layer_norm=checkpoint["layer_norm"]
        )
    elif model_name == "deep-ensemble":
        model = MLPEnsemble(
            n_models=infer_n_models(checkpoint["model_state_dict"]),
            input_dim=checkpoint["input_dim"],
            n_layers=checkpoint["n_layers"],
            output_dim=checkpoint["output_dim"],
            layer_norm=checkpoint["layer_norm"]
        )
    else:
        raise ValueError(f"Model {model_name} is not supported.")

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


def build_master_index(input_dir: Path, persons: list[str]):
    """Build a master index of genes (key triple) across all persons."""
    all_keys = []
    key_set = set()

    for person in tqdm(persons, desc="Indexing genes (metadata)"):
        pdir   = input_dir / person
        chroms = np.load(pdir / "chroms.npy", allow_pickle=True)
        ensids = np.load(pdir / "ensids.npy", allow_pickle=True)
        tss    = np.load(pdir / "tss.npy", allow_pickle=True)

        keys = make_keys(ensids, chroms, tss)
        for k in keys:
            if k not in key_set:
                key_set.add(k)
                all_keys.append(k)

    key_to_row = {k: i for i, k in enumerate(all_keys)}
    return all_keys, key_to_row


def write_csvs(out_dir: Path, ct_to_matrix, idx2ct, persons,
               master_ensids, master_chroms, master_tss):
    """Write one CSV per cell-type into out_dir."""
    os.makedirs(out_dir, exist_ok=True)
    header = "gene,chrom,tss," + ",".join(persons) + "\n"
    num_rows = len(master_ensids)

    for ct in idx2ct:
        out_path = out_dir / f"{ct}.csv"
        M = ct_to_matrix[ct]  # (num_rows, num_persons)
        with open(out_path, "w") as f:
            f.write(header)
            for r in range(num_rows):
                values = ",".join(map(str, M[r, :].tolist()))
                f.write(f"{master_ensids[r]},{master_chroms[r]},{master_tss[r]},{values}\n")


def undo_log(ct_to_matrix):
    return {ct: np.expm1(M) for ct, M in ct_to_matrix.items()}


def main() -> None:
    args = parse_args()

    # load some data first...
    idx2ct  = np.load(args.ct_mapping, allow_pickle=True)
    model   = load_model(args.teacher_model, args.model_name)
    persons = os.listdir(args.input_dir)

    if args.norm_targets == "percentiles":
        raise NotImplementedError("Undoing percentile normalization is not supported, since it is not a bijective transformation. Please set --norm-targets to 'none' or 'log' when running this script.")

    # first we build a master index of genes (key triple) across all persons
    all_keys, key_to_row = build_master_index(args.input_dir, persons)
    num_rows = len(all_keys)

    # master keys -> column vectors for output CSVs
    master_ensids = np.array([k[0] for k in all_keys], dtype=object)
    master_chroms = np.array([k[1] for k in all_keys], dtype=object)
    master_tss    = np.array([k[2] for k in all_keys], dtype=np.int64)

    is_ensemble = args.model_name == "deep-ensemble"

    # preallocate gene x person matrices for each cell-type. For the ensemble we
    # keep separate matrices for predictions and the two uncertainty estimates.
    def empty_ct_matrices():
        return {
            ct: np.full((num_rows, len(persons)), np.nan, dtype=np.float32)
            for ct in idx2ct
        }

    ct_to_pred = empty_ct_matrices()
    ct_to_aleatoric = empty_ct_matrices() if is_ensemble else None
    ct_to_epistemic = empty_ct_matrices() if is_ensemble else None

    # now perform inference for each person, place into correct row in each matrix based on key
    for person_idx, person in enumerate(persons):
        pdir   = args.input_dir / person
        chroms = np.load(pdir / "chroms.npy", allow_pickle=True)
        ensids = np.load(pdir / "ensids.npy", allow_pickle=True)
        tss    = np.load(pdir / "tss.npy", allow_pickle=True)
        feats  = np.load(pdir / "features.npy", allow_pickle=True)  # (n_person_genes, D)

        keys = make_keys(ensids, chroms, tss)
        n_person_genes = feats.shape[0]

        # batched inference
        preds = np.empty((n_person_genes, len(idx2ct)), dtype=np.float32)
        if is_ensemble:
            aleatoric = np.empty((n_person_genes, len(idx2ct)), dtype=np.float32)
            epistemic = np.empty((n_person_genes, len(idx2ct)), dtype=np.float32)

        with torch.no_grad():
            for start in tqdm(range(0, n_person_genes, args.batch_size), desc=f"Predicting {person}"):
                end = min(start + args.batch_size, n_person_genes)
                x = torch.from_numpy(feats[start:end]).float().to(device)
                if is_ensemble:
                    # eval mode returns (prediction, aleatoric_unc, epistemic_unc)
                    pred, ale, epi = model(x)
                    preds[start:end]     = pred.detach().cpu().numpy()
                    aleatoric[start:end] = ale.detach().cpu().numpy()
                    epistemic[start:end] = epi.detach().cpu().numpy()
                else:
                    y = model(x)  # (B, num_ct)
                    preds[start:end] = y.detach().cpu().numpy()

        # place into matrices using master row index
        for i, k in enumerate(keys):
            row = key_to_row.get(k)
            for ct_idx, ct in enumerate(idx2ct): # fill all cell-types for this gene and person
                ct_to_pred[ct][row, person_idx] = preds[i, ct_idx]
                if is_ensemble:
                    ct_to_aleatoric[ct][row, person_idx] = aleatoric[i, ct_idx]
                    ct_to_epistemic[ct][row, person_idx] = epistemic[i, ct_idx]

    if args.norm_targets == "log":
        # undo log normalization on the predicted means. The uncertainties are
        # variances in the (log-)transformed space; expm1 is not a valid inverse
        # for a variance, so we leave them in the model's output space.
        ct_to_pred = undo_log(ct_to_pred)

    # now write output CSVs
    if is_ensemble:
        print(f"Writing ensemble output CSVs to {args.output_dir}...")
        write_csvs(args.output_dir / "preds", ct_to_pred, idx2ct, persons,
                   master_ensids, master_chroms, master_tss)
        write_csvs(args.output_dir / "aleatoric", ct_to_aleatoric, idx2ct, persons,
                   master_ensids, master_chroms, master_tss)
        write_csvs(args.output_dir / "epistemic", ct_to_epistemic, idx2ct, persons,
                   master_ensids, master_chroms, master_tss)
    else:
        print(f"Writing output CSVs to {args.output_dir}...")
        write_csvs(args.output_dir, ct_to_pred, idx2ct, persons,
                   master_ensids, master_chroms, master_tss)

    print("Done!")

if __name__ == "__main__":
    main()
