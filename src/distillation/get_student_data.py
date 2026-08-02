import os
import torch
import argparse
import numpy as np
from tqdm import tqdm
from pathlib import Path

from src.training.models.mlp import MLPPredictor
from src.training.models.mlp_sep import (
    MLPPredictor as SeparateMLPPredictor,
)
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
output_dir/totvar/*.csv      (total predictive variance: aleatoric + epistemic)
"""

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def numpy_checkpoint_safe_globals() -> list:
    """Return the NumPy scalar types used by legacy training metadata."""
    numpy_core = getattr(np, "_core", None)
    if numpy_core is None:
        numpy_core = np.core

    safe_globals = [
        numpy_core.multiarray.scalar,
        np.dtype,
    ]
    scalar_types = (
        np.bool_,
        np.int8,
        np.int16,
        np.int32,
        np.int64,
        np.uint8,
        np.uint16,
        np.uint32,
        np.uint64,
        np.float16,
        np.float32,
        np.float64,
        np.complex64,
        np.complex128,
        np.str_,
        np.bytes_,
    )
    safe_globals.extend(type(np.dtype(scalar_type)) for scalar_type in scalar_types)
    return list(dict.fromkeys(safe_globals))


def load_teacher_checkpoint(teacher_model_path: Path) -> dict:
    """Load a project checkpoint without enabling unrestricted pickle."""
    with torch.serialization.safe_globals(numpy_checkpoint_safe_globals()):
        checkpoint = torch.load(
            teacher_model_path,
            map_location=device,
            weights_only=True,
        )
    if not isinstance(checkpoint, dict):
        raise TypeError(
            "Expected the teacher checkpoint to contain a dictionary, got "
            f"{type(checkpoint).__name__}."
        )
    return checkpoint


def resolve_checkpoint_setting(
    checkpoint: dict,
    key: str,
    requested_value: str | None,
    legacy_default: str,
) -> str:
    """Resolve a CLI setting against self-describing checkpoint metadata."""
    checkpoint_value = checkpoint.get(key)
    if checkpoint_value is None:
        return requested_value if requested_value is not None else legacy_default

    checkpoint_value = str(checkpoint_value)
    if requested_value is not None and requested_value != checkpoint_value:
        raise ValueError(
            f"Requested {key}={requested_value!r}, but the checkpoint records "
            f"{key}={checkpoint_value!r}."
        )
    return checkpoint_value


def resolve_cell_types(
    checkpoint: dict,
    ct_mapping_path: Path | None,
) -> np.ndarray:
    """Load cell types from the checkpoint and optionally validate a mapping."""
    checkpoint_cell_types = checkpoint.get("cell_types")
    checkpoint_cell_types = (
        [str(cell_type) for cell_type in checkpoint_cell_types]
        if checkpoint_cell_types is not None
        else None
    )
    mapping_cell_types = (
        [
            str(cell_type)
            for cell_type in np.load(ct_mapping_path, allow_pickle=True)
        ]
        if ct_mapping_path is not None
        else None
    )

    if checkpoint_cell_types is None and mapping_cell_types is None:
        raise ValueError(
            "The checkpoint has no cell-type metadata. Provide --ct-mapping "
            "when using a legacy checkpoint."
        )
    if (
        checkpoint_cell_types is not None
        and mapping_cell_types is not None
        and checkpoint_cell_types != mapping_cell_types
    ):
        raise ValueError(
            "The ordered cell types in --ct-mapping do not match the "
            f"checkpoint: mapping={mapping_cell_types}, "
            f"checkpoint={checkpoint_cell_types}."
        )

    cell_types = (
        checkpoint_cell_types
        if checkpoint_cell_types is not None
        else mapping_cell_types
    )
    output_dim = int(checkpoint["output_dim"])
    if len(cell_types) != output_dim:
        raise ValueError(
            f"Resolved {len(cell_types)} cell types, but the checkpoint has "
            f"output_dim={output_dim}."
        )
    return np.asarray(cell_types, dtype=object)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare target variables for downstream analysis."
    )
    parser.add_argument("-i", "--input-dir",     type=Path, required=True)
    parser.add_argument("-o", "--output-dir",    type=Path, required=True)
    parser.add_argument("-t", "--teacher-model", type=Path, required=True)
    parser.add_argument(
        "-c", "--ct-mapping",
        type=Path,
        default=None,
        help=(
            "Optional cell-type mapping. New checkpoints contain this "
            "metadata; when supplied, the mapping is validated against it. "
            "Required for legacy checkpoints without cell_types."
        ),
    )
    parser.add_argument("-b", "--batch-size",    type=int, default=1)
    parser.add_argument("-m", "--model-name",
                        type=str, default=None,
                        choices=["mlp", "mlp-sep", "deep-ensemble"],
                        help=(
                            "Optional teacher-model type. Inferred from new "
                            "checkpoints and validated when supplied. Legacy "
                            "checkpoints default to 'mlp'."
                        ))
    parser.add_argument("-nt", "--norm-targets", 
                        type=str, default=None,
                        choices=["none", "log", "percentiles"], 
                        help=(
                            "Optional target normalization. Inferred from new "
                            "checkpoints and validated when supplied. Legacy "
                            "checkpoints default to 'none'."
                        ))
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


def load_model(
    teacher_model_path: Path,
    model_name: str | None = None,
    *,
    checkpoint: dict | None = None,
):
    if checkpoint is None:
        checkpoint = load_teacher_checkpoint(teacher_model_path)
    model_name = resolve_checkpoint_setting(
        checkpoint,
        key="model_name",
        requested_value=model_name,
        legacy_default="mlp",
    )

    if model_name == "mlp":
        model = MLPPredictor(
            input_dim=checkpoint["input_dim"],
            n_layers=checkpoint["n_layers"],
            output_dim=checkpoint["output_dim"],
            layer_norm=checkpoint["layer_norm"],
            dropout=checkpoint.get("dropout", 0.0),
        )
    elif model_name == "mlp-sep":
        model = SeparateMLPPredictor(
            input_dim=checkpoint["input_dim"],
            n_layers=checkpoint["n_layers"],
            output_dim=checkpoint["output_dim"],
            layer_norm=checkpoint["layer_norm"],
            dropout=checkpoint.get("dropout", 0.0),
        )
    elif model_name == "deep-ensemble":
        model = MLPEnsemble(
            n_models=infer_n_models(checkpoint["model_state_dict"]),
            input_dim=checkpoint["input_dim"],
            n_layers=checkpoint["n_layers"],
            output_dim=checkpoint["output_dim"],
            layer_norm=checkpoint["layer_norm"],
            dropout=checkpoint.get("dropout", 0.0),
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
    checkpoint = load_teacher_checkpoint(args.teacher_model)
    model_name = resolve_checkpoint_setting(
        checkpoint,
        key="model_name",
        requested_value=args.model_name,
        legacy_default="mlp",
    )
    norm_targets = resolve_checkpoint_setting(
        checkpoint,
        key="norm_targets",
        requested_value=args.norm_targets,
        legacy_default="none",
    )
    if norm_targets not in {"none", "log", "percentiles"}:
        raise ValueError(
            f"Checkpoint records unsupported norm_targets={norm_targets!r}."
        )

    idx2ct = resolve_cell_types(checkpoint, args.ct_mapping)
    model = load_model(
        args.teacher_model,
        model_name,
        checkpoint=checkpoint,
    )
    persons = os.listdir(args.input_dir)

    if norm_targets == "percentiles":
        raise NotImplementedError("Undoing percentile normalization is not supported, since it is not a bijective transformation. Please set --norm-targets to 'none' or 'log' when running this script.")

    # first we build a master index of genes (key triple) across all persons
    all_keys, key_to_row = build_master_index(args.input_dir, persons)
    num_rows = len(all_keys)

    # master keys -> column vectors for output CSVs
    master_ensids = np.array([k[0] for k in all_keys], dtype=object)
    master_chroms = np.array([k[1] for k in all_keys], dtype=object)
    master_tss    = np.array([k[2] for k in all_keys], dtype=np.int64)

    is_ensemble = model_name == "deep-ensemble"

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
    ct_to_totvar = empty_ct_matrices() if is_ensemble else None

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
                    ct_to_totvar[ct][row, person_idx] = (
                        aleatoric[i, ct_idx] + epistemic[i, ct_idx]
                    )

    if norm_targets == "log":
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
        write_csvs(args.output_dir / "totvar", ct_to_totvar, idx2ct, persons,
                   master_ensids, master_chroms, master_tss)
    else:
        print(f"Writing output CSVs to {args.output_dir}...")
        write_csvs(args.output_dir, ct_to_pred, idx2ct, persons,
                   master_ensids, master_chroms, master_tss)

    print("Done!")

if __name__ == "__main__":
    main()
