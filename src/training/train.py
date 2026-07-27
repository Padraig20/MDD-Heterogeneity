from __future__ import annotations
import argparse
import logging
import os
import random
import pandas as pd
import numpy as np
from pathlib import Path

import torch
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR
from src.training.dataset import MddDataset, ReferencePopulationMddDataset
from src.training.utils import get_train_test_dataset, EarlyStopping

from src.training.models.mlp import MLPPredictor
from src.training.models.mlp_sep import MLPPredictor as SeparateMLPPredictor
from src.training.models.mlp_deep_ensemble import MLPEnsemble

from src.training.utils import train_single_model, evaluate_single_model
from src.training.utils import train_separate_model, evaluate_separate_model
from src.training.utils import train_ensemble_model, evaluate_ensemble_model

from src.training.loss.cossim_loss import CosineSimilarityLoss
from src.training.loss.mpc_loss import MPCLoss
from src.training.loss.mse_loss import MSELoss
from src.training.loss.pnll_loss import PNLLLoss
from src.training.loss.gnll_loss import GaussianNLLLoss

from src.training.wandb_logger import WandBLogger

"""
train.py

We train a model to predict cell type-specific gene expression from
epigenetic features using MLP.
"""

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Prepare target variables for downstream analysis."
    )
    parser.add_argument(
        "-X", "--observations",
        type=Path,
        required=True,
        help=(
            "The common name (without extension) of an observation set with sibling "
            "files <name>.features.npy, <name>.ensids.npy, <name>.chroms.npy."
        )
    )
    parser.add_argument(
        "-y", "--targets",
        type=Path,
        required=True,
        help=(
            "Path to the target file. Either a CSV (single-individual targets, rows = "
            "cell types, columns = ENSIDs) or a pseudo-bulk population .h5ad. "
            "Detected automatically from extension."
        )
    )
    parser.add_argument(
        "--cell-type",
        type=str,
        default=None,
        help=(
            "When using a population h5ad target, optionally restrict to a single cell "
            "type label. Otherwise all cell types in obs are used (output_dim = n_cell_types)."
        )
    )
    parser.add_argument(
        "-m", "--model-name",
        type=str,
        default="mlp",
        choices=["mlp", "mlp-sep", "deep-ensemble"],
        help="Name of the model to train."
    )
    parser.add_argument(
        "-b", "--batch-size",
        type=int,
        default=1,
        help="Batch size for training."
    )
    parser.add_argument(
        "-lr", "--learning-rate",
        type=float,
        default=5e-5,
        help="Learning rate for training."
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.05,
        help="Dropout after each hidden activation."
    )
    parser.add_argument(
        "-wd", "--weight-decay",
        type=float,
        default=5e-4,
        help="Adam weight decay."
    )
    parser.add_argument(
        "-e", "--epochs",
        type=int,
        default=10,
        help="Number of training epochs."
    )
    parser.add_argument(
        "-nl", "--n-layers",
        type=int,
        default=1,
        help="Number of hidden layers in the neural network."
    )
    parser.add_argument(
        "-es", "--early-stop",
        action="store_true",
        help="Enable early stopping on eval metric."
    )
    parser.add_argument(
        "-o", "--output",
        type=Path,
        default=None,
        help="Path to save trained model (as .pth)."
    )
    parser.add_argument(
        "-v", "--verbose",
        action="count",
        default=0,
        help="Increase verbosity."
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="Name of the WandB run. Leave empty to not use WandB logging."
    )
    parser.add_argument(
        "--cossim-lambda",
        type=float,
        default=0.0,
        help="Lambda parameter (weight) for the Cosine Similarity loss."
    )
    parser.add_argument(
        "--mpc-lambda",
        type=float,
        default=0.0,
        help="Lambda parameter (weight) for the Mutual Pearson Correlation loss."
    )
    parser.add_argument(
        "--mse-lambda",
        type=float,
        default=0.0,
        help="Lambda parameter (weight) for the Mean Squared Error loss."
    )
    parser.add_argument(
        "--pnll-lambda",
        type=float,
        default=0.0,
        help="Lambda parameter (weight) for the Poisson Negative Log Likelihood loss."
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for model/optimizer initialization and training."
    )
    parser.add_argument(
        "--norm-inputs",
        action="store_true",
        help="Log-transform input features."
    )
    parser.add_argument(
        "--norm-layer",
        action="store_true",
        help="Enable layer normalization on inputs."
    )
    parser.add_argument(
        "-nt", "--norm-targets",
        type=str,
        default="none",
        choices=["none", "log", "percentiles"],
        help="Normalization method for target labels."
    )
    parser.add_argument(
        "-sg", "--select_genes",
        type=Path,
        default=None,
        help="Path to a file containing a list of genes to select. Will perform separate evaluation on the selected genes if provided."
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


def seed_everything(seed: int) -> None:
    """Seed RNGs before model/optimizer construction and training."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_cell_type_names(
    dataset: MddDataset | ReferencePopulationMddDataset,
) -> list[str]:
    if hasattr(dataset, "y") and isinstance(dataset.y, pd.DataFrame):
        return dataset.y.index.astype(str).tolist()
    return [str(cell_type) for cell_type in dataset.cell_types]


def main() -> None:
    args = parse_args()
    setup_logging(args.verbose)
    logging.debug("Arguments: %s", args)

    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive.")
    if args.learning_rate <= 0.0:
        raise ValueError("--learning-rate must be positive.")
    if args.epochs <= 0:
        raise ValueError("--epochs must be positive.")
    if args.n_layers < 0:
        raise ValueError("--n-layers must be non-negative.")
    if not 0.0 <= args.dropout < 1.0:
        raise ValueError("--dropout must be in [0, 1).")
    if args.weight_decay < 0.0:
        raise ValueError("--weight-decay must be non-negative.")

    if args.targets.suffix == ".h5ad":
        logging.info(
            "Detected reference-population setup: -X is a single (reference) "
            "feature set and -y is .h5ad. Using ReferencePopulationMddDataset "
            "(shared reference features, per-individual targets)."
        )
        dataset = ReferencePopulationMddDataset(
            X_feats=args.observations.with_suffix(".features.npy"),
            X_ensids=args.observations.with_suffix(".ensids.npy"),
            X_chroms=args.observations.with_suffix(".chroms.npy"),
            y=args.targets,
            normalize=args.norm_targets,
            cell_type=args.cell_type,
        )
    else:
        if args.cell_type is not None:
            logging.warning(
                "--cell-type is only used with population h5ad targets; ignoring."
            )
        dataset = MddDataset(
            X_feats=args.observations.with_suffix(".features.npy"),
            X_ensids=args.observations.with_suffix(".ensids.npy"),
            X_chroms=args.observations.with_suffix(".chroms.npy"),
            y=args.targets,
            normalize=args.norm_targets
        )

    cell_type_names = get_cell_type_names(dataset)

    if args.select_genes is not None:
        selected_genes = set(pd.read_csv(args.select_genes, sep="\t")["ENSID"])
        logging.debug(f"Selecting {len(selected_genes)} genes for separate evaluation: {selected_genes}")
        selected_dataset = dataset.select_genes(selected_genes)
        logging.debug(f"Selected dataset size: {len(selected_dataset)}")

    train_dataset, eval_dataset, test_dataset = get_train_test_dataset(dataset)

    logging.debug(
        "Dataset loaded using the fixed scPrediXcan chromosome split."
    )
    logging.debug(f"Train dataset size: {len(train_dataset)}")
    logging.debug(f"Eval dataset size:  {len(eval_dataset)}")
    logging.debug(f"Test dataset size:  {len(test_dataset)}")
    logging.debug(f"Total dataset size: {len(dataset)}")
    partition_gene_counts = (
        len(train_dataset.X_ensids),
        len(eval_dataset.X_ensids),
        len(test_dataset.X_ensids),
    )
    if min(partition_gene_counts) == 0:
        raise RuntimeError(
            "At least one chromosome partition is empty. Check the values in "
            "<X>.chroms.npy."
        )

    # For the reference-population setup, every individual shares the same x
    # (reference features) per gene but has a different y. Under MSE-style
    # training the model converges to the population mean of y at that x, so
    # loss/Pearson evaluation should compare against that same mean rather
    # than any single individual's value. We keep the original per-individual
    # datasets around as "calibration" datasets, since uncertainty
    # calibration (ENCE / uncertainty-error Spearman, deep-ensemble only)
    # needs the real inter-individual spread to be meaningful.
    eval_calibration_dataset = None
    test_calibration_dataset = None
    selected_calibration_dataset = None
    if isinstance(dataset, ReferencePopulationMddDataset):
        logging.info(
            "Reference-population dataset detected: loss/Pearson evaluation "
            "will use the per-gene population mean target (what MSE-style "
            "training converges to) instead of any single individual's value."
        )
        eval_calibration_dataset = eval_dataset
        test_calibration_dataset = test_dataset
        eval_dataset = eval_dataset.to_population_mean()
        test_dataset = test_dataset.to_population_mean()
        if args.select_genes is not None:
            selected_calibration_dataset = selected_dataset
            selected_dataset = selected_dataset.to_population_mean()

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True
    )
    eval_loader = torch.utils.data.DataLoader(
        eval_dataset, batch_size=args.batch_size, shuffle=False
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False
    )
    eval_calibration_loader = (
        torch.utils.data.DataLoader(eval_calibration_dataset, batch_size=args.batch_size, shuffle=False)
        if eval_calibration_dataset is not None else None
    )
    test_calibration_loader = (
        torch.utils.data.DataLoader(test_calibration_dataset, batch_size=args.batch_size, shuffle=False)
        if test_calibration_dataset is not None else None
    )

    if args.norm_inputs:
        logging.debug("Applying log-transform to input features...")
        train_dataset.apply_feature_log_transform()
        eval_dataset.norm_features = train_dataset.norm_features
        test_dataset.norm_features = train_dataset.norm_features
        if args.select_genes is not None:
            selected_dataset.norm_features = train_dataset.norm_features
        if selected_calibration_dataset is not None:
            selected_calibration_dataset.norm_features = (
                train_dataset.norm_features
            )
        if eval_calibration_dataset is not None:
            eval_calibration_dataset.norm_features = train_dataset.norm_features
        if test_calibration_dataset is not None:
            test_calibration_dataset.norm_features = train_dataset.norm_features

    input_dim  = train_dataset[0][0].shape[0]
    output_dim = train_dataset[0][1].shape[0]
    logging.debug(f"Input dim: {input_dim}, Output dim: {output_dim}")

    seed_everything(args.seed)
    logging.debug(
        "Seeded model/optimizer initialization and training with seed %d.",
        args.seed,
    )

    if args.model_name == 'mlp':
        model = MLPPredictor(input_dim=input_dim,
                             output_dim=output_dim,
                             n_layers=args.n_layers,
                             layer_norm=args.norm_layer,
                             dropout=args.dropout).to(device)
    elif args.model_name == 'mlp-sep':
        model = SeparateMLPPredictor(input_dim=input_dim,
                                     output_dim=output_dim,
                                     n_layers=args.n_layers,
                                     layer_norm=args.norm_layer,
                                     dropout=args.dropout).to(device)
    elif args.model_name == 'deep-ensemble':
        model = MLPEnsemble(n_models=5,
                            input_dim=input_dim,
                            output_dim=output_dim,
                            n_layers=args.n_layers,
                            layer_norm=args.norm_layer,
                            dropout=args.dropout).to(device)
    else:
        raise ValueError(f"Model {args.model_name} is not supported.")
    
    if args.run_name is not None:
        wb_logger = WandBLogger(enabled=True, model=model, run_name=args.run_name)
    else:
        wb_logger = WandBLogger(enabled=False)

    if (
        args.cossim_lambda
        + args.mpc_lambda
        + args.mse_lambda
        + args.pnll_lambda
        <= 0.0
        and args.model_name != "deep-ensemble"
    ):
        raise ValueError(
            "At least one loss lambda must be set to a positive value."
        )
    
    loss_dict = {}
    loss_lambda_dict = {}
    if args.cossim_lambda > 0.0:
        loss_dict['cossim'] = CosineSimilarityLoss()
        loss_lambda_dict['cossim'] = args.cossim_lambda
    if args.mpc_lambda > 0.0:
        loss_dict['mpc'] = MPCLoss()
        loss_lambda_dict['mpc'] = args.mpc_lambda
    if args.mse_lambda > 0.0:
        loss_dict['mse'] = MSELoss()
        loss_lambda_dict['mse'] = args.mse_lambda
    if args.pnll_lambda > 0.0:
        loss_dict['pnll'] = PNLLLoss()
        loss_lambda_dict['pnll'] = args.pnll_lambda
    
    if args.model_name == 'deep-ensemble':
        # for deep ensemble, we always add the Gaussian NLL loss for uncertainty estimation
        # we delete all other losses, since they don't work with the (mean, var) output format of the ensemble
        loss_dict['gnll'] = GaussianNLLLoss()
        loss_lambda_dict['gnll'] = 1.0 # TODO implement scale later, or not?

    early_stopping = EarlyStopping(patience=5, min_delta=1e-6, mode="min") if args.early_stop else None
    logging.debug(f"Early stopping: {early_stopping is not None}")

    total_steps = args.epochs * len(train_loader)
    warmup_steps = int(0.1 * total_steps)  # 10% warmup

    def make_scheduler(optimizer):
        if warmup_steps == 0:
            return CosineAnnealingLR(optimizer, T_max=max(total_steps, 1))

        # start at 1% of lr and linearly warm up to 100%
        warmup_scheduler = LinearLR(
            optimizer,
            start_factor=0.01,
            end_factor=1.0,
            total_iters=warmup_steps,
        )
        cosine_scheduler = CosineAnnealingLR(
            optimizer,
            T_max=max(total_steps - warmup_steps, 1),
        )
        return SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[warmup_steps],
        )

    separate_model = args.model_name == "mlp-sep"
    if separate_model:
        optimizers = [
            torch.optim.Adam(
                cell_model.parameters(),
                lr=args.learning_rate,
                weight_decay=args.weight_decay,
            )
            for cell_model in model.cell_type_models
        ]
        schedulers = [make_scheduler(optimizer) for optimizer in optimizers]
    else:
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=args.learning_rate,
            weight_decay=args.weight_decay,
        )
        scheduler = make_scheduler(optimizer)

    logging.info(
        "Optimization: Adam(lr=%g, weight_decay=%g), dropout=%g, "
        "scheduler=warmup-cosine.",
        args.learning_rate,
        args.weight_decay,
        args.dropout,
    )

    logging.info("Starting training...")

    if args.model_name == 'deep-ensemble':
        train_ensemble_model(
            model=model,
            train_loader=train_loader,
            eval_loader=eval_loader,
            loss_dict=loss_dict,
            loss_lambda_dict=loss_lambda_dict,
            optimizer=optimizer,
            scheduler=scheduler,
            wb_logger=wb_logger,
            early_stopping=early_stopping,
            epochs=args.epochs,
            device=device,
            eval_calibration_loader=eval_calibration_loader
        )

        logging.info("Training done, starting evaluation...")

        evaluate_ensemble_model(
            model=model,
            eval_loader=test_loader,
            loss_dict=loss_dict,
            loss_lambda_dict=loss_lambda_dict,
            device=device,
            wb_logger=wb_logger,
            mode="test",
            calibration_loader=test_calibration_loader
        )

        if args.select_genes is not None:
            logging.info(f"Starting separate evaluation on selected genes...")
            selected_calibration_loader = (
                torch.utils.data.DataLoader(selected_calibration_dataset, batch_size=args.batch_size, shuffle=False)
                if selected_calibration_dataset is not None else None
            )
            evaluate_ensemble_model(
                model=model,
                eval_loader=torch.utils.data.DataLoader(selected_dataset, batch_size=args.batch_size, shuffle=False),
                loss_dict=loss_dict,
                loss_lambda_dict=loss_lambda_dict,
                device=device,
                wb_logger=wb_logger,
                mode="test_selected_genes",
                calibration_loader=selected_calibration_loader
            )
    elif separate_model:
        train_separate_model(
            model=model,
            train_loader=train_loader,
            eval_loader=eval_loader,
            loss_dict=loss_dict,
            loss_lambda_dict=loss_lambda_dict,
            optimizers=optimizers,
            schedulers=schedulers,
            wb_logger=wb_logger,
            early_stopping=early_stopping,
            epochs=args.epochs,
            device=device,
        )

        logging.info("Training done, starting evaluation...")

        evaluate_separate_model(
            model=model,
            eval_loader=test_loader,
            loss_dict=loss_dict,
            loss_lambda_dict=loss_lambda_dict,
            device=device,
            wb_logger=wb_logger,
            mode="test",
        )

        if args.select_genes is not None:
            logging.info(f"Starting separate evaluation on selected genes...")
            evaluate_separate_model(
                model=model,
                eval_loader=torch.utils.data.DataLoader(
                    selected_dataset,
                    batch_size=args.batch_size,
                    shuffle=False
                ),
                loss_dict=loss_dict,
                loss_lambda_dict=loss_lambda_dict,
                device=device,
                wb_logger=wb_logger,
                mode="test_selected_genes",
            )
    else:
        train_single_model(
            model=model,
            train_loader=train_loader,
            eval_loader=eval_loader,
            loss_dict=loss_dict,
            loss_lambda_dict=loss_lambda_dict,
            optimizer=optimizer,
            scheduler=scheduler,
            wb_logger=wb_logger,
            early_stopping=early_stopping,
            epochs=args.epochs,
            device=device
        )

        logging.info("Training done, starting evaluation...")

        evaluate_single_model(
            model=model,
            eval_loader=test_loader,
            loss_dict=loss_dict,
            loss_lambda_dict=loss_lambda_dict,
            device=device,
            wb_logger=wb_logger,
            mode="test"
        )

        if args.select_genes is not None:
            logging.info(f"Starting separate evaluation on selected genes...")
            evaluate_single_model(
                model=model,
                eval_loader=torch.utils.data.DataLoader(selected_dataset, batch_size=args.batch_size, shuffle=False),
                loss_dict=loss_dict,
                loss_lambda_dict=loss_lambda_dict,
                device=device,
                wb_logger=wb_logger,
                mode="test_selected_genes"
            )

    wb_logger.finish()

    if args.output is not None:
        os.makedirs(args.output.parent, exist_ok=True)
        torch.save({
            "model_state_dict": model.state_dict(),
            "input_dim": input_dim,
            "output_dim": output_dim,
            "n_layers": args.n_layers,
            "layer_norm": args.norm_layer,
            "model_name": args.model_name,
            "norm_targets": args.norm_targets,
            "dropout": args.dropout,
            "weight_decay": args.weight_decay,
            "learning_rate": args.learning_rate,
            "batch_size": args.batch_size,
            "epochs": args.epochs,
            "seed": args.seed,
            "scheduler": "warmup-cosine",
            "loss_lambdas": dict(loss_lambda_dict),
            "posthoc_variance_calibration": (
                "validation_scalar_gaussian_nll"
                if args.model_name == "deep-ensemble"
                else None
            ),
            "variance_scale": (
                float(model.variance_scale.item())
                if args.model_name == "deep-ensemble"
                else None
            ),
            "cell_types": list(cell_type_names),
            "chromosome_split": {
                "train": list(dict.fromkeys(train_dataset.X_chroms.astype(str))),
                "validation": list(
                    dict.fromkeys(eval_dataset.X_chroms.astype(str))
                ),
                "test": list(dict.fromkeys(test_dataset.X_chroms.astype(str))),
            },
        }, args.output)
        logging.info(f"Model saved to {args.output}.")

        idx2ct = np.asarray(cell_type_names, dtype=object)

        mapping_path = args.output.with_suffix(".idx2ct.npy")
        np.save(mapping_path, idx2ct)
        logging.info(f"Cell-type mapping saved to {mapping_path}.")

    logging.info("Done.")

if __name__ == "__main__":
    main()
