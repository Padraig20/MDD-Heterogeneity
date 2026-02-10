from __future__ import annotations
import argparse
import logging
from pathlib import Path

import torch
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR
from dataset import MddDataset
from utils import get_train_test_dataset, EarlyStopping

from models.mlp import MLPPredictor
from models.mlp_deep_ensemble import MLPEnsemble

from utils import train_single_model, evaluate_single_model
from utils import train_ensemble_model, evaluate_ensemble_model

from loss.cossim_loss import CosineSimilarityLoss
from loss.mpc_loss import MPCLoss
from loss.mse_loss import MSELoss
from loss.pnll_loss import PNLLLoss
from loss.gnll_loss import GaussianNLLLoss

from wandb_logger import WandBLogger

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
        help="Path to input observations files; type in only the common name of the files, i.e. without extension(s)."
    )
    parser.add_argument(
        "-y", "--targets",
        type=Path,
        required=True,
        help="Path to the target file (csv)."
    )
    parser.add_argument(
        "-m", "--model-name",
        type=str,
        default="mlp",
        choices=["mlp", "deep_ensemble"],
        help="Name of the model to train. List of available models will be expanded in the future."
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
        "-e", "--epochs",
        type=int,
        default=10,
        help="Number of training epochs."
    )
    parser.add_argument(
        "-nl", "--n-layers",
        type=int,
        default=1,
        help="Number of layers in the neural network."
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
        help="Random seed for splitting the data."
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
        "--norm-targets",
        action="store_true",
        help="Log-transform target labels."
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

def main() -> None:
    args = parse_args()
    setup_logging(args.verbose)
    logging.debug("Arguments: %s", args)

    dataset = MddDataset(
        X_feats=args.observations.with_suffix(".features.npy"),
        X_ensids=args.observations.with_suffix(".ensids.npy"),
        X_chroms=args.observations.with_suffix(".chroms.npy"),
        y=args.targets,
        normalize=args.norm_targets,
    )

    train_dataset, eval_dataset, test_dataset = get_train_test_dataset(dataset, seed=args.seed)

    logging.debug(f"Dataset loaded and split according to seed {args.seed}.")
    logging.debug(f"Train dataset size: {len(train_dataset)}")
    logging.debug(f"Eval dataset size:  {len(eval_dataset)}")
    logging.debug(f"Test dataset size:  {len(test_dataset)}")
    logging.debug(f"Total dataset size: {len(dataset)}")

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True
    )
    eval_loader = torch.utils.data.DataLoader(
        eval_dataset, batch_size=args.batch_size, shuffle=False
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False
    )

    if args.norm_inputs:
        logging.debug("Applying log-transform to input features...")
        train_dataset.apply_feature_log_transform()
        eval_dataset.norm_features = train_dataset.norm_features
        test_dataset.norm_features = train_dataset.norm_features

    input_dim  = train_dataset[0][0].shape[0]
    output_dim = train_dataset[0][1].shape[0]
    logging.debug(f"Input dim: {input_dim}, Output dim: {output_dim}")
    if args.model_name == 'mlp':
        model = MLPPredictor(input_dim=input_dim,
                             output_dim=output_dim,
                             n_layers=args.n_layers,
                             layer_norm=args.norm_layer).to(device)
    elif args.model_name == 'deep_ensemble':
        model = MLPEnsemble(n_models=5,
                            input_dim=input_dim,
                            output_dim=output_dim,
                            n_layers=args.n_layers,
                            layer_norm=args.norm_layer).to(device)
    else:
        raise ValueError(f"Model {args.model_name} is not supported.")
    
    if args.run_name is not None:
        wb_logger = WandBLogger(enabled=True, model=model, run_name=args.run_name)
    else:
        wb_logger = WandBLogger(enabled=False)
        
    if args.cossim_lambda + args.mpc_lambda + args.mse_lambda + args.pnll_lambda <= 0.0:
        raise ValueError("I'm afraid you'll have to set at least one loss lambda to a positive value ;)")
    
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
    
    if args.model_name == 'deep_ensemble':
        # for deep ensemble, we always add the Gaussian NLL loss for uncertainty estimation
        # we delete all other losses, since they don't work with the (mean, var) output format of the ensemble
        loss_dict = {}
        loss_lambda_dict = {}
        loss_dict['gnll'] = GaussianNLLLoss()
        loss_lambda_dict['gnll'] = 1.0 # TODO implement scale later, or not?

    early_stopping = EarlyStopping(patience=5, min_delta=1e-6, mode="min") if args.early_stop else None
    logging.debug(f"Early stopping: {early_stopping is not None}")
    
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    total_steps = args.epochs * len(train_loader)
    warmup_steps = int(0.1 * total_steps)  # 10% warmup

    # start at 1% or lr and linearly warm up to 100%
    warmup_scheduler = LinearLR(optimizer, start_factor=0.01, end_factor=1.0, total_iters=warmup_steps)
    cosine_scheduler = CosineAnnealingLR(optimizer, T_max=total_steps - warmup_steps)
    scheduler        = SequentialLR(optimizer, schedulers=[warmup_scheduler, cosine_scheduler], milestones=[warmup_steps])

    logging.info("Starting training...")

    if args.model_name == 'deep_ensemble':
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
            device=device
        )

        logging.info("Training done, starting evaluation...")

        evaluate_ensemble_model(
            model=model,
            eval_loader=test_loader,
            loss_dict=loss_dict,
            loss_lambda_dict=loss_lambda_dict,
            device=device,
            wb_logger=wb_logger,
            mode="test"
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

    if args.output is not None:
        torch.save(model.state_dict(), args.output)
        logging.info(f"Model saved to {args.output}.")
    
    wb_logger.finish()

    logging.info("Done.")

if __name__ == "__main__":
    main()