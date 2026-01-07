from __future__ import annotations
import argparse
import logging
from pathlib import Path
from tqdm import tqdm

import torch
from utils import get_train_test_dataset
from models.mlp import MLPPredictor
from dataset import MddDataset
from loss.cossim_loss import CosineSimilarityLoss
from loss.mpc_loss import MPCLoss
from loss.mse_loss import MSELoss
from loss.pnll_loss import PNLLLoss
from metrics import MeanCellPearson, MeanGenePearson
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
        choices=["mlp"],
        help="Name of the model to train. List of available models will be expanded in the future."
    )
    parser.add_argument(
        "-l", "--loss",
        type=str,
        default="seq2cells",
        choices=["seq2cells", "composite"],
        help="Name of the loss function to use."
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
        "-hd", "--hidden-dim",
        type=int,
        default=512,
        help="Hidden dimension of neural network."
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
        default=1.0,
        help="Lambda parameter (weight) for the Mean Squared Error loss."
    )
    parser.add_argument(
        "--pnll-lambda",
        type=float,
        default=0.0,
        help="Lambda parameter (weight) for the Poisson Negative Log Likelihood loss."
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

def train_model(
    model: torch.nn.Module,
    train_loader: torch.utils.data.DataLoader,
    loss_dict: dict,
    loss_lambda_dict: dict,
    optimizer: torch.optim.Optimizer,
    wb_logger: WandBLogger,
    epochs: int = 10,
) -> None:
    """Train the model."""

    log_dict = {
        "train/loss": 0.0,
        "train/pearson_cells": 0.0,
        "train/pearson_genes": 0.0,
    }

    for key in loss_dict.keys():
        log_dict[f"train/{key}_loss"] = 0.0

    model.train()
    for epoch in range(epochs):
        metric_cells = MeanCellPearson(n_cells=model.output_dim).to(device)
        metric_genes = MeanGenePearson(n_cells=model.output_dim, n_genes=len(train_loader.dataset)).to(device)
        for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}"):
            inputs, targets = batch
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)

            composite_loss = torch.tensor(0.0, device=device)
            for key in loss_dict.keys():
                loss = loss_dict[key](outputs, targets) * loss_lambda_dict[key]
                log_dict[f"train/{key}_loss"] += loss.item()
                composite_loss += loss

            composite_loss.backward()
            optimizer.step()
            metric_cells.update(outputs, targets)
            metric_genes.update(outputs, targets)
            log_dict["train/loss"] += composite_loss.item()

        log_dict["train/loss"] /= len(train_loader)
        log_dict["train/pearson_cells"] = metric_cells.compute().mean().item()
        log_dict["train/pearson_genes"] = metric_genes.compute().mean().item()

        for key in loss_dict.keys():
            log_dict[f"train/{key}_loss"] /= len(train_loader)
        
        tqdm.write(f"Epoch {epoch + 1}/{epochs}, Loss: {log_dict['train/loss']:.4f}, PC Cells: {log_dict['train/pearson_cells']:.4f}, PC Genes: {log_dict['train/pearson_genes']:.4f}")
        
        wb_logger.log(log_dict)

def evaluate_model(
    model: torch.nn.Module,
    eval_loader: torch.utils.data.DataLoader,
    loss_dict: dict,
    loss_lambda_dict: dict,
    wb_logger: WandBLogger,
) -> None:
    """Evaluate the model."""

    log_dict = {
        "eval/loss": 0.0,
        "eval/pearson_cells": 0.0,
        "eval/pearson_genes": 0.0,
    }
    
    for key in loss_dict.keys():
        log_dict[f"eval/{key}_loss"] = 0.0

    model.eval()
    metric_cells = MeanCellPearson(n_cells=model.output_dim).to(device)
    metric_genes = MeanGenePearson(n_cells=model.output_dim, n_genes=len(eval_loader.dataset)).to(device)
    with torch.no_grad():
        for batch in tqdm(eval_loader, desc="Evaluating"):
            inputs, targets = batch
            inputs, targets = inputs.to(device), targets.to(device)

            outputs = model(inputs)
            composite_loss = torch.tensor(0.0, device=device)
            for key in loss_dict.keys():
                loss = loss_dict[key](outputs, targets) * loss_lambda_dict[key]
                log_dict[f"eval/{key}_loss"] += loss.item()
                composite_loss += loss

            log_dict["eval/loss"] += composite_loss.item()
            metric_cells.update(outputs, targets)
            metric_genes.update(outputs, targets)

    log_dict["eval/loss"] /= len(eval_loader)
    log_dict["eval/pearson_cells"] = metric_cells.compute().mean().item()
    log_dict["eval/pearson_genes"] = metric_genes.compute().mean().item()

    for key in loss_dict.keys():
        log_dict[f"eval/{key}_loss"] /= len(eval_loader)
    
    logging.info(f"Test Loss: {log_dict['eval/loss']:.4f}, PC Cells: {metric_cells.compute().mean():.4f}, PC Genes: {metric_genes.compute().mean():.4f}")
    
    wb_logger.log(log_dict)

def main() -> None:
    args = parse_args()
    setup_logging(args.verbose)
    logging.debug("Arguments: %s", args)

    dataset = MddDataset(
        X_feats=args.observations.with_suffix(".features.npy"),
        X_ensids=args.observations.with_suffix(".ensids.npy"),
        X_chroms=args.observations.with_suffix(".chroms.npy"),
        y=args.targets,
    )

    train_dataset, eval_dataset = get_train_test_dataset(dataset)

    logging.debug(f"Train dataset size: {len(train_dataset)}")
    logging.debug(f"Eval dataset size:  {len(eval_dataset)}")
    logging.debug(f"Total dataset size: {len(dataset)}")

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True
    )
    eval_loader = torch.utils.data.DataLoader(
        eval_dataset, batch_size=args.batch_size, shuffle=False
    )

    input_dim = train_dataset[0][0].shape[0]
    output_dim = train_dataset[0][1].shape[0]
    logging.debug(f"Input dim: {input_dim}, Output dim: {output_dim}")
    if args.model_name == 'mlp':
        model = MLPPredictor(input_dim=input_dim,
                             output_dim=output_dim,
                             hidden_dim=args.hidden_dim).to(device)
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
    
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    logging.info("Starting training...")

    train_model(
        model=model,
        train_loader=train_loader,
        loss_dict=loss_dict,
        loss_lambda_dict=loss_lambda_dict,
        optimizer=optimizer,
        wb_logger=wb_logger,
        epochs=args.epochs
    )

    logging.info("Training done, starting evaluation...")

    evaluate_model(
        model=model,
        eval_loader=eval_loader,
        loss_dict=loss_dict,
        loss_lambda_dict=loss_lambda_dict,
        wb_logger=wb_logger,
    )

    if args.output is not None:
        torch.save(model.state_dict(), args.output)
        logging.info(f"Model saved to {args.output}.")
    
    wb_logger.finish()

    logging.info("Done.")

if __name__ == "__main__":
    main()