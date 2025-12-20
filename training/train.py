from __future__ import annotations
import argparse
import logging
from pathlib import Path
from tqdm import tqdm

import torch
from utils import get_train_test_dataset
from models.mlp import MLPPredictor, MLPUncertainty
from dataset import MddDataset
from loss.composite_loss import CompositeLoss
from loss.seq2cells_loss import Seq2CellsLoss
from loss.lika_loss import LikaLoss
from metrics import MeanCellPearson, MeanGenePearson

"""
get_feats_from_seqs.py

Script that takes input from the human reference genome sequences we extracted
earlier (around the TSS) and then extracts features from these sequences using
some sort of pLM/gLM model. Here, we use e.g. Enformer via Hugging Face

https://huggingface.co/EleutherAI/enformer-official-rough
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
        "-u", "--use-uncertainty",
        action="store_true",
        help="Whether to use uncertainty estimation in the model."
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

def train_model(
    model: torch.nn.Module,
    unc_model: torch.nn.Module,
    train_loader: torch.utils.data.DataLoader,
    loss_fn: torch.nn.Module,
    unc_loss_fn: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    unc_optimizer: torch.optim.Optimizer,
    epochs: int = 10
) -> None:
    """Train the model."""
    model.train()
    if unc_model:
        unc_model.train()
    for epoch in range(epochs):
        total_loss     = 0.0
        unc_total_loss = 0.0
        metric_cells = MeanCellPearson(n_cells=model.output_dim).to(device)
        metric_genes = MeanGenePearson(n_cells=model.output_dim, n_genes=len(train_loader.dataset)).to(device)
        for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}"):
            inputs, targets = batch
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss    = loss_fn(outputs, targets)
            loss.backward()
            optimizer.step()
            metric_cells.update(outputs, targets)
            metric_genes.update(outputs, targets)
            total_loss += loss.item()

            if unc_model:
                unc_optimizer.zero_grad()
                unc_outputs = unc_model(inputs)
                unc_mu      = outputs.detach()
                unc_sigma   = unc_outputs
                unc_loss    = unc_loss_fn(unc_mu, unc_sigma, targets)
                unc_loss.backward()
                unc_optimizer.step()
                unc_total_loss += unc_loss.item()
                
                avg_unc_loss = unc_total_loss / len(train_loader)

        avg_loss = total_loss / len(train_loader)

        if unc_model:
            unc_loss_fn.set_temperatures()
            avg_unc_loss = unc_total_loss / len(train_loader)
        
        if unc_model:
            tqdm.write(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}, Uncertainty: {avg_unc_loss:.4f}, PC Cells: {metric_cells.compute().mean():.4f}, PC Genes: {metric_genes.compute().mean():.4f}")
        else:
            tqdm.write(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}, PC Cells: {metric_cells.compute().mean():.4f}, PC Genes: {metric_genes.compute().mean():.4f}")

def evaluate_model(
    model: torch.nn.Module,
    unc_model: torch.nn.Module,
    test_loader: torch.utils.data.DataLoader,
    loss_fn: torch.nn.Module,
    unc_loss_fn: torch.nn.Module
) -> None:
    """Evaluate the model."""
    model.eval()
    if unc_model:
        unc_model.eval()
    total_loss = 0.0
    unc_total_loss = 0.0
    metric_cells = MeanCellPearson(n_cells=model.output_dim).to(device)
    metric_genes = MeanGenePearson(n_cells=model.output_dim, n_genes=len(test_loader.dataset)).to(device)
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating"):
            inputs, targets = batch
            inputs, targets = inputs.to(device), targets.to(device)

            outputs = model(inputs)
            loss    = loss_fn(outputs, targets)
            total_loss += loss.item()
            metric_cells.update(outputs, targets)
            metric_genes.update(outputs, targets)

            if unc_model:
                unc_outputs = unc_model(inputs)
                unc_mu     = outputs
                unc_sigma  = unc_outputs
                unc_loss   = unc_loss_fn(unc_mu, unc_sigma, targets)
                unc_total_loss += unc_loss.item()

    avg_loss = total_loss / len(test_loader)
    if unc_model:
        avg_unc_loss = unc_total_loss / len(test_loader)
        logging.info(f"Test Loss: {avg_loss:.4f}, Uncertainty: {avg_unc_loss:.4f}, PC Cells: {metric_cells.compute().mean():.4f}, PC Genes: {metric_genes.compute().mean():.4f}")
    else:
        logging.info(f"Test Loss: {avg_loss:.4f}, PC Cells: {metric_cells.compute().mean():.4f}, PC Genes: {metric_genes.compute().mean():.4f}")

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

    train_dataset, test_dataset = get_train_test_dataset(dataset)

    logging.debug(f"Train dataset size: {len(train_dataset)}")
    logging.debug(f"Test dataset size:  {len(test_dataset)}")
    logging.debug(f"Total dataset size: {len(dataset)}")

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False
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
    
    unc_model     = None
    unc_loss_fn   = None
    unc_optimizer = None
    if args.use_uncertainty:
        unc_model = MLPUncertainty(input_dim=input_dim,
                                   output_dim=output_dim,
                                   hidden_dim=args.hidden_dim).to(device)
        unc_loss_fn   = LikaLoss()
        unc_optimizer = torch.optim.Adam(unc_model.parameters(), lr=args.learning_rate)
        logging.info("Using uncertainty estimation.")
    
    if args.loss == 'composite':
        loss_fn = CompositeLoss()
    elif args.loss == 'seq2cells':
        loss_fn = Seq2CellsLoss()
    else:
        raise ValueError(f"Loss function {args.loss} is not supported.")
    
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    logging.info("Starting training...")

    train_model(
        model=model,
        unc_model=unc_model,
        train_loader=train_loader,
        loss_fn=loss_fn,
        unc_loss_fn=unc_loss_fn,
        optimizer=optimizer,
        unc_optimizer=unc_optimizer,
        epochs=args.epochs
    )

    logging.info("Training done, starting evaluation...")

    evaluate_model(
        model=model,
        unc_model=unc_model,
        test_loader=test_loader,
        loss_fn=loss_fn,
        unc_loss_fn=unc_loss_fn
    )

    if args.output is not None:
        torch.save(model.state_dict(), args.output)
        logging.info(f"Model saved to {args.output}.")

        if unc_model:
            unc_path = args.output.with_name(args.output.stem + "_uncertainty.pth")
            torch.save(unc_model.state_dict(), unc_path)
            logging.info(f"Uncertainty model saved to {unc_path}.")

    logging.info("Done.")

if __name__ == "__main__":
    main()