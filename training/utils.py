import logging

from tqdm import tqdm
from dataset import MddDataset
from copy import deepcopy
import torch
import random

from torch.optim.lr_scheduler import SequentialLR

from training.metrics import MeanCellPearson, MeanGenePearson
from training.wandb_logger import WandBLogger

# taken from scPrediXcan tutorial
# https://github.com/hakyimlab/scPrediXcan/blob/master/Scripts/ctPred/Tutorial.ipynb
all_chromosomes = ["1", "10", "13", "15", "16", "17", "18", "19", "2", "21", "22", "3", "4", "6", "8", "9", "X", "Y"] + ["11", "14", "7"] + ["12", "20", "5"]

def get_train_test_dataset(dataset: MddDataset, seed: int = 42):
    """Load dataset and split into train, val and test sets."""
    # chromosomes split into 3 parts, with 18, 3 and 3 chromosomes respectively
    random.seed(seed)
    random.shuffle(all_chromosomes)
    train_set = dataset.split_by_chromosome(all_chromosomes[:18])
    val_set   = dataset.split_by_chromosome(all_chromosomes[18:21])
    test_set  = dataset.split_by_chromosome(all_chromosomes[21:])
    return train_set, val_set, test_set

class EarlyStopping:
    def __init__(self, patience: int = 5, min_delta: float = 1e-6, mode: str = "min"):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode  # "min" or "max"
        self.best = None
        self.bad_epochs = 0
        self.best_state = None

    def _is_improvement(self, current: float) -> bool:
        if self.best is None:
            return True
        if self.mode == "min":
            return current < (self.best - self.min_delta)
        else:
            return current > (self.best + self.min_delta)

    def step(self, current: float, model: torch.nn.Module) -> bool:
        """Returns True if we should stop."""
        if self._is_improvement(current):
            self.best = current
            self.bad_epochs = 0
            # store best weights in RAM
            self.best_state = deepcopy(model.state_dict())
            return False

        self.bad_epochs += 1
        return self.bad_epochs >= self.patience

    def restore_best_weights(self, model: torch.nn.Module) -> None:
        if self.best_state is not None:
            model.load_state_dict(self.best_state)

# ---------------------------------------------------------------------------- #
# ---------------- TRAINING AND EVALUATION FOR SINGLE MODEL ------------------ #
# ---------------------------------------------------------------------------- #

def train_single_model(
    model: torch.nn.Module,
    train_loader: torch.utils.data.DataLoader,
    eval_loader: torch.utils.data.DataLoader,
    loss_dict: dict,
    loss_lambda_dict: dict,
    optimizer: torch.optim.Optimizer,
    scheduler: SequentialLR,
    wb_logger: WandBLogger,
    early_stopping: EarlyStopping,
    epochs: int = 10,
    device: torch.device = torch.device("cpu")
) -> None:
    """Train the model. Evaluate after each epoch."""

    log_dict = {
        "train/epoch": 0,
        "train/loss": 0.0,
        "train/pearson_cells": 0.0,
        "train/pearson_genes": 0.0,
        "train/pearson": 0.0
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

            scheduler.step() # update lr and log
            wb_logger.log({"train/lr": optimizer.param_groups[0]['lr']})

            metric_cells.update(outputs, targets)
            metric_genes.update(outputs, targets)
            log_dict["train/loss"] += composite_loss.item()

        log_dict["train/epoch"] = epoch
        log_dict["train/loss"] /= len(train_loader)
        log_dict["train/pearson_cells"] = metric_cells.compute().mean().item()
        log_dict["train/pearson_genes"] = metric_genes.compute().mean().item()
        log_dict["train/pearson"] = (log_dict["train/pearson_cells"] + log_dict["train/pearson_genes"]) / 2.0

        for key in loss_dict.keys():
            log_dict[f"train/{key}_loss"] /= len(train_loader)
        
        tqdm.write(f"Epoch {epoch + 1}/{epochs}, Loss: {log_dict['train/loss']:.4f}, PC Cells: {log_dict['train/pearson_cells']:.4f}, PC Genes: {log_dict['train/pearson_genes']:.4f}, PC: {log_dict['train/pearson']:.4f}")
        
        wb_logger.log(log_dict)

        eval_loss = evaluate_single_model(
                        model=model,
                        eval_loader=eval_loader,
                        loss_dict=loss_dict,
                        loss_lambda_dict=loss_lambda_dict,
                        wb_logger=wb_logger,
                        device=device,
                        epoch=epoch,
                    mode="eval"
        )

        if early_stopping is not None:
            if early_stopping.step(eval_loss, model):
                logging.info("Early stopping triggered! Restoring best model weights...")
                early_stopping.restore_best_weights(model)
                break

def evaluate_single_model(
    model: torch.nn.Module,
    eval_loader: torch.utils.data.DataLoader,
    loss_dict: dict,
    loss_lambda_dict: dict,
    wb_logger: WandBLogger,
    device: torch.device = torch.device("cpu"),
    mode : str = "eval", # eval or test
    epoch: int = 0,
) -> float:
    """Evaluate the model."""

    log_dict = {
        f"{mode}/epoch": epoch,
        f"{mode}/loss": 0.0,
        f"{mode}/pearson_cells": 0.0,
        f"{mode}/pearson_genes": 0.0,
        f"{mode}/pearson": 0.0
    }
    
    for key in loss_dict.keys():
        log_dict[f"{mode}/{key}_loss"] = 0.0
    
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
                log_dict[f"{mode}/{key}_loss"] += loss.item()
                composite_loss += loss

            log_dict[f"{mode}/loss"] += composite_loss.item()
            metric_cells.update(outputs, targets)
            metric_genes.update(outputs, targets)

    log_dict[f"{mode}/loss"] /= len(eval_loader)
    log_dict[f"{mode}/pearson_cells"] = metric_cells.compute().mean().item()
    log_dict[f"{mode}/pearson_genes"] = metric_genes.compute().mean().item()
    log_dict[f"{mode}/pearson"] = (log_dict[f"{mode}/pearson_cells"] + log_dict[f"{mode}/pearson_genes"]) / 2.0

    for key in loss_dict.keys():
        log_dict[f"{mode}/{key}_loss"] /= len(eval_loader)
    
    logging.info(f"{mode.capitalize()} Loss: {log_dict[f'{mode}/loss']:.4f}, PC Cells: {metric_cells.compute().mean():.4f}, PC Genes: {metric_genes.compute().mean():.4f}, PC: {log_dict[f'{mode}/pearson']:.4f}")
    
    wb_logger.log(log_dict)

    return log_dict[f"{mode}/loss"]

# ---------------------------------------------------------------------------- #
# -------------- TRAINING AND EVALUATION FOR ENSEMBLE MODEL ------------------ #
# ---------------------------------------------------------------------------- #

# TBA...