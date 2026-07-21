import logging

from tqdm import tqdm
from src.training.dataset import MddDataset
from copy import deepcopy
import torch
import random

from torch.optim.lr_scheduler import SequentialLR

from src.training.metrics import MeanCellPearson, MeanGenePearson, UncertaintyCalibration
from src.training.wandb_logger import WandBLogger
import wandb
import matplotlib.pyplot as plt

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

    for epoch in range(epochs):
        model.train()
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
        log_dict["train/pearson_cells"] = metric_cells.compute().nanmean().item()
        log_dict["train/pearson_genes"] = metric_genes.compute().nanmean().item()
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
    log_dict[f"{mode}/pearson_cells"] = metric_cells.compute().nanmean().item()
    log_dict[f"{mode}/pearson_genes"] = metric_genes.compute().nanmean().item()
    log_dict[f"{mode}/pearson"] = (log_dict[f"{mode}/pearson_cells"] + log_dict[f"{mode}/pearson_genes"]) / 2.0

    for key in loss_dict.keys():
        log_dict[f"{mode}/{key}_loss"] /= len(eval_loader)
    
    logging.info(f"{mode.capitalize()} Loss: {log_dict[f'{mode}/loss']:.4f}, PC Cells: {metric_cells.compute().nanmean():.4f}, PC Genes: {metric_genes.compute().nanmean():.4f}, PC: {log_dict[f'{mode}/pearson']:.4f}")
    
    wb_logger.log(log_dict)

    return log_dict[f"{mode}/loss"]

# ---------------------------------------------------------------------------- #
# -------------- TRAINING AND EVALUATION FOR ENSEMBLE MODEL ------------------ #
# ---------------------------------------------------------------------------- #

def train_ensemble_model(
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
    device: torch.device = torch.device("cpu"),
    eval_calibration_loader: torch.utils.data.DataLoader | None = None,
) -> None:
    """Train the ensemble model. Evaluate after each epoch."""

    log_dict = {
        "train/epoch": 0,
        "train/loss": 0.0,
        "train/pearson_cells": 0.0,
        "train/pearson_genes": 0.0,
        "train/pearson": 0.0
    }

    for key in loss_dict.keys():
        log_dict[f"train/{key}_loss"] = 0.0

    for epoch in range(epochs):
        model.train()
        metric_cells = MeanCellPearson(n_cells=model.output_dim).to(device)
        metric_genes = MeanGenePearson(n_cells=model.output_dim, n_genes=len(train_loader.dataset)).to(device)
        for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}"):
            inputs, targets = batch
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()

            means, vars = model(inputs)

            composite_loss = torch.tensor(0.0, device=device)
            for m, v in zip(means, vars):
                output = torch.stack([m, v], dim=2)
                for key in loss_dict.keys():
                    loss = loss_dict[key](output, targets) * loss_lambda_dict[key]
                    log_dict[f"train/{key}_loss"] += loss.item()
                    composite_loss += loss

            composite_loss.backward()
            optimizer.step()

            output = torch.mean(torch.stack(means), dim=0)

            scheduler.step() # update lr and log
            wb_logger.log({"train/lr": optimizer.param_groups[0]['lr']})
            metric_cells.update(output, targets)
            metric_genes.update(output, targets)
            log_dict["train/loss"] += composite_loss.item()

        log_dict["train/epoch"] = epoch
        log_dict["train/loss"] /= len(train_loader)
        log_dict["train/pearson_cells"] = metric_cells.compute().nanmean().item()
        log_dict["train/pearson_genes"] = metric_genes.compute().nanmean().item()
        log_dict["train/pearson"] = (log_dict["train/pearson_cells"] + log_dict["train/pearson_genes"]) / 2.0

        for key in loss_dict.keys():
            log_dict[f"train/{key}_loss"] /= len(train_loader)
        
        tqdm.write(f"Epoch {epoch + 1}/{epochs}, Loss: {log_dict['train/loss']:.4f}, PC Cells: {log_dict['train/pearson_cells']:.4f}, PC Genes: {log_dict['train/pearson_genes']:.4f}, PC: {log_dict['train/pearson']:.4f}")
        
        wb_logger.log(log_dict)

        eval_loss = evaluate_ensemble_model(
                        model=model,
                        eval_loader=eval_loader,
                        loss_dict=loss_dict,
                        loss_lambda_dict=loss_lambda_dict,
                        wb_logger=wb_logger,
                        device=device,
                        epoch=epoch,
                        mode="eval",
                        calibration_loader=eval_calibration_loader,
        )

        if early_stopping is not None:
            if early_stopping.step(eval_loss, model):
                logging.info("Early stopping triggered! Restoring best model weights...")
                early_stopping.restore_best_weights(model)
                break

def evaluate_ensemble_model(
    model: torch.nn.Module,
    eval_loader: torch.utils.data.DataLoader,
    loss_dict: dict,
    loss_lambda_dict: dict,
    wb_logger: WandBLogger,
    device: torch.device = torch.device("cpu"),
    mode : str = "eval", # eval or test
    epoch: int = 0,
    calibration_loader: torch.utils.data.DataLoader | None = None,
) -> float:
    """Evaluate the ensemble model.

    `eval_loader` drives the loss/Pearson metrics (and the aleatoric/epistemic
    averages, which are a deterministic function of x and thus unaffected by
    which loader is used). If `calibration_loader` is given, it is used
    instead for uncertainty calibration (ENCE, uncertainty-error Spearman) --
    this lets callers evaluate accuracy against a population-mean target
    while still calibrating predicted variance against real per-individual
    targets. If `calibration_loader` is None, calibration falls back to
    `eval_loader` (previous behaviour).
    """

    log_dict = {
        f"{mode}/epoch": epoch,
        f"{mode}/loss": 0.0,
        f"{mode}/pearson_cells": 0.0,
        f"{mode}/pearson_genes": 0.0,
        f"{mode}/pearson": 0.0,
        f"{mode}/aleatoric": 0.0, # should theoretically stay consistent
        f"{mode}/epistemic": 0.0, # should theoretically go down with training
        f"{mode}/ence": 0.0,                       # calibration error of total uncertainty (lower is better)
        f"{mode}/uncertainty_error_spearman": 0.0  # rank corr. of total uncertainty vs error (higher is better)
    }
    
    for key in loss_dict.keys():
        log_dict[f"{mode}/{key}_loss"] = 0.0
    
    model.eval()
    metric_cells = MeanCellPearson(n_cells=model.output_dim).to(device)
    metric_genes = MeanGenePearson(n_cells=model.output_dim, n_genes=len(eval_loader.dataset)).to(device)
    calibration  = UncertaintyCalibration()

    epistemic_uncertainties = []
    aleatoric_uncertainties = []

    with torch.no_grad():
        for batch in tqdm(eval_loader, desc="Evaluating"):
            inputs, targets = batch
            inputs, targets = inputs.to(device), targets.to(device)

            prediction, aleatoric_unc, epistemic_unc = model(inputs)

            aleatoric_uncertainties.append(aleatoric_unc.mean(dim=0)) # avg aleatoric uncertainty across cells
            epistemic_uncertainties.append(epistemic_unc.mean(dim=0)) # avg epistemic uncertainty across cells

            if calibration_loader is None:
                # no separate per-individual pass available; fall back to
                # calibrating against whatever targets eval_loader provides.
                # total predictive variance = aleatoric + epistemic
                calibration.update(prediction, aleatoric_unc + epistemic_unc, targets)

            outputs = torch.stack([prediction, aleatoric_unc], dim=2) # shape (batch_size, output_dim, 2)
            composite_loss = torch.tensor(0.0, device=device)
            for key in loss_dict.keys():
                loss = loss_dict[key](outputs, targets) * loss_lambda_dict[key]
                log_dict[f"{mode}/{key}_loss"] += loss.item()
                composite_loss += loss

            log_dict[f"{mode}/loss"] += composite_loss.item()
            metric_cells.update(prediction, targets)
            metric_genes.update(prediction, targets)

    if calibration_loader is not None:
        with torch.no_grad():
            for batch in tqdm(calibration_loader, desc="Calibrating"):
                inputs, targets = batch
                inputs, targets = inputs.to(device), targets.to(device)

                prediction, aleatoric_unc, epistemic_unc = model(inputs)
                # total predictive variance = aleatoric + epistemic
                calibration.update(prediction, aleatoric_unc + epistemic_unc, targets)

    log_dict[f"{mode}/loss"] /= len(eval_loader)
    log_dict[f"{mode}/pearson_cells"] = metric_cells.compute().nanmean().item()
    log_dict[f"{mode}/pearson_genes"] = metric_genes.compute().nanmean().item()
    log_dict[f"{mode}/pearson"] = (log_dict[f"{mode}/pearson_cells"] + log_dict[f"{mode}/pearson_genes"]) / 2.0
    log_dict[f"{mode}/aleatoric"] = torch.mean(torch.stack(aleatoric_uncertainties)).item()
    log_dict[f"{mode}/epistemic"] = torch.mean(torch.stack(epistemic_uncertainties)).item()
    log_dict[f"{mode}/ence"] = calibration.compute_ence()
    log_dict[f"{mode}/uncertainty_error_spearman"] = calibration.compute_spearman()

    for key in loss_dict.keys():
        log_dict[f"{mode}/{key}_loss"] /= len(eval_loader)
    
    logging.info(f"{mode.capitalize()} Loss: {log_dict[f'{mode}/loss']:.4f}, PC Cells: {metric_cells.compute().nanmean():.4f}, PC Genes: {metric_genes.compute().nanmean():.4f}, PC: {log_dict[f'{mode}/pearson']:.4f}")
    logging.info(f"{mode.capitalize()} ENCE: {log_dict[f'{mode}/ence']:.4f}, Uncertainty-Error Spearman: {log_dict[f'{mode}/uncertainty_error_spearman']:.4f}")
    
    wb_logger.log(log_dict)

    # render calibration boxplot only when logging
    if wb_logger.enabled:
        fig = calibration.make_boxplot(title=f"{mode} uncertainty")
        if fig is not None:
            wb_logger.log({f"{mode}/mse_vs_uncertainty": wandb.Image(fig)})
            plt.close(fig)

    return log_dict[f"{mode}/loss"]