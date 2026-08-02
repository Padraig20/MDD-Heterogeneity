from collections.abc import Sequence
import logging

from tqdm import tqdm
from src.training.dataset import MddDataset, ReferencePopulationMddDataset
from copy import deepcopy
import numpy as np
import torch
import random

from src.training.metrics import (
    MeanCellPearson,
    MeanGenePearson,
    PopulationVarianceEvaluation,
    UncertaintyCalibration,
)
from src.training.wandb_logger import WandBLogger
import wandb
import matplotlib.pyplot as plt

# taken from scPrediXcan tutorial
# https://github.com/hakyimlab/scPrediXcan/blob/master/Scripts/ctPred/Tutorial.ipynb
ALL_CHROMOSOMES = ["1", "10", "13", "15", "16", "17", "18", "19", "2", "21", "22", "3", "4", "6", "8", "9", "X", "Y"] + ["11", "14", "7"] + ["12", "20", "5"]

# Backwards-compatible name, now immutable so repeated calls cannot mutate it.
all_chromosomes = ALL_CHROMOSOMES

def _canonical_chromosome(chromosome: object) -> str:
    value = str(chromosome)
    if value.lower().startswith("chr"):
        value = value[3:]
    return value.upper()


def _labels_in_dataset(
    dataset: MddDataset,
    canonical_chromosomes: Sequence[str],
) -> list[str]:
    """Map canonical chromosome names to the dataset's actual label style."""
    requested = {
        _canonical_chromosome(chromosome)
        for chromosome in canonical_chromosomes
    }
    return [
        str(chromosome)
        for chromosome in dict.fromkeys(dataset.X_chroms.astype(str))
        if _canonical_chromosome(chromosome) in requested
    ]


def get_train_test_dataset(dataset: MddDataset):
    """Load dataset and split into train, val and test sets."""
    # chromosomes split into 3 parts, with 18, 3 and 3 chromosomes respectively
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
    scheduler: torch.optim.lr_scheduler.LRScheduler | None,
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

            if scheduler is not None:
                scheduler.step()
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
                log_dict[f"{mode}/{key}_loss"] += (
                    loss.item() * targets.shape[0]
                )
                composite_loss += loss

            log_dict[f"{mode}/loss"] += (
                composite_loss.item() * targets.shape[0]
            )
            metric_cells.update(outputs, targets)
            metric_genes.update(outputs, targets)

    log_dict[f"{mode}/loss"] /= len(eval_loader.dataset)
    log_dict[f"{mode}/pearson_cells"] = metric_cells.compute().nanmean().item()
    log_dict[f"{mode}/pearson_genes"] = metric_genes.compute().nanmean().item()
    log_dict[f"{mode}/pearson"] = (log_dict[f"{mode}/pearson_cells"] + log_dict[f"{mode}/pearson_genes"]) / 2.0

    for key in loss_dict.keys():
        log_dict[f"{mode}/{key}_loss"] /= len(eval_loader.dataset)
    
    logging.info(f"{mode.capitalize()} Loss: {log_dict[f'{mode}/loss']:.4f}, PC Cells: {metric_cells.compute().nanmean():.4f}, PC Genes: {metric_genes.compute().nanmean():.4f}, PC: {log_dict[f'{mode}/pearson']:.4f}")
    
    wb_logger.log(log_dict)

    return log_dict[f"{mode}/loss"]

# ---------------------------------------------------------------------------- #
# -------- TRAINING AND EVALUATION FOR CELL-TYPE-SEPARATE MODELS ------------ #
# ---------------------------------------------------------------------------- #

def train_separate_model(
    model: torch.nn.Module,
    train_loader: torch.utils.data.DataLoader,
    eval_loader: torch.utils.data.DataLoader,
    loss_dict: dict,
    loss_lambda_dict: dict,
    optimizers: Sequence[torch.optim.Optimizer],
    schedulers: Sequence[torch.optim.lr_scheduler.LRScheduler | None],
    wb_logger: WandBLogger,
    early_stopping: EarlyStopping | None,
    epochs: int = 10,
    device: torch.device = torch.device("cpu"),
) -> None:
    """Train one scalar-output MLP independently for each cell type.

    Every cell-type model receives only its matching target column, owns its
    optimizer and scheduler, and completes backward/step before the next model
    is considered.  Consequently neither gradients nor optimizer state are
    shared across cell types.
    """

    n_cell_types = model.output_dim
    if len(model.cell_type_models) != n_cell_types:
        raise ValueError(
            "Expected one model per output cell type, got "
            f"{len(model.cell_type_models)} models for {n_cell_types} outputs."
        )
    if len(optimizers) != n_cell_types or len(schedulers) != n_cell_types:
        raise ValueError(
            "Cell-type-separate training requires one optimizer and scheduler "
            f"per cell type; got {len(optimizers)} optimizers and "
            f"{len(schedulers)} schedulers for {n_cell_types} cell types."
        )

    for epoch in range(epochs):
        log_dict = {
            "train/epoch": epoch,
            "train/loss": 0.0,
            "train/pearson_cells": 0.0,
            "train/pearson_genes": 0.0,
            "train/pearson": 0.0,
        }
        for key in loss_dict:
            log_dict[f"train/{key}_loss"] = 0.0

        model.train()
        metric_cells = MeanCellPearson(n_cells=n_cell_types).to(device)
        metric_genes = MeanGenePearson(
            n_cells=n_cell_types,
            n_genes=len(train_loader.dataset),
        ).to(device)

        for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}"):
            inputs, targets = batch
            inputs, targets = inputs.to(device), targets.to(device)

            cell_outputs = []
            batch_loss = 0.0
            batch_losses = {key: 0.0 for key in loss_dict}

            for cell_type_index, (optimizer, scheduler) in enumerate(
                zip(optimizers, schedulers)
            ):
                optimizer.zero_grad()
                output = model.forward_cell_type(inputs, cell_type_index)
                target = targets[..., cell_type_index : cell_type_index + 1]

                cell_loss = torch.tensor(0.0, device=device)
                for key, loss_fn in loss_dict.items():
                    weighted_loss = (
                        loss_fn(output, target) * loss_lambda_dict[key]
                    )
                    cell_loss = cell_loss + weighted_loss
                    batch_losses[key] += (
                        weighted_loss.detach().item() / n_cell_types
                    )

                cell_loss.backward()
                optimizer.step()
                if scheduler is not None:
                    scheduler.step()

                cell_outputs.append(output.detach())
                batch_loss += cell_loss.detach().item() / n_cell_types

            outputs = torch.cat(cell_outputs, dim=-1)
            metric_cells.update(outputs, targets)
            metric_genes.update(outputs, targets)

            log_dict["train/loss"] += batch_loss
            for key in loss_dict:
                log_dict[f"train/{key}_loss"] += batch_losses[key]

            # Schedules are initialized identically, so logging the first
            # optimizer retains the existing scalar learning-rate interface.
            wb_logger.log({"train/lr": optimizers[0].param_groups[0]["lr"]})

        log_dict["train/loss"] /= len(train_loader)
        log_dict["train/pearson_cells"] = metric_cells.compute().nanmean().item()
        log_dict["train/pearson_genes"] = metric_genes.compute().nanmean().item()
        log_dict["train/pearson"] = (
            log_dict["train/pearson_cells"]
            + log_dict["train/pearson_genes"]
        ) / 2.0
        for key in loss_dict:
            log_dict[f"train/{key}_loss"] /= len(train_loader)

        tqdm.write(
            f"Epoch {epoch + 1}/{epochs}, Loss: {log_dict['train/loss']:.4f}, "
            f"PC Cells: {log_dict['train/pearson_cells']:.4f}, "
            f"PC Genes: {log_dict['train/pearson_genes']:.4f}, "
            f"PC: {log_dict['train/pearson']:.4f}"
        )
        wb_logger.log(log_dict)

        eval_loss = evaluate_separate_model(
            model=model,
            eval_loader=eval_loader,
            loss_dict=loss_dict,
            loss_lambda_dict=loss_lambda_dict,
            wb_logger=wb_logger,
            device=device,
            epoch=epoch,
            mode="eval",
        )

        if early_stopping is not None and early_stopping.step(eval_loss, model):
            logging.info(
                "Early stopping triggered! Restoring best model weights..."
            )
            early_stopping.restore_best_weights(model)
            break


def evaluate_separate_model(
    model: torch.nn.Module,
    eval_loader: torch.utils.data.DataLoader,
    loss_dict: dict,
    loss_lambda_dict: dict,
    wb_logger: WandBLogger,
    device: torch.device = torch.device("cpu"),
    mode: str = "eval",
    epoch: int = 0,
) -> float:
    """Evaluate independent cell-type models through their combined output."""

    n_cell_types = model.output_dim
    log_dict = {
        f"{mode}/epoch": epoch,
        f"{mode}/loss": 0.0,
        f"{mode}/pearson_cells": 0.0,
        f"{mode}/pearson_genes": 0.0,
        f"{mode}/pearson": 0.0,
    }
    for key in loss_dict:
        log_dict[f"{mode}/{key}_loss"] = 0.0

    model.eval()
    metric_cells = MeanCellPearson(n_cells=n_cell_types).to(device)
    metric_genes = MeanGenePearson(
        n_cells=n_cell_types,
        n_genes=len(eval_loader.dataset),
    ).to(device)

    with torch.no_grad():
        for batch in tqdm(eval_loader, desc="Evaluating"):
            inputs, targets = batch
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)

            batch_loss = 0.0
            for cell_type_index in range(n_cell_types):
                output = outputs[
                    ..., cell_type_index : cell_type_index + 1
                ]
                target = targets[
                    ..., cell_type_index : cell_type_index + 1
                ]
                for key, loss_fn in loss_dict.items():
                    weighted_loss = (
                        loss_fn(output, target)
                        * loss_lambda_dict[key]
                        / n_cell_types
                    )
                    log_dict[f"{mode}/{key}_loss"] += (
                        weighted_loss.item() * targets.shape[0]
                    )
                    batch_loss += weighted_loss.item()

            log_dict[f"{mode}/loss"] += batch_loss * targets.shape[0]
            metric_cells.update(outputs, targets)
            metric_genes.update(outputs, targets)

    log_dict[f"{mode}/loss"] /= len(eval_loader.dataset)
    log_dict[f"{mode}/pearson_cells"] = (
        metric_cells.compute().nanmean().item()
    )
    log_dict[f"{mode}/pearson_genes"] = (
        metric_genes.compute().nanmean().item()
    )
    log_dict[f"{mode}/pearson"] = (
        log_dict[f"{mode}/pearson_cells"]
        + log_dict[f"{mode}/pearson_genes"]
    ) / 2.0
    for key in loss_dict:
        log_dict[f"{mode}/{key}_loss"] /= len(eval_loader.dataset)

    logging.info(
        f"{mode.capitalize()} Loss: {log_dict[f'{mode}/loss']:.4f}, "
        f"PC Cells: {log_dict[f'{mode}/pearson_cells']:.4f}, "
        f"PC Genes: {log_dict[f'{mode}/pearson_genes']:.4f}, "
        f"PC: {log_dict[f'{mode}/pearson']:.4f}"
    )
    wb_logger.log(log_dict)

    return log_dict[f"{mode}/loss"]

# ---------------------------------------------------------------------------- #
# -------------- TRAINING AND EVALUATION FOR ENSEMBLE MODEL ------------------ #
# ---------------------------------------------------------------------------- #

def fit_posthoc_variance_scale(
    model: torch.nn.Module,
    calibration_loader: torch.utils.data.DataLoader,
    device: torch.device = torch.device("cpu"),
    eps: float = 1e-8,
) -> float:
    """Fit one Gaussian variance temperature on validation residuals.

    For fixed means and raw total variances ``v``, the scalar minimizing
    Gaussian NLL for ``alpha * v`` is

        alpha = mean((y - mu) ** 2 / v).

    Raw (unscaled) model outputs are always used, so calling this function
    after successive epochs never compounds the previous epoch's scale.
    """
    model.eval()
    ratio_sum = torch.tensor(0.0, dtype=torch.float64, device=device)
    n_valid = 0

    with torch.no_grad():
        for inputs, targets in tqdm(
            calibration_loader,
            desc="Fitting variance scale",
        ):
            inputs, targets = inputs.to(device), targets.to(device)
            prediction, aleatoric_unc, epistemic_unc = (
                model.forward_uncalibrated(inputs)
            )
            raw_total_variance = (
                aleatoric_unc + epistemic_unc
            ).clamp_min(eps)
            squared_error = (targets - prediction) ** 2
            finite = (
                torch.isfinite(raw_total_variance)
                & torch.isfinite(squared_error)
            )
            if not finite.any():
                continue

            variance = raw_total_variance[finite].to(torch.float64)
            error = squared_error[finite].to(torch.float64)
            ratio_sum += torch.sum(error / variance)
            n_valid += int(finite.sum().item())

    if n_valid == 0:
        raise RuntimeError(
            "Cannot fit post-hoc variance scale: validation data contained "
            "no finite prediction/target pairs."
        )

    variance_scale = float((ratio_sum / n_valid).item())
    if not np.isfinite(variance_scale) or variance_scale < 0.0:
        raise RuntimeError(
            "Post-hoc variance fitting produced an invalid scale: "
            f"{variance_scale}."
        )
    # A perfect validation fit puts the Gaussian-NLL optimum at the boundary
    # alpha=0. Keep the stored predictive variance strictly positive.
    variance_scale = max(variance_scale, eps)
    model.set_variance_scale(variance_scale)

    return variance_scale


def train_ensemble_model(
    model: torch.nn.Module,
    train_loader: torch.utils.data.DataLoader,
    eval_loader: torch.utils.data.DataLoader,
    loss_dict: dict,
    loss_lambda_dict: dict,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None,
    wb_logger: WandBLogger,
    early_stopping: EarlyStopping,
    epochs: int = 10,
    device: torch.device = torch.device("cpu"),
    eval_calibration_loader: torch.utils.data.DataLoader | None = None,
) -> None:
    """Jointly train, fit validation variance scale, and evaluate each epoch."""

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

            if scheduler is not None:
                scheduler.step()
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

        variance_calibration_loader = (
            eval_calibration_loader
            if eval_calibration_loader is not None
            else eval_loader
        )
        variance_scale = fit_posthoc_variance_scale(
            model=model,
            calibration_loader=variance_calibration_loader,
            device=device,
        )
        logging.info(
            "Epoch %d post-hoc variance scale: %.6g.",
            epoch + 1,
            variance_scale,
        )

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
    targets. For a population dataset, that same per-individual dataset also
    supplies the empirical across-donor variance target used to evaluate the
    aleatoric head on unseen genes. If `calibration_loader` is None,
    calibration falls back to `eval_loader` (previous behaviour).
    """

    population_variance = None
    if calibration_loader is not None:
        population_dataset = calibration_loader.dataset
        mean_dataset = eval_loader.dataset
        if isinstance(
            population_dataset,
            ReferencePopulationMddDataset,
        ) and isinstance(mean_dataset, ReferencePopulationMddDataset):
            same_genes = np.array_equal(
                population_dataset.X_ensids.astype(str),
                mean_dataset.X_ensids.astype(str),
            )
            same_cell_types = (
                list(population_dataset.cell_types)
                == list(mean_dataset.cell_types)
            )
            sequential_genes = isinstance(
                eval_loader.sampler,
                torch.utils.data.SequentialSampler,
            )
            if (
                len(mean_dataset.individuals) == 1
                and same_genes
                and same_cell_types
                and sequential_genes
            ):
                target_variance, donor_counts = (
                    population_dataset.population_variance_targets()
                )
                population_variance = PopulationVarianceEvaluation(
                    target_variance=target_variance,
                    cell_types=list(population_dataset.cell_types),
                    donor_counts=donor_counts,
                    variance_scale=float(model.variance_scale.item()),
                )
            else:
                logging.warning(
                    "Skipping population-variance evaluation because the "
                    "population and population-mean datasets are not aligned "
                    "to exactly the same unseen genes and cell types, or the "
                    "evaluation loader is not sequential."
                )

    log_dict = {
        f"{mode}/epoch": epoch,
        f"{mode}/loss": 0.0,
        f"{mode}/pearson_cells": 0.0,
        f"{mode}/pearson_genes": 0.0,
        f"{mode}/pearson": 0.0,
        f"{mode}/aleatoric": 0.0, # should theoretically stay consistent
        f"{mode}/epistemic": 0.0, # should theoretically go down with training
        f"{mode}/variance_scale": float(model.variance_scale.item()),
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

            if population_variance is not None:
                population_variance.update(aleatoric_unc)

            aleatoric_uncertainties.append(aleatoric_unc.mean(dim=0)) # avg aleatoric uncertainty across cells
            epistemic_uncertainties.append(epistemic_unc.mean(dim=0)) # avg epistemic uncertainty across cells

            if calibration_loader is None:
                # no separate per-individual pass available; fall back to
                # calibrating against whatever targets eval_loader provides.
                # total predictive variance = aleatoric + epistemic
                calibration.update(prediction, aleatoric_unc + epistemic_unc, targets)

            # Post-hoc calibration is fitted to total predictive variance,
            # therefore GNLL evaluation must use that same quantity. The
            # other losses only consume outputs[..., 0] (the mean).
            total_unc = aleatoric_unc + epistemic_unc
            outputs = torch.stack(
                [prediction, total_unc],
                dim=2,
            ) # shape (batch_size, output_dim, 2)
            composite_loss = torch.tensor(0.0, device=device)
            for key in loss_dict.keys():
                loss = loss_dict[key](outputs, targets) * loss_lambda_dict[key]
                log_dict[f"{mode}/{key}_loss"] += (
                    loss.item() * targets.shape[0]
                )
                composite_loss += loss

            log_dict[f"{mode}/loss"] += (
                composite_loss.item() * targets.shape[0]
            )
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

    log_dict[f"{mode}/loss"] /= len(eval_loader.dataset)
    log_dict[f"{mode}/pearson_cells"] = metric_cells.compute().nanmean().item()
    log_dict[f"{mode}/pearson_genes"] = metric_genes.compute().nanmean().item()
    log_dict[f"{mode}/pearson"] = (log_dict[f"{mode}/pearson_cells"] + log_dict[f"{mode}/pearson_genes"]) / 2.0
    log_dict[f"{mode}/aleatoric"] = torch.mean(torch.stack(aleatoric_uncertainties)).item()
    log_dict[f"{mode}/epistemic"] = torch.mean(torch.stack(epistemic_uncertainties)).item()
    log_dict[f"{mode}/ence"] = calibration.compute_ence()
    log_dict[f"{mode}/uncertainty_error_spearman"] = calibration.compute_spearman()

    if population_variance is not None:
        population_variance_scalars, _ = population_variance.compute()
        log_dict[f"{mode}/population_aleatoric_pearson"] = (
            population_variance_scalars["pearson_macro"]
        )

    for key in loss_dict.keys():
        log_dict[f"{mode}/{key}_loss"] /= len(eval_loader.dataset)
    
    logging.info(f"{mode.capitalize()} Loss: {log_dict[f'{mode}/loss']:.4f}, PC Cells: {metric_cells.compute().nanmean():.4f}, PC Genes: {metric_genes.compute().nanmean():.4f}, PC: {log_dict[f'{mode}/pearson']:.4f}")
    logging.info(f"{mode.capitalize()} ENCE: {log_dict[f'{mode}/ence']:.4f}, Uncertainty-Error Spearman: {log_dict[f'{mode}/uncertainty_error_spearman']:.4f}")
    if population_variance is not None:
        logging.info(
            "%s aleatoric vs empirical across-donor variance on unseen genes: "
            "across-cell-type Pearson=%.4f.",
            mode.capitalize(),
            log_dict[f"{mode}/population_aleatoric_pearson"],
        )
    
    wb_logger.log(log_dict)

    # render calibration boxplot only when logging
    if wb_logger.enabled:
        fig = calibration.make_boxplot(title=f"{mode} uncertainty")
        if fig is not None:
            wb_logger.log({f"{mode}/mse_vs_uncertainty": wandb.Image(fig)})
            plt.close(fig)

    return log_dict[f"{mode}/loss"]
