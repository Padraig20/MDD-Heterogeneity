from dataset import MddDataset
from copy import deepcopy
import torch

# taken from scPrediXcan tutorial
# https://github.com/hakyimlab/scPrediXcan/blob/master/Scripts/ctPred/Tutorial.ipynb
train_set = ["1", "10", "13", "15", "16", "17", "18", "19", "2", "21", "22", "3", "4", "6", "8", "9", "X", "Y"]
val_set = ["11", "14", "7"]
test_set = ["12", "20", "5"]

def get_train_test_dataset(dataset: MddDataset):
    """Load dataset and split into train, val and test sets."""
    train_dataset = dataset.split_by_chromosome(train_set)
    val_dataset   = dataset.split_by_chromosome(val_set)
    test_dataset  = dataset.split_by_chromosome(test_set)
    return train_dataset, val_dataset, test_dataset

class EarlyStopping:
    def __init__(self, patience: int = 20, min_delta: float = 1e-6, mode: str = "min"):
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