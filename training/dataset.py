import torch
from torch.utils.data import Dataset

import numpy as np
import pandas as pd
from pathlib import Path

from scipy.stats import skew

class MddDataset(Dataset):
    
    def __init__(self, X_feats: Path|np.ndarray, X_ensids: Path|np.ndarray, X_chroms: Path|np.ndarray, y: Path|pd.DataFrame, normalize: bool = False):
        """
        Args:
            X_feats (Path|np.ndarray):  Path to input features file (npy) or numpy array:
                                        Contains the pLM/gLM embeddings.
            X_ensids (Path|np.ndarray): Path to input ensids (for features) file (npy) or numpy array:
                                        Contains the ensids corresponding to the rows in X_feats.
            X_chroms (Path|np.ndarray): Path to input chroms (for features) file (npy) or numpy array:
                                        Contains the chromosomes corresponding to the rows in X_feats.
            y (Path|pd.DataFrame):      Path to target labels file (csv) or pandas DataFrame:
                                        2d array; (rows: cell-types, columns: ensid)
            normalize (bool):           Whether to log-transform the target labels.
        """
        if isinstance(X_feats, Path) and isinstance(X_ensids, Path) and \
           isinstance(X_chroms, Path) and isinstance(y, Path):
            self.X_feats  = np.load(X_feats, mmap_mode='r')
            self.X_ensids = np.load(X_ensids, allow_pickle=True)
            self.X_chroms = np.load(X_chroms, allow_pickle=True).astype(str)
            self.y        = pd.read_csv(y, index_col=0)
        else:
            self.X_feats  = X_feats
            self.X_ensids = X_ensids
            self.X_chroms = X_chroms
            self.y        = y

        self.normalize = normalize
        self.norm_features = None
    
    def split_by_chromosome(self, chrom: list[str]) -> Dataset:
        """Return a new MddDataset containing only data from the specified chromosomes."""
        mask = np.isin(self.X_chroms, chrom)
        X_feats_chrom  = self.X_feats[mask]
        X_ensids_chrom = self.X_ensids[mask]
        X_chroms_chrom = self.X_chroms[mask]
        return MddDataset(X_feats_chrom, X_ensids_chrom, X_chroms_chrom, self.y, normalize=self.normalize)

    def apply_feature_log_transform(self, threshold=1.0) -> None:
        """Apply log-transform to input features."""
        self.norm_features = torch.zeros(self.X_feats.shape[1])
        for i in range(self.X_feats.shape[1]):
            col = self.X_feats[:, i]
            # if all values are non-negative and skewed, apply log-transform
            if np.all(col >= 0) and skew(col) > threshold:
                self.norm_features[i] = 1.0
    
    def __len__(self) -> int:
        return self.X_ensids.shape[0]
    
    def __getitem__(self, idx) -> tuple[torch.Tensor, torch.Tensor]:
        ensid = self.X_ensids[idx]
        if ensid not in self.y.columns:
            raise KeyError(f"Ensid {ensid} not found in target labels.")
        x = torch.from_numpy(self.X_feats[idx])
        y = torch.from_numpy(self.y[ensid].values)
        if self.norm_features is not None:
            x = x.clone()
            x[self.norm_features == 1.0] = torch.log(1 + x[self.norm_features == 1.0])
        if self.normalize:
            y = torch.log(1 + y)
        return x, y

if __name__ == "__main__":
    # example usage
    dataset = MddDataset(X_feats=Path("X_sub.features.npy"), X_ensids=Path("X_sub.ensids.npy"), X_chroms=Path("X_sub.chroms.npy"), y=Path("y_sub.csv"))
    print(f"Dataset size: {len(dataset)}")
    for i in range(len(dataset)):
        input_data, label = dataset[i]
        print(f"Input shape at index {i}: {input_data.shape}, Label: {label.shape}")
    print(dataset.y.head())
    print("Splitting dataset by chromosomes 1 and 2...")
    new_dataset = dataset.split_by_chromosome(['1', '2'])
    print(f"New dataset size (chromosomes 1 and 2): {len(new_dataset)}")
    print(new_dataset.y.head())