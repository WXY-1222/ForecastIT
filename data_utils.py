import pickle
import numpy as np
from typing import Dict, Tuple
import torch
import torch.nn as nn
from torch.utils.data import Dataset

class StandardScaler:
    def __init__(self, mean: float, std: float):
        self.mean = float(mean)
        self.std = float(std if std > 0 else 1.0)
    def transform(self, x):
        return (x - self.mean) / self.std
    def inverse(self, x):
        return x * self.std + self.mean

def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)

# -----------------------------
# Dataset / Loader
# -----------------------------
class SeqDataset(Dataset):
    def __init__(self, data_npz, index_triplets, scaler: StandardScaler, steps: Tuple[int, int]):
        self._data = data_npz["data"]  # (T_all, N, C)
        self._triplets = index_triplets  # (M, 3)
        self.scaler = scaler
        self.in_steps, self.out_steps = steps

    def __len__(self):
        return self._triplets.shape[0]

    def __getitem__(self, idx):
        xs, xe, ye = self._triplets[idx]
        X = self._data[xs:xe]           # (Tin, N, C)
        Y = self._data[xe:ye, :, 0]     # (Tout, N) 

        
        X = X.copy()
        X[..., 0] = self.scaler.transform(X[..., 0])

        
        X = torch.from_numpy(X).float().unsqueeze(0)
        Y = torch.from_numpy(Y).float().unsqueeze(0)

        return X[0], Y[0]


def load_indices(idx_npz) -> Dict[str, np.ndarray]:
    keys = list(idx_npz.keys())

    cand = {
        "train": ["train", "train_index", "train_idx"],
        "val": ["val", "valid", "val_index", "valid_index"],
        "test": ["test", "test_index"]
    }
    out = {}
    for split, names in cand.items():
        for n in names:
            if n in keys:
                out[split] = idx_npz[n]
                break
        if split not in out:
            raise KeyError(f"index.npz 缺少 {split} 索引（可用键：{keys}）")
    return out