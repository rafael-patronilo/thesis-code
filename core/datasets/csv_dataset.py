from . import SplitDataset
from typing import Optional
import pandas as pd
import torch
from torch.utils.data import TensorDataset

class CSVDataset(SplitDataset):
    def __init__(self, 
                 path: str, 
                 target: list[str], 
                 features: Optional[list[str]] = None, 
                 split: float | tuple[float, float] = (0.7, 0.15),
                 suffle: bool = True,
                 random_state = None):
        self.path = path
        self.target = target
        self.data = pd.read_csv(path)
        self.features : list[str]
        if features is None:
            self.features = [col for col in self.data.columns if col not in self.target]
        else:
            self.features = features
        if suffle:
            self.data = self.data.sample(frac=1, random_state=random_state).reset_index(drop=True)
        splits : tuple[float, float]
        if split is float:
            splits = (splits, 1 - splits)
        elif split is tuple:
            splits = split
        else:
            raise ValueError("Split must be a float or a tuple of floats")
        train_bound = int(len(self.data) * splits[0])
        val_bound = int(len(self.data) * (splits[0] + splits[1]))
        train_rows = self.data.iloc[:train_bound]
        val_rows = self.data.iloc[train_bound:val_bound]
        test_rows = self.data.iloc[int(len(self.data) * split):]
        def torchify(rows):
            if rows is None or len(rows) == 0:
                return None
            return TensorDataset(
                torch.tensor(rows[self.features].values).float(), 
                torch.tensor(rows[self.target].values).float()
            )
        super().__init__(
            train_data=torchify(train_rows), 
            val_data=torchify(val_rows), 
            test_data=torchify(test_rows)
        )