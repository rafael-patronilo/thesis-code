from . import SplitDataset
import logging
from typing import Optional
import pandas as pd
import torch
from torch.utils.data import TensorDataset

logger = logging.getLogger(__name__)

class CSVDataset(SplitDataset):
    def __init__(self, 
                 path: str, 
                 target: list[str], 
                 features: Optional[list[str]] = None, 
                 split: float | tuple[float, float] = (0.7, 0.15),
                 shuffle: bool = True,
                 random_state = None):
        super().__init__()
        self.path = path
        self.target = target
        self.features = features
        self.splits : tuple[float, float]
        if type(split) == tuple:
            self.splits = split
        else:
            self.splits = (split, 1.0 - split) # type: ignore
        self.shuffle = shuffle
        self.random_state = random_state

    
    def _load(self):
        if self.loaded:
            return
        logger.info(f"Loading CSV dataset from {self.path}")
        self.data = pd.read_csv(self.path)
        if self.features is None:
            self.features = [col for col in self.data.columns if col not in self.target]
        if self.shuffle:
            self.data = self.data.sample(frac=1, random_state=self.random_state).reset_index(drop=True)
        train_bound = int(len(self.data) * self.splits[0])
        val_bound = int(len(self.data) * (self.splits[0] + self.splits[1]))
        train_rows = self.data.iloc[:train_bound]
        val_rows = self.data.iloc[train_bound:val_bound]
        test_rows = self.data.iloc[val_bound:]
        def torchify(rows):
            if rows is None or len(rows) == 0:
                return None
            return TensorDataset(
                torch.tensor(rows[self.features].values).float(), 
                torch.tensor(rows[self.target].values).float()
            )
        self.train_data=torchify(train_rows)
        self.val_data=torchify(val_rows)
        self.test_data=torchify(test_rows)
        self.loaded = True
