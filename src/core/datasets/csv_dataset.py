from . import SplitDataset, CollumnReferences, CollumnSubReferences
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
                 splits: float | tuple[float, float] = (0.7, 0.15),
                 shuffle: bool = True,
                 random_state = None):
        super().__init__()
        self.path = path
        self.target = target
        self.features = features
        self.splits = self._cast_splits(splits)
        self.shuffle = shuffle
        self.random_state = random_state
        if self.features is not None:
            self._set_collumn_references(features, target) # type: ignore

    def _set_collumn_references(self, features: list[str], target: list[str]):
        self.collumn_references = CollumnReferences(
            CollumnSubReferences(
                {name: idx for idx, name in enumerate(features)},
                features
            ),
            CollumnSubReferences(
                {name: idx for idx, name in enumerate(target)},
                target
            )
        )
    
    def _load(self):
        if self.loaded:
            return
        logger.info(f"Loading CSV dataset from {self.path}")
        self.data = pd.read_csv(self.path)
        if self.features is None:
            self.features = [col for col in self.data.columns if col not in self.target]
            self._set_collumn_references(self.features, self.target)
        if self.shuffle:
            self.data = self.data.sample(frac=1, random_state=self.random_state).reset_index(drop=True)
        train_bound, val_bound = self._split(len(self.data), self.splits)
        train_rows = self.data.iloc[:train_bound]
        val_rows = self.data.iloc[train_bound:val_bound]
        test_rows = self.data.iloc[val_bound:]

        logger.debug(
f"""Dataset ({len(self.data):_} samples) split:
Training: \t [0, {train_bound:_}[ \t ({len(train_rows):_} samples, {self.splits[0] * 100}%)
Validation: \t [{train_bound:_}, {val_bound}[ \t ({len(val_rows):_} samples, {self.splits[1] * 100}%)
Testing: \t [{val_bound:_}, {len(self.data):_}[ \t ({len(test_rows):_} samples, {(1 - self.splits[0] - self.splits[1]) * 100}%)
Num features: \t {len(self.features)}
Num target: \t {len(self.target)}
Shuffle: \t {self.shuffle}
Seed: \t {self.random_state}""")

        def torchify(rows):
            if rows is None or len(rows) == 0:
                logger.debug("No data available")
                return None
            logger.debug(f"Selected {len(rows)} samples")
            features = rows[self.features]
            target = rows[self.target]
            logger.debug(f"Feature columns: {features.columns}")
            logger.debug(f"Target columns: {target.columns}")
            features = torch.tensor(features.values).float()
            target = torch.tensor(target.values).float()
            return TensorDataset(features, target)
        logger.debug("Selecting training data")
        self.train_data=torchify(train_rows)
        logger.debug("Selecting validation data")
        self.val_data=torchify(val_rows)
        logger.debug("Selecting testing data")
        self.test_data=torchify(test_rows)
        self.loaded = True
