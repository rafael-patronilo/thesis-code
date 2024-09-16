from typing import Optional
from abc import ABC, abstractmethod
from torch.utils.data import Dataset

class SplitDataset(ABC):
    def __init__(self, train_data = None, val_data = None, test_data = None):
        self.train_data = train_data
        self.val_data = val_data
        self.test_data = test_data
        self.loaded = False

    def _load(self):
        pass

    def for_training(self) -> Dataset:
        self._load()
        if self.train_data is None:
            raise ValueError("No training data available")
        return self.train_data

    def for_validation(self) -> Dataset:
        self._load()
        if self.val_data is None:
            raise ValueError("No validation data available")
        return self.val_data

    def for_testing(self) -> Optional[Dataset]:
        self._load()
        return self.test_data
    
    def has_testing(self) -> bool:
        return self.test_data is not None

from .csv_dataset import CSVDataset

dataset_registry : dict[str, SplitDataset] = {}