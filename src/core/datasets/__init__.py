from typing import Optional, NamedTuple
from abc import ABC, abstractmethod
from torch.utils.data import Dataset, IterableDataset


CollumnSubReferences = NamedTuple("CollumnSubReferences", [("names_to_collumn", dict[str, int]), ("collumns_to_names", list[str])])
CollumnReferences = NamedTuple("CollumnReferences", [("features", CollumnSubReferences), ("labels", CollumnSubReferences)])

class SplitDataset(ABC):
    
    def __init__(self, train_data = None, val_data = None, test_data = None):
        self.train_data = train_data
        self.val_data = val_data
        self.test_data = test_data
        self.loaded = False
        self.collumn_references : Optional[CollumnReferences] = None

    def _load(self):
        pass

    @classmethod
    def _cast_splits(cls, splits : float | tuple[float, float]) -> tuple[float, float]:
        if type(splits) == tuple:
            return splits
        else:
            return (splits, 1.0 - splits) # type: ignore

    @classmethod
    def _split(cls, size : int, splits : tuple[float, float]) -> tuple[int, int]:
        train_bound = int(size * splits[0])
        val_bound = int(size * (splits[0] + splits[1]))
        return train_bound, val_bound

    def for_training(self) -> Dataset | IterableDataset:
        self._load()
        if self.train_data is None:
            raise ValueError("No training data available")
        return self.train_data

    def for_validation(self) -> Dataset | IterableDataset:
        self._load()
        if self.val_data is None:
            raise ValueError("No validation data available")
        return self.val_data

    def for_testing(self) -> Optional[Dataset | IterableDataset]:
        self._load()
        return self.test_data
    
    def has_testing(self) -> bool:
        return self.test_data is not None



from .csv_dataset import CSVDataset
from .binary_generator import BinaryGeneratorBuilder
from .random_dataset import RandomDataset

dataset_registry : dict[str, SplitDataset] = {}