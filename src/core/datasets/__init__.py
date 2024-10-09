from typing import Optional, NamedTuple
from torch.utils.data import Dataset, IterableDataset
import logging
logger = logging.getLogger(__name__)

CollumnSubReferences = NamedTuple("CollumnSubReferences", [("names_to_collumn", dict[str, int]), ("collumns_to_names", list[str])])
CollumnReferences = NamedTuple("CollumnReferences", [("features", CollumnSubReferences), ("labels", CollumnSubReferences)])

class SplitDataset:
    
    def __init__(
            self, 
            train_data = None, 
            val_data = None, 
            test_data = None,
            collumn_references : Optional[CollumnReferences] = None):
        self.train_data = train_data
        self.val_data = val_data
        self.test_data = test_data
        self.loaded = False
        self.collumn_references = collumn_references

    def get_metric(self, metric : str):
        raise NotImplementedError("Metric retrieval not implemented")

    def _load(self):
        logger.debug("Data loading not implemented")
        self.loaded = True

    @classmethod
    def _cast_splits(cls, splits : float | tuple[float, float]) -> tuple[float, float]:
        if type(splits) == tuple:
            return splits
        else:
            return (splits, 1.0 - splits) # type: ignore
    
    def get_collumn_references(self) -> CollumnReferences:
        if self.collumn_references is None:
            raise ValueError("Collumn references not available")
        return self.collumn_references

    @classmethod
    def _split(cls, size : int, splits : tuple[float, float]) -> tuple[int, int]:
        train_bound = int(size * splits[0])
        val_bound = int(size * (splits[0] + splits[1]))
        return train_bound, val_bound
    
    def _attach_self(self, data):
        if not hasattr(data, 'dataset'):
            data.dataset = self

    def for_training(self) -> Dataset | IterableDataset:
        self._load()
        if self.train_data is None:
            raise ValueError("No training data available")
        self._attach_self(self.train_data)
        return self.train_data

    def for_validation(self) -> Dataset | IterableDataset:
        self._load()
        if self.val_data is None:
            raise ValueError("No validation data available")
        self._attach_self(self.val_data)
        return self.val_data

    def for_testing(self) -> Optional[Dataset | IterableDataset]:
        self._load()
        if not self.has_testing():
            return None
        self._attach_self(self.test_data)
        return self.test_data
    
    def has_testing(self) -> bool:
        return hasattr(self, 'test_data') and self.test_data is not None



from .csv_dataset import CSVDataset
from . import binary_generator
from .random_dataset import RandomDataset

dataset_registry : dict[str, SplitDataset] = {}