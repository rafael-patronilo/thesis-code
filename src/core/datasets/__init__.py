from typing import Optional, NamedTuple, Any
from core.util.typing import TorchDataset
import logging
from torch.utils.data import Dataset, IterableDataset

logger = logging.getLogger(__name__)

CollumnSubReferences = NamedTuple("CollumnSubReferences", [("names_to_collumn", dict[str, int]), ("collumns_to_names", list[str])])
class CollumnReferences(NamedTuple):
    features: CollumnSubReferences 
    labels : CollumnSubReferences
    eval_only_collumns : Optional[list[str]] = None

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

    def __reduce__(self) -> str | tuple[Any, ...]:
        if not hasattr(self, 'name'):
            return (self.__class__, (self.train_data, self.val_data, self.test_data, self.collumn_references))
        else:
            return (get_dataset, (getattr(self, 'name'),))

    def get_shape(self):
        dataset = self.for_training()
        if isinstance(dataset, IterableDataset):
            sample = next(iter(dataset))
        else:
            sample = dataset[0]
        return sample.shape

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

    def for_training(self) -> Dataset:
        self._load()
        if self.train_data is None:
            raise ValueError("No training data available")
        self._attach_self(self.train_data)
        return self.train_data
    
    def for_training_eval(self) -> TorchDataset:
        """Training set prepared for calculating metrics.
        Useful if some collumns are hidden during training.
        Default implementation is the same as for_training()

        Returns:
            TorchDataset: the training set
        """
        return self.for_training()

    def for_validation(self) -> Dataset:
        self._load()
        if self.val_data is None:
            raise ValueError("No validation data available")
        self._attach_self(self.val_data)
        return self.val_data

    def for_testing(self) -> Optional[Dataset]:
        self._load()
        if not self.has_testing():
            return None
        self._attach_self(self.test_data)
        return self.test_data
    
    def has_testing(self) -> bool:
        return hasattr(self, 'test_data') and self.test_data is not None
    
    def __str__(self) -> str:
        if hasattr(self, 'name'):
            return f"{self.__class__.__name__}({getattr(self, 'name')})"
        else:
            return f"{self.__class__.__name__}()"



from .csv_dataset import CSVDataset
from .csv_img_dataset import CSVImageDataset
from . import binary_generator
from .random_dataset import RandomDataset
from .autoencoder_dataset import AutoencoderDataset

_dataset_registry : dict[str, SplitDataset] = {}

def get_dataset(name : str) -> SplitDataset:
    if name not in _dataset_registry:
        raise ValueError(f"Dataset {name} not found")
    return _dataset_registry[name]

def register_datasets(**kwargs):
    for name, dataset in kwargs.items():
        dataset.name = name
        _dataset_registry[name] = dataset