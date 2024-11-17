from typing import Optional, NamedTuple, Any
from core.util.typing import TorchDataset
import logging
from torch.utils.data import Dataset, IterableDataset

logger = logging.getLogger(__name__)

CollumnSubReferences = NamedTuple("CollumnSubReferences", [("names_to_collumn", dict[str, int]), ("collumns_to_names", list[str])])
class CollumnReferences(NamedTuple):
    features: CollumnSubReferences 
    labels : CollumnSubReferences

    def as_sample(self) -> tuple[list[str], list[str]]:
        return (self.features.collumns_to_names, self.labels.collumns_to_names)
    
    def get_feature_indices(self, collumns : list[str]) -> list[int]:
        """Get indices for a list of feature collumn names. Respects argument order

        Args:
            collumns (list[str]): Names of collumns

        Returns:
            list[int]: Indices of collumns, in the same order as the respective name in `collumns`
        """
        return [self.features.names_to_collumn[collumn] for collumn in collumns]
    
    def get_label_indices(self, collumns : list[str]) -> list[int]:
        """Get indices for a list of label collumn names. Respects argument order
        
        Args:
            collumns (list[str]): Names of collumns

        Returns:
            list[int]: Indices of collumns, in the same order as the respective name in `collumns`
        """
        return [self.labels.names_to_collumn[collumn] for collumn in collumns]
    
    @classmethod
    def from_sample(cls, sample : tuple[list[str], list[str]]):
        return cls(
            CollumnSubReferences(
                {name: idx for idx, name in enumerate(sample[0])},
                sample[0]
            ),
            CollumnSubReferences(
                {name: idx for idx, name in enumerate(sample[1])},
                sample[1]
            )
        )

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
        def get_shape_rec(shape):
            if isinstance(shape, tuple):
                return tuple(get_shape_rec(sub_shape) for sub_shape in shape)
            elif isinstance(shape, list):
                return [get_shape_rec(sub_shape) for sub_shape in shape]
            elif isinstance(shape, dict):
                return {key: get_shape_rec(sub_shape) for key, sub_shape in shape.items()}
            else:
                return shape.shape
        if isinstance(dataset, IterableDataset):
            sample = next(iter(dataset))
        else:
            sample = dataset[0]
        return get_shape_rec(sample)

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
    
    def __repr__(self) -> str:
        def try_len(dataset):
            if dataset is None:
                return '0'
            try:
                return f"{len(dataset):_}"
            except:
                return 'Unknown'
        sizes = (
            f"\tTraining: {try_len(self.for_training())} samples\n"
            f"\tValidation: {try_len(self.for_validation())} samples\n"
            f"\tTesting: {try_len(self.for_testing())} samples\n"
        )
        if hasattr(self, 'name'):
            return (
                f"{self.__class__.__name__}(\n"
                f"\t{getattr(self, 'name')}\n{sizes})"
            )
        else:
            return f"{self.__class__.__name__}(\n{sizes})"



from .csv_dataset import CSVDataset
from .csv_img_dataset import CSVImageDataset
from . import binary_generator
from .random_dataset import RandomDataset


_dataset_registry : dict[str, SplitDataset] = {}

def get_dataset(name : str) -> SplitDataset:
    if name not in _dataset_registry:
        raise ValueError(f"Dataset {name} not found")
    return _dataset_registry[name]

def register_datasets(**kwargs):
    for name, dataset in kwargs.items():
        dataset.name = name
        _dataset_registry[name] = dataset