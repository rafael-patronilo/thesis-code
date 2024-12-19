from typing import Optional, NamedTuple, Any
from core.util.typing import TorchDataset
import logging
from torch.utils.data import Dataset, IterableDataset

logger = logging.getLogger(__name__)

ColumnSubReferences = NamedTuple("ColumnSubReferences", [("names_to_column", dict[str, int]), ("columns_to_names", list[str])])
class ColumnReferences(NamedTuple):
    features: ColumnSubReferences 
    labels : ColumnSubReferences

    def as_sample(self) -> tuple[list[str], list[str]]:
        return (self.features.columns_to_names, self.labels.columns_to_names)
    
    def get_feature_indices(self, columns : list[str]) -> list[int]:
        """Get indices for a list of feature column names. Respects argument order

        Args:
            columns (list[str]): Names of columns

        Returns:
            list[int]: Indices of columns, in the same order as the respective name in `columns`
        """
        return [self.features.names_to_column[column] for column in columns]
    
    def get_label_indices(self, columns : list[str]) -> list[int]:
        """Get indices for a list of label column names. Respects argument order
        
        Args:
            columns (list[str]): Names of columns

        Returns:
            list[int]: Indices of columns, in the same order as the respective name in `columns`
        """
        return [self.labels.names_to_column[column] for column in columns]
    
    @classmethod
    def from_sample(cls, sample : tuple[list[str], list[str]]):
        return cls(
            ColumnSubReferences(
                {name: idx for idx, name in enumerate(sample[0])},
                sample[0]
            ),
            ColumnSubReferences(
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
            column_references : Optional[ColumnReferences] = None):
        self.train_data = train_data
        self.val_data = val_data
        self.test_data = test_data
        self.loaded = False
        self.column_references = column_references

    def __reduce__(self) -> str | tuple[Any, ...]:
        if not hasattr(self, 'name'):
            return (self.__class__, (self.train_data, self.val_data, self.test_data, self.column_references))
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
    
    def get_column_references(self) -> ColumnReferences:
        if self.column_references is None:
            raise ValueError("Column references not available")
        return self.column_references

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
        Useful if some columns are hidden during training.
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
        
    @classmethod
    def of(cls, dataset : 'Dataset | SplitDataset') -> 'SplitDataset':
        if isinstance(dataset, cls):
            return dataset
        elif hasattr(dataset, 'dataset'):
            return cls.of(getattr(dataset, 'dataset'))
        elif isinstance(dataset, SplitDataset):
            return dataset.__class__.of(dataset)
        else:
            raise ValueError(f"Cannot convert {dataset} to SplitDataset")




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