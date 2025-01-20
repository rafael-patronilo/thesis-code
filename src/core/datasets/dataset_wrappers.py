from abc import ABC
from math import log
from typing import Callable, Literal, Protocol, Sequence, Self

from pandas.io.sql import abstractmethod
import torch
from core.datasets import ColumnReferences, SplitDataset
from torch.utils.data import Dataset, IterableDataset

import logging

logger = logging.getLogger(__name__)

class SplitDatasetWrapper(SplitDataset):
    def __init__(self, inner : SplitDataset) -> None:
        super().__init__()
        self.inner : SplitDataset = inner
    
    def get_shape(self):
        return self.inner.get_shape()

    def get_metric(self, metric : str):
        return self.inner.get_metric(metric)

    def get_column_references(self, load_if_needed = True) -> ColumnReferences:
        return self.inner.get_column_references(load_if_needed)

    def for_training(self) -> Dataset:
        return self.inner.for_training()

    def for_validation(self) -> Dataset:
        return self.inner.for_validation()

    def for_testing(self) -> Dataset | None:
        return self.inner.for_testing()
    
    def unwrap(self) -> SplitDataset:
        innermost = self.inner
        while isinstance(innermost, SplitDatasetWrapper):
            innermost = innermost.inner
        return innermost
    
    def __repr__(self):
        fields = "".join(f", {k}={v}" for k, v in self.__dict__.items() if k != 'inner')
        return f"{self.__class__.__name__}({self.inner}{fields})"
    
    @classmethod
    def of(cls, dataset : Dataset | SplitDataset) -> SplitDataset:
        if isinstance(dataset, cls):
            return dataset.unwrap()
        else:
            return SplitDataset.of(dataset)

def unwrap(dataset : SplitDataset) -> SplitDataset:
    if isinstance(dataset, SplitDatasetWrapper):
        return dataset.unwrap()
    else:
        return dataset


class _DatasetWrapper(Dataset):
    def __init__(self, dataset : Dataset, mapper : Callable) -> None:
        super().__init__()
        self.inner : Dataset = dataset
        self.mapper = mapper
    
    def __len__(self):
        return len(self.inner) # type: ignore

    def __getitem__(self, index):
        return self.mapper(self.inner.__getitem__(index))

class _IterableDatasetWrapper(IterableDataset):
    def __init__(self, dataset : IterableDataset, mapper : Callable) -> None:
        super().__init__()
        self.inner : IterableDataset = dataset
        self.mapper = mapper
    
    def __len__(self):
        return len(self.inner) # type: ignore

    def __iter__(self):
        for sample in self.inner:
            yield self.mapper(sample)

class ItemMapperABC(SplitDatasetWrapper, ABC):
    def __init__(self, inner: SplitDataset) -> None:
        super().__init__(inner)

    @abstractmethod
    def _mapper(self, sample : tuple[torch.Tensor, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        pass

    @abstractmethod
    def _cols_mapper(self, sample : tuple[list[str], list[str]]) -> tuple[list[str], list[str]]:
        pass

    def get_column_references(self, load_if_needed = True) -> ColumnReferences:
        sample = self.inner.get_column_references(load_if_needed).as_sample()
        mapped_sample = self._cols_mapper(sample)
        return ColumnReferences.from_sample(mapped_sample)
    
    def _wrap_torch_dataset(self, dataset : Dataset) -> Dataset:
        if isinstance(dataset, IterableDataset):
            wrapped = _IterableDatasetWrapper(dataset, self._mapper)
        else:
            wrapped = _DatasetWrapper(dataset, self._mapper)
        self._attach_self(wrapped)
        return wrapped
    
    def for_training(self) -> Dataset:
        return self._wrap_torch_dataset(self.inner.for_training())
    
    def for_validation(self) -> Dataset:
        return self._wrap_torch_dataset(self.inner.for_validation())
    
    def for_testing(self) -> Dataset | None:
        dataset = self.inner.for_testing()
        if dataset is None:
            return None
        return self._wrap_torch_dataset(dataset)

class SelectCols(ItemMapperABC):
    def __init__(
            self, 
            dataset : SplitDataset, 
            select_x : int | list[int] | None = None,
            select_y : int | list[int] | None = None 
        ):
        if isinstance(select_x, int):
            select_x = [select_x]
        if isinstance(select_y, int):
            select_y = [select_y]
        self.select_x : list | None = select_x
        self.select_y : list | None = select_y
        super().__init__(dataset)
    
    def _seq_mapper[T](self, sample : tuple[T, T]) -> tuple[T, T]:
        x, y = sample
        if self.select_x is not None:
            x = x[self.select_x] # type: ignore
        if self.select_y is not None:
            y = y[self.select_y] # type: ignore
        return x, y

    def _mapper(self, sample : tuple[torch.Tensor, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        return self._seq_mapper(sample)

    def _cols_mapper(self, sample : tuple[list[str], list[str]]) -> tuple[list[str], list[str]]:
        return self._seq_mapper(sample)


class ForAutoencoder(ItemMapperABC):
    @classmethod
    def _seq_mapper[T](cls, sample : tuple[T, T]) -> tuple[T, T]:
        x, _ = sample
        return x, x

    def _mapper(self, sample: tuple[torch.Tensor, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        return self._seq_mapper(sample)

    def _cols_mapper(self, sample: tuple[list[str], list[str]]) -> tuple[list[str], list[str]]:
        return self._seq_mapper(sample)

    def __init__(self, dataset : SplitDataset):
        super().__init__(dataset)

class ConcatConst(ItemMapperABC):
    def __init__(self, dataset : SplitDataset, value : torch.Tensor | float, target_tensor : Literal['x', 'y']):
        if not isinstance(value, torch.Tensor):
            value = torch.tensor([value])
        self.value = value.to('cpu')
        self.target_tensor = target_tensor
        super().__init__(dataset)

    def _cols_mapper(self, sample : tuple[list[str], list[str]]) -> tuple[list[str], list[str]]:
        features, labels = sample
        value_list = [self.value.item()] if self.value.numel() == 1 else self.value.tolist()
        if self.target_tensor == 'x':
            features.extend([f"const_{x}" for x in value_list])
        else:
            labels.extend([f"const_{x}" for x in value_list])
        return features, labels

    def _mapper(self, sample):
        x, y = sample
        if self.target_tensor == 'x':
            x = torch.hstack((x, self.value))
        else:
            y = torch.hstack((y, self.value))
        return x, y