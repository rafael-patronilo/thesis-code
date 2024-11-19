from typing import Callable
from core.datasets import CollumnReferences, SplitDataset
from torch.utils.data import Dataset, IterableDataset

class SplitDatasetWrapper(SplitDataset):
    def __init__(self, inner : SplitDataset) -> None:
        super().__init__()
        self.inner : SplitDataset = inner
    
    def get_shape(self):
        return self.inner.get_shape()

    def get_metric(self, metric : str):
        return self.inner.get_metric(metric)

    def get_collumn_references(self) -> CollumnReferences:
        return self.inner.get_collumn_references()

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

class ItemMapper(SplitDatasetWrapper):
    def __init__(self, inner: SplitDataset, mapper : Callable) -> None:
        super().__init__(inner)
        self.mapper = mapper
    
    def get_collumn_references(self) -> CollumnReferences:
        sample = self.inner.get_collumn_references().as_sample()
        mapped_sample = self.mapper(sample)
        return CollumnReferences.from_sample(mapped_sample)
    
    def _wrap_torch_dataset(self, dataset : Dataset) -> Dataset:
        if isinstance(dataset, IterableDataset):
            wrapped = _IterableDatasetWrapper(dataset, self.mapper)
        else:
            wrapped = _DatasetWrapper(dataset, self.mapper)
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


class SelectCols(ItemMapper):
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
        super().__init__(dataset, self._select_mapper)
    
    def _select_mapper(self, sample):
        x, y = sample
        if self.select_x is not None:
            x = x[self.select_x]
        if self.select_y is not None:
            y = y[self.select_y]
        return x, y

class ForAutoencoder(ItemMapper):
    @classmethod
    def _autoencoder_mapper(cls, sample):
        x, _ = sample
        return x, x

    def __init__(self, dataset : SplitDataset):
        super().__init__(dataset, self._autoencoder_mapper)