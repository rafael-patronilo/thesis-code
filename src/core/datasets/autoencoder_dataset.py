from core.datasets import SplitDataset, CollumnReferences
from torch.utils.data import Dataset, IterableDataset

class _DatasetWrapper(Dataset):
    def __init__(self, dataset : Dataset) -> None:
        super().__init__()
        self.inner : Dataset = dataset
    
    def __len__(self):
        return len(self.inner) # type: ignore

    def __getitem__(self, index):
        x, _ = self.inner.__getitem__(index)
        return x, x

class _IterableDatasetWrapper(IterableDataset):
    def __init__(self, dataset : IterableDataset) -> None:
        super().__init__()
        self.inner : IterableDataset = dataset
    
    def __len__(self):
        return len(self.inner) # type: ignore

    def __iter__(self):
        for x, _ in self.inner:
            yield x, x

class AutoencoderDataset(SplitDataset):
    def __init__(self, dataset : SplitDataset):
        self.inner = dataset

    def _wrap_for_autoencoder(self, dataset : Dataset) -> Dataset:
        if isinstance(dataset, IterableDataset):
            wrapped = _IterableDatasetWrapper(dataset)
        else:
            wrapped = _DatasetWrapper(dataset)
        setattr(wrapped, 'dataset', self.inner)
        return wrapped

    def get_shape(self):
        return self.inner.get_shape()
    
    def get_metric(self, metric : str):
        return self.inner.get_metric(metric)

    def get_collumn_references(self) -> CollumnReferences:
        raise NotImplementedError("Collumn references not available for autoencoder dataset")

    def for_training(self) -> Dataset:
        return self._wrap_for_autoencoder(self.inner.for_training())
    
    def for_validation(self) -> Dataset:
        return self._wrap_for_autoencoder(self.inner.for_validation())
    
    def for_testing(self) -> Dataset | None:
        dataset = self.inner.for_testing()
        if dataset is None:
            return None
        return self._wrap_for_autoencoder(dataset)

