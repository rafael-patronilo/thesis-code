from core.datasets import SplitDataset, CollumnReferences
from torch.utils.data import Dataset, IterableDataset

class _DatasetWrapper(Dataset):
    def __init__(self, dataset : Dataset) -> None:
        super().__init__()
        self.wrapped : Dataset = dataset
    
    def __getitem__(self, index):
        x, _ = self.wrapped.__getitem__(index)
        return x, x

class _IterableDatasetWrapper(IterableDataset):
    def __init__(self, dataset : IterableDataset) -> None:
        super().__init__()
        self.wrapped : IterableDataset = dataset
    
    def __iter__(self):
        for x, _ in self.wrapped:
            yield x, x

def wrap_for_autoencoder(dataset : Dataset) -> Dataset:
    if isinstance(dataset, IterableDataset):
        return _IterableDatasetWrapper(dataset)
    else:
        return _DatasetWrapper(dataset)

class AutoencoderDataset(SplitDataset):
    def __init__(self, dataset : SplitDataset):
        self.wrapped = dataset

    def get_shape(self):
        return self.wrapped.get_shape()
    
    def get_metric(self, metric : str):
        return self.wrapped.get_metric(metric)

    def get_collumn_references(self) -> CollumnReferences:
        raise NotImplementedError("Collumn references not available for autoencoder dataset")

    def for_training(self) -> Dataset:
        return wrap_for_autoencoder(self.wrapped.for_training())
    
    def for_validation(self) -> Dataset:
        return wrap_for_autoencoder(self.wrapped.for_validation())
    
    def for_testing(self) -> Dataset | None:
        dataset = self.wrapped.for_testing()
        if dataset is None:
            return None
        return wrap_for_autoencoder(dataset)

