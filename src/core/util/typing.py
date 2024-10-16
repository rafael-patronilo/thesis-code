import typing
import os
from torch.utils.data import Dataset, IterableDataset

type PathLike = str | os.PathLike
type TorchDataset = Dataset | IterableDataset