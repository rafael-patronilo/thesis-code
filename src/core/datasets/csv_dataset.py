from . import SplitDataset, CollumnReferences, CollumnSubReferences
import logging
from typing import Optional, Any, Callable
import pandas as pd
import torch
from torch.utils.data import Dataset
from pathlib import Path

logger = logging.getLogger(__name__)

class CSVDataset(SplitDataset):
    def __init__(self, 
                 path: str | Path, 
                 target: str | list[str], 
                 features: Optional[list[str]] = None, 
                 splits: float | tuple[float, float] = (0.7, 0.15),
                 shuffle: bool = True,
                 random_state = None,
                 filter : Optional[Callable[[pd.Series], bool]] = None,
                 read_csv_kw : dict = {}):
        super().__init__()
        self.path = path
        self.target : list[str] = target if isinstance(target, list) else [target]
        self.features = features
        self.splits = self._cast_splits(splits)
        self.scalar_preprocessors = {}
        self.tensor_preprocessors = {}
        self.shuffle = shuffle
        self.random_state = random_state
        self.filter = filter
        self.read_csv_kw = read_csv_kw

    def _set_collumn_references(self, features: list[str], target: list[str]):
        self.collumn_references = CollumnReferences(
            CollumnSubReferences(
                {name: idx for idx, name in enumerate(features)},
                features
            ),
            CollumnSubReferences(
                {name: idx for idx, name in enumerate(target)},
                target
            )
        )
    
    def add_collumn_preprocessor(
            self, 
            collumn : str, 
            preprocessor : Callable,
            is_scalar : bool = False
        ):
        """Adds a preprocessor for a collumn in the dataset

        Args:
            collumn (str): 
                The collumn name
            preprocessor (Callable[[Any], torch.Tensor]): 
                A function that returns a tensor/scalar given the value of the collumn
            is_scalar (bool, optional): 
                Whether the preprocessor returns scalars.
                Scalars will be included as a collumn in the first tensor. 
                Defaults to False.
        """
        if is_scalar:
            self.scalar_preprocessors[collumn] = preprocessor
        else:
            self.tensor_preprocessors[collumn] = preprocessor

    def _get_preprocessors(
            self,
            data : pd.DataFrame, 
            cols : list[str]
        ) -> tuple[list[tuple[str, Callable]], list[tuple[str, Callable]]]:
        scalars = data[cols].select_dtypes(include=['number'])
        scalar_preprocessors = []
        tensor_preprocessors = []
        for col in cols:
            scalar_preprocessor = self.scalar_preprocessors.get(col)
            tensor_preprocessor = self.tensor_preprocessors.get(col)
            if scalar_preprocessor is not None and tensor_preprocessor is not None:
                logger.error(f"Collumn {col} has both scalar and tensor preprocessors. Scalar will be used")
            if col not in data.columns:
                logger.error(f"Collumn {col} not found in dataset. It will be ignored")
            elif scalar_preprocessor is not None:
                scalar_preprocessors.append((col, scalar_preprocessor))
            elif tensor_preprocessor is not None:
                tensor_preprocessors.append((col, tensor_preprocessor))
            elif col in scalars.columns:
                scalar_preprocessors.append((col, lambda x: x))
            else:
                logger.error(f"Collumn {col} is not scalar and not preprocessor was provided. It will be ignored")
        return scalar_preprocessors, tensor_preprocessors

    def _load(self):
        if self.loaded:
            return
        logger.info(f"Loading CSV dataset from {self.path}")
        self.data = pd.read_csv(self.path, **self.read_csv_kw)
        if self.filter:
            self.data = self.data[self.data.apply(self.filter, axis=1)]
        if self.features is None:
            self.features = [col for col in self.data.columns if col not in self.target]
        self._set_collumn_references(self.features, self.target)

        scalar_features, tensor_features = self._get_preprocessors(self.data, self.features)
        scalar_target, tensor_targets = self._get_preprocessors(self.data, self.target)

        if self.shuffle:
            logger.debug("Shuffling dataset")
            self.data = self.data.sample(frac=1, random_state=self.random_state).reset_index(drop=True)
        train_bound, val_bound = self._split(len(self.data), self.splits)
        train_rows = self.data.iloc[:train_bound]
        val_rows = self.data.iloc[train_bound:val_bound]
        test_rows = self.data.iloc[val_bound:]

        logger.debug(
f"""Dataset ({len(self.data):_} samples) split:
Training: \t [0, {train_bound:_}[ \t ({len(train_rows):_} samples, {self.splits[0] * 100}%)
Validation: \t [{train_bound:_}, {val_bound}[ \t ({len(val_rows):_} samples, {self.splits[1] * 100}%)
Testing: \t [{val_bound:_}, {len(self.data):_}[ \t ({len(test_rows):_} samples, {(1 - self.splits[0] - self.splits[1]) * 100}%)
Num features: \t {len(self.features)}
Num target: \t {len(self.target)}
Shuffle: \t {self.shuffle}
Seed: \t {self.random_state}""")
        self.train_data = self.DFDataset(
            scalar_features, scalar_target, tensor_features, tensor_targets, train_rows)
        self.val_data = self.DFDataset(
            scalar_features, scalar_target, tensor_features, tensor_targets, val_rows)
        self.test_data = self.DFDataset(
            scalar_features, scalar_target, tensor_features, tensor_targets, test_rows) if len(test_rows) > 0 else None
        
        self.loaded = True
    
    class DFDataset(Dataset):
        def __init__(
                self,
                scalar_features: list[tuple[str, Callable]],
                scalar_targets : list[tuple[str, Callable]],
                tensor_features : list[tuple[str, Callable]],
                tensor_targets : list[tuple[str, Callable]],
                rows : pd.DataFrame,
                ) -> None:
            self.rows = rows
            self.scalar_features = scalar_features
            self.scalar_targets = scalar_targets
            self.tensor_features = tensor_features
            self.tensor_targets = tensor_targets

        def __len__(self):
            return len(self.rows)

        def _torchify_row(
                self, 
                row : pd.Series, 
                scalar_preprocessors : list[tuple[str, Callable]], 
                tensor_preprocessors : list[tuple[str, Callable]]
            ):
            if len(scalar_preprocessors) == 0:
                scalars = None
            else:
                scalars = [preprocessor(row[col]) for col, preprocessor in scalar_preprocessors]
                scalars = torch.tensor(scalars, device='cpu')
            if len(tensor_preprocessors) == 0:
                tensors = None
            else:
                tensors = [preprocessor(row[col]) for col, preprocessor in tensor_preprocessors]
            if scalars is None:
                if tensors is not None and len(tensors) == 1:
                    return tensors[0]
                else:
                    return tensors
            elif tensors is None:
                return scalars
            else:
                tensors.insert(0, scalars)
                return tensors

        def __getitem__(self, index : int) -> Any:
            row = self.rows.iloc[index]
            features = self._torchify_row(row, self.scalar_features, self.tensor_features)
            target = self._torchify_row(row, self.scalar_targets, self.tensor_targets)
            return features, target


