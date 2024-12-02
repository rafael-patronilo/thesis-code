from typing import Callable, Iterable, TYPE_CHECKING, Literal
from torcheval.metrics import Metric
from core.datasets import SplitDataset
import warnings

if TYPE_CHECKING:
    from torch.utils.data import Dataset as TorchDataset
    import torch

def chain(*constructors : Callable[[Metric], Metric], inner : Callable[[], Metric]) -> Callable[[], Metric]:
    def constructor():
        result = inner()
        for constructor in reversed(constructors):
            result = constructor(result)
        return result
    return constructor

class MetricWrapper(Metric):
    def __init__(self, inner : Metric) -> None:
        self.inner = inner

    def reset(self):
        self.inner.reset()

    def to(self, device):
        self.inner.to(device)

    def merge_state(self, metrics: Iterable[Metric]) -> Metric:
        return self.inner.merge_state(unwrap(metric) for metric in metrics)

    def update(self, y_pred, y_true):
        self.inner.update(y_pred, y_true)

    def compute(self):
        return self.inner.compute()

    def unwrap(self) -> Metric:
        innermost = self.inner
        while isinstance(innermost, MetricWrapper):
            innermost = innermost.inner
        return innermost

    @classmethod
    def partial(cls, *args, **kwargs) -> Callable[[Metric], Metric]:
        def constructor(inner):
            return cls(inner, *args, **kwargs)
        return constructor

    def __repr__(self):
        fields = "".join(f", {k}={v}" for k, v in self.__dict__.items() if k != 'inner')
        return f"{self.__class__.__name__}({self.inner}{fields})"

def unwrap(metric : Metric) -> Metric:
    if isinstance(metric, MetricWrapper):
        return metric.unwrap()
    else:
        return metric

class Flatten(MetricWrapper):
    def __init__(self, inner : Metric, start_dim = 0, end_dim=-1) -> None:
        super().__init__(inner)
        self.start_dim = start_dim
        self.end_dim = end_dim

    def update(self, y_pred, y_true):
        y_pred = y_pred.flatten(self.start_dim, self.end_dim)
        y_true = y_true.flatten(self.start_dim, self.end_dim)
        return self.inner.update(y_pred, y_true)

class SelectCol(MetricWrapper):
    def __init__(
            self, 
            inner : Metric, 
            select : int | list[int],
            apply_to_preds : bool = True,
            apply_to_true : bool = True,
            flatten : bool = True,
            batched : bool = True
        ) -> None:
        super().__init__(inner)
        if isinstance(select, int):
            select = [select]
        self.select : list = select
        self.apply_to_preds = apply_to_preds
        self.apply_to_true = apply_to_true
        self.flatten = flatten
        self.batched = batched

    def _select_tensor(self, tensor : 'torch.Tensor') -> 'torch.Tensor':
        if self.batched:
            tensor = tensor[:, self.select]
        else:
            tensor = tensor[self.select]
        if self.flatten:
            tensor = tensor.flatten()
        return tensor

    def update(self, y_pred, y_true):
        if self.apply_to_preds:
            y_pred = self._select_tensor(y_pred)
        if self.apply_to_true:
            y_true = self._select_tensor(y_true)
        return self.inner.update(y_pred, y_true)
    
    @classmethod
    def col_wise(
            cls, 
            dataset : 'TorchDataset | SplitDataset', 
            metrics:dict[str, Callable[[],Metric]], 
            out_dict : dict[str, Metric] | None = None
        ) -> dict[str, Metric]:
        split_dataset = SplitDataset.of(dataset)
        try:
            label_names = split_dataset.get_collumn_references().labels.collumns_to_names
        except:
            warnings.warn(f"Could not get collumn references from {dataset}. SelectCol will use indices instead of names.")
            label_names = map(str, range(split_dataset.get_shape()[1][0]))
        result = out_dict or {}
        for name, metric_factory in metrics.items():
            result[name] = Flatten(metric_factory())
            for label_idx, label_name in enumerate(label_names):
                result[f"{name}_{label_name}"] = cls(metric_factory(), label_idx)
        return result
    
    @classmethod
    def from_names(
            cls, 
            names : str | list[str],  
            dataset : 'TorchDataset | SplitDataset',  
            metric : Metric, 
            **kwargs
        ) -> 'SelectCol':
        split_dataset = SplitDataset.of(dataset)
        names_to_cols = split_dataset.get_collumn_references().labels.names_to_collumn
        if not isinstance(names, list):
            names = [names]
        cols = [names_to_cols[name] for name in names]
        return SelectCol(metric, cols, **kwargs)
        

class ToDtype(MetricWrapper):
    def __init__(self, 
                 inner : Metric, 
                 dtype : 'torch.dtype' | Literal['pred', 'true'], 
                 apply_to_pred : bool = True,
                 apply_to_true : bool = True) -> None:
        super().__init__(inner)
        self.dtype = dtype
        self.apply_to_pred = apply_to_pred
        self.apply_to_true = apply_to_true

    def update(self, y_pred, y_true):
        if self.dtype == 'pred':
            dtype = y_pred.dtype
        elif self.dtype == 'true':
            dtype = y_true.dtype
        else:
            dtype = self.dtype
        if self.apply_to_pred:
            y_pred = y_pred.to(dtype)
        if self.apply_to_true:
            y_true = y_true.to(dtype)
        return self.inner.update(y_pred, y_true)