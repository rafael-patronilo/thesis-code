import logging
from sys import exc_info
from typing import Callable, Iterable, TYPE_CHECKING, Literal, Self
from torcheval.metrics import Metric
from core.datasets import SplitDataset
import warnings

if TYPE_CHECKING:
    from torch.utils.data import Dataset as TorchDataset
    import torch

class MetricWrapper(Metric):
    def __init__(self, inner : Metric) -> None:
        self.inner = inner
        super().__init__()

    def reset(self) -> Self:
        self.inner.reset()
        return self

    def to(self, device, *_, **__):
        self.inner.to(device)
        return self

    def merge_state(self, metrics: Iterable[Metric]) -> Metric:
        return self.inner.merge_state(unwrap(metric) for metric in metrics)

    def update(self, y_pred, y_true):
        self.inner.update(y_pred, y_true)
        return self

    def compute(self):
        return self.inner.compute()

    def unwrap(self) -> Metric:
        innermost = self.inner
        while isinstance(innermost, MetricWrapper):
            innermost = innermost.inner
        return innermost

    def __repr__(self):
        fields = "".join(f", {k}={v}" for k, v in self.__dict__.items() if k != 'inner')
        return f"{self.__class__.__name__}({self.inner}{fields})"

class MultiMetricWrapper(Metric):
    def __init__(self, *metrics : Metric) -> None:
        self.metrics = metrics
        super().__init__()

    def reset(self):
        for metric in self.metrics:
            metric.reset()
        return self

    def to(self, device, *_, **__):
        for metric in self.metrics:
            metric.to(device)
        return self

    def merge_state(self, metrics: Iterable[Metric]) -> Metric:
        merged_metrics = []
        for metric, others in zip(self.metrics, zip(metrics)):
            merged_metrics.append(metric.merge_state(others))
        return self.__class__(*merged_metrics)
    
    def update(self, y_pred, y_true):
        for metric in self.metrics:
            metric.update(y_pred, y_true)
        return self

    def compute(self):
        raise NotImplementedError("Derive this class with some agreggation strategy")

    def __repr__(self):
        return f"{self.__class__.__name__}({', '.join(map(str, self.metrics))})"
    
    @classmethod
    def foreach_class(
            cls,
            dataset : 'TorchDataset | SplitDataset', 
            metric : Callable[[], Metric],
        ) -> 'MultiMetricWrapper':
        return cls(*SelectCol.col_wise(dataset, {'metric' : metric}, reduction=None).values())


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
        self.inner.update(y_pred, y_true)
        return self

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
        self.inner.update(y_pred, y_true)
        return self
    
    @classmethod
    def col_wise(
            cls, 
            dataset_or_label_names : 'TorchDataset | SplitDataset | list[str]',
            metrics:dict[str, Callable[[],Metric]], 
            reduction : Literal['flatten_first', 'min'] | None = 'flatten_first',
            out_dict : dict[str, Metric] | None = None
        ) -> dict[str, Metric]:
        if isinstance(dataset_or_label_names, list):
            label_names = dataset_or_label_names
        else:
            split_dataset = SplitDataset.of(dataset_or_label_names)
            try:
                label_names = split_dataset.get_column_references().labels.columns_to_names
            except: # noqa
                warnings.warn(f"Could not get column references from {dataset_or_label_names}. "
                              f"SelectCol will use indices instead of names."
                              )
                label_names = map(str, range(split_dataset.get_shape()[1][0]))
        result = out_dict or {}
        for name, metric_factory in metrics.items():
            to_reduce = []
            for label_idx, label_name in enumerate(label_names):
                metric = cls(metric_factory(), label_idx)
                result[f"{name}_{label_name}"] = metric
                to_reduce.append(SkipUpdate(metric))
            match reduction:
                case 'flatten_first':
                    result[name] = Flatten(metric_factory())
                case 'min':
                    result[name] = MinOf(*to_reduce)
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
        names_to_cols = split_dataset.get_column_references().labels.names_to_column
        if not isinstance(names, list):
            names = [names]
        cols = [names_to_cols[name] for name in names]
        return SelectCol(metric, cols, **kwargs)


class MinOf(MultiMetricWrapper):
    def __init__(self, *metrics : Metric) -> None:
        super().__init__(*metrics)

    def compute(self):
        return min(metric.compute() for metric in self.metrics)

class SkipUpdate(MetricWrapper):
    def update(self, y_pred, y_true):
        return self

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
        self.inner.update(y_pred, y_true)
        return self

def to_int(factory : Callable[[], Metric]) -> Callable[[], Metric]:
    import torch
    return lambda: ToDtype(factory(), torch.int32, apply_to_pred=False)

class Unary(MetricWrapper):
    def __init__(self, inner : Metric, side : Literal['pred', 'true'] = 'pred') -> None:
        super().__init__(inner)
        self.side = side

    def update(self, y_pred, y_true):
        if self.side == 'pred':
            self.inner.update(y_pred)
        else:
            self.inner.update(y_true)
        return self