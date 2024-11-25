from typing import Callable, Iterable, TYPE_CHECKING
from torcheval.metrics import Metric
from core.datasets import SplitDataset
import warnings

if TYPE_CHECKING:
    from torch.utils.data import Dataset as TorchDataset

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
            flatten : bool = True
        ) -> None:
        super().__init__(inner)
        if isinstance(select, int):
            select = [select]
        self.select : list = select
        self.apply_to_preds = apply_to_preds
        self.apply_to_true = apply_to_true
        self.flatten = flatten

    def update(self, y_pred, y_true):
        if self.apply_to_preds:
            y_pred = y_pred[self.select]
            if self.flatten:
                y_pred = y_pred.flatten()
        if self.apply_to_true:
            y_true = y_true[self.select]
            if self.flatten:
                y_true = y_true.flatten()
        return self.inner.update(y_pred, y_true)
    
    @classmethod
    def col_wise(cls, dataset : 'TorchDataset | SplitDataset',  metrics:dict[str, Callable[[],Metric]]) -> dict[str, Metric]:
        split_dataset = SplitDataset.of(dataset)
        try:
            label_names = split_dataset.get_collumn_references().labels.collumns_to_names
        except:
            warnings.warn(f"Could not get collumn references from {dataset}. SelectCol will use indices instead of names.")
            label_names = map(str, range(split_dataset.get_shape()[1][0]))
        result = {}
        for name, metric_factory in metrics.items():
            result[name] = Flatten(metric_factory())
            for label_idx, label_name in enumerate(label_names):
                result[f"{name}_{label_name}"] = cls(metric_factory(), label_idx)
        return result
