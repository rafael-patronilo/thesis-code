from typing import Iterable
from torcheval.metrics import Metric

class MetricWrapper(Metric):
    def __init__(self, inner : Metric) -> None:
        self.inner = inner

    def reset(self):
        self.inner.reset()

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

class SelectCol(MetricWrapper):
    def __init__(
            self, 
            inner : Metric, 
            select : int | list[int],
            apply_to_preds : bool = True,
            apply_to_true : bool = True
        ) -> None:
        super().__init__(inner)
        if isinstance(select, int):
            select = [select]
        self.select : list = select
        self.apply_to_preds = apply_to_preds
        self.apply_to_true = apply_to_true

    def update(self, y_pred, y_true):
        if self.apply_to_preds:
            y_pred = y_pred[self.select]
        if self.apply_to_true:
            y_true = y_true[self.select]
        return self.inner.update(y_pred, y_true)
