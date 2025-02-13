import logging
import torch
from typing import Any, Optional, Sequence
import torch
import inspect
from torcheval.metrics import functional as torch_metrics
from torcheval.metrics import Metric, BinaryConfusionMatrix
import torcheval
from torcheval.metrics.classification.confusion_matrix import TBinaryConfusionMatrix

from core.datasets import SplitDataset
from .elapsed import Elapsed
from typing import Callable

MetricFunction = Callable[[torch.Tensor, torch.Tensor], float] | Metric
NamedMetricFunction = str | tuple[str, MetricFunction]

class DecoratedTorchMetric:

    def __init__(
            self, 
            metric : Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
            flatten_tensors : bool = False
        ) -> None:
        self.metric = metric
        self.flatten_tensors = flatten_tensors

    def __call__(self, y_pred, y_true, **kwargs):
        if self.flatten_tensors:
            y_pred = y_pred.flatten()
            y_true = y_true.flatten()
        return self.metric(y_pred, y_true).item()
    
    def __repr__(self):
        return f"DecoratedTorchMetric({self.metric})"

__all_metrics : dict[str, MetricFunction | type] = {
    'epoch_elapsed' : Elapsed
}


def get_metric(name : str) -> MetricFunction | None:
    if name in __all_metrics:
        metric = __all_metrics[name]
        if inspect.isclass(metric):
            metric = metric()
        return metric
    else:
        torch_metric = torch_metrics.__dict__.get(name)
        if torch_metric is not None:
            return DecoratedTorchMetric(torch_metric)
        else:
            return None

def metric_exists(name):
    return get_metric(name) is not None

def select_metrics(metrics : Sequence[NamedMetricFunction], dataset : Optional[SplitDataset] = None) -> dict[str, MetricFunction]:
    if dataset is not None:
        # TODO metrics logger uses this function but doesn't know the dataset
        raise NotImplementedError("Dataset specific metrics are not yet implemented")
    metric_functions = {}
    for metric in metrics:
        if isinstance(metric, str):
            metric_functions[metric] = get_metric(metric)
        else:
            name, metric_function = metric
            metric_functions[name] = metric_function
    return metric_functions

class BinaryBalancedAccuracy(BinaryConfusionMatrix):
    def __init__(self):
        assert torcheval.version.__version__ == '0.0.7', "confusion matrix order may have been changed: https://github.com/pytorch/torcheval/issues/183"
        super().__init__()

    def update(
        self, input: torch.Tensor, target: torch.Tensor
    ):
        super().update(input, target)
        return self

    def compute(self):
        cm = super().compute()
        # docs are wrong: https://github.com/pytorch/torcheval/issues/183
        tn = cm[0, 0]
        fp = cm[0, 1]
        fn = cm[1, 0]
        tp = cm[1, 1]
        specificity = tn / (tn + fp)
        recall = tp / (tp + fn)
        result = (specificity + recall) / 2
        return result

class BinaryBalancedSpecificity(BinaryConfusionMatrix):
    def __init__(self):
        assert torcheval.version.__version__ == '0.0.7', "confusion matrix order may have been changed: https://github.com/pytorch/torcheval/issues/183"
        super().__init__()

    def update(
            self, input: torch.Tensor, target: torch.Tensor
    ):
        super().update(input, target)
        return self

    def compute(self):
        cm = super().compute()
        # docs are wrong: https://github.com/pytorch/torcheval/issues/183
        tn = cm[0, 0]
        fp = cm[0, 1]
        #fn = cm[1, 0]
        #tp = cm[1, 1]
        specificity = tn / (tn + fp)
        return specificity

class BinarySpecificity(BinaryConfusionMatrix):
    def __init__(self):
        assert torcheval.version.__version__ == '0.0.7', "confusion matrix order may have been changed: https://github.com/pytorch/torcheval/issues/183"
        super().__init__()

    def compute(self):
        cm = super().compute()
        # docs are wrong: https://github.com/pytorch/torcheval/issues/183
        tn = cm[0, 0]
        fp = cm[0, 1]
        return tn / (tn + fp)
    
class BinaryPositiveRate(BinaryConfusionMatrix):
    def __init__(self):
        assert torcheval.version.__version__ == '0.0.7', "confusion matrix order may have been changed: https://github.com/pytorch/torcheval/issues/183"
        super().__init__()

    def compute(self):
        cm = super().compute()
        # docs are wrong: https://github.com/pytorch/torcheval/issues/183
        tn = cm[0, 0]
        fp = cm[0, 1]
        fn = cm[1, 0]
        tp = cm[1, 1]
        positives = tp + fp
        negatives = tn + fn
        total = positives + negatives
        return positives / total

from .pearson_correlation import PearsonCorrelationCoefficient

__all__=[
    'BinaryBalancedAccuracy',
    'BinarySpecificity',
    'BinaryPositiveRate',
    'PearsonCorrelationCoefficient',
    'Elapsed'
]