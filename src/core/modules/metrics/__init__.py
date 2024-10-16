import torch
from typing import Any, Optional, Sequence
import torch
import inspect
from torcheval.metrics import functional as torch_metrics
from core.datasets import SplitDataset
from .elapsed import Elapsed
from typing import Callable

MetricFunction = Callable[[torch.Tensor, torch.Tensor], float]
NamedMetricFunction = str | tuple[str, MetricFunction]

def decorate_torch_metric(metric : Callable[[torch.Tensor, torch.Tensor], torch.Tensor], flatten_tensors=False) -> MetricFunction:
    def decorated_metric(y_pred, y_true, **kwargs):
        if flatten_tensors:
            y_pred = y_pred.flatten()
            y_true = y_true.flatten()
        return metric(y_pred, y_true).item()
    return decorated_metric

__all_metrics : dict[str, MetricFunction | type] = {
    'accuracy' : decorate_torch_metric(torch_metrics.binary_accuracy, flatten_tensors=True),
    'f1_score' : decorate_torch_metric(torch_metrics.binary_f1_score, flatten_tensors=True),
    'epoch_elapsed' : Elapsed,
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
            return decorate_torch_metric(torch_metric)
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