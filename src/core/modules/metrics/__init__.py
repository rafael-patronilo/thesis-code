import torch
from typing import Any
import torch
import inspect
from torcheval.metrics import functional as torch_metrics
from .elapsed import Elapsed
from typing import Callable

def decorate_torch_metric(metric : Callable[[torch.Tensor, torch.Tensor], torch.Tensor], flatten_tensors=False) -> Callable[[torch.Tensor, torch.Tensor], float]:
    def decorated_metric(y_pred, y_true, **kwargs):
        if flatten_tensors:
            y_pred = y_pred.flatten()
            y_true = y_true.flatten()
        return metric(y_pred, y_true).item()
    return decorated_metric

__all_metrics = {
    'accuracy' : decorate_torch_metric(torch_metrics.binary_accuracy, flatten_tensors=True),
    'f1_score' : decorate_torch_metric(torch_metrics.binary_f1_score, flatten_tensors=True),
    'epoch_elapsed' : Elapsed,
}

def get_metric(name):
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

def select_metrics(metrics : list[str]) -> dict[str, Any]:
    metric_functions = {}
    for metric_name in metrics:
        metric_functions[metric_name] = get_metric(metric_name)
    return metric_functions