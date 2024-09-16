import torch
from typing import Any
import torch
import inspect
from torcheval.metrics import functional as torch_metrics
from epoch_time import EpochTime
from typing import Callable

def _decorate_torch_metric(metric : Callable[[torch.Tensor, torch.Tensor], torch.Tensor]) -> Callable[[torch.Tensor, torch.Tensor], float]:
    def decorated_metric(y_pred, y_true):
        return metric(y_pred, y_true).item()
    return decorated_metric

__all_metrics = {
    'accuracy' : _decorate_torch_metric(torch_metrics.binary_accuracy),
    'epoch_delta_time' : EpochTime,
}

def select_metrics(metrics : list[str]) -> dict[str, Any]:
    metric_functions = {}
    for metric_name in metrics:
        if metric_name in __all_metrics:
            metric = __all_metrics[metric_name]
            if inspect.isclass(metric):
                metric = metric()
            metric_functions[metric_name] = metric
        else:
            torch_metric = torch_metrics.__dict__.get(metric_name)
            if torch_metric is not None:
                metric_functions[metric_name] = _decorate_torch_metric(torch_metric)
            else:
                raise ValueError(f"Invalid metric: {metric_name}")
    return metric_functions