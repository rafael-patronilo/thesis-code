import torch
from typing import Any
import torch
from torcheval.metrics import functional as torch_metrics
from typing import Callable

def _decorate_torch_metric(metric : Callable[[torch.Tensor, torch.Tensor], torch.Tensor]) -> Callable[[torch.Tensor, torch.Tensor], float]:
    def decorated_metric(y_pred, y_true):
        return metric(y_pred, y_true).item()
    return decorated_metric

all_metrics = {
    'accuracy' : _decorate_torch_metric(torch_metrics.binary_accuracy),
}

def select_metrics(metrics : list[str]) -> dict[str, Any]:
    metric_functions = {}
    for metric in metrics:
        if metric not in all_metrics:
            torch_metric = torch_metrics.__dict__.get(metric)
            if torch_metric is not None:
                metric_functions[metric] = _decorate_torch_metric(torch_metric)
            else:
                raise ValueError(f"Invalid metric: {metric}")
        metric_functions[metric] = all_metrics[metric]
    return metric_functions


all_losses = {
    'bce': torch.nn.functional.binary_cross_entropy,
}

def get_loss_function(name):
    if name not in all_losses:
        raise ValueError(f"Invalid loss function: {name}")
    return all_losses[name]

def get_optimizer(optimizer_name, model):
    optimizer = torch.optim.__dict__.get(optimizer_name)
    if optimizer is None:
        raise ValueError(f"Invalid optimizer: {optimizer_name}")
    return optimizer(model.parameters())