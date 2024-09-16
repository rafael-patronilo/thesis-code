import torch
from typing import Any
import torch
import inspect
from torcheval.metrics import functional as torch_metrics
from typing import Callable
from . import metrics


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