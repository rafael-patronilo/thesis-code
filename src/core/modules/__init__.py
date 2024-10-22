import torch
from typing import Any, Callable
import torch
from . import metrics # export

__all_losses = {
    'bce': torch.nn.functional.binary_cross_entropy,
}

def loss_function_exists(name : str):
    return name in __all_losses

def get_loss_function(name : str) -> Callable[[torch.Tensor, torch.Tensor], torch.Tensor]:
    if name not in __all_losses:
        raise ValueError(f"Invalid loss function: {name}")
    return __all_losses[name]

def __find_optimizer(name):
    return torch.optim.__dict__.get(name)

def optimizer_exists(name):
    return __find_optimizer(name) is not None

def get_optimizer(optimizer_name, model):
    optimizer = __find_optimizer(optimizer_name)
    if optimizer is None:
        raise ValueError(f"Invalid optimizer: {optimizer_name}")
    return optimizer(model.parameters())