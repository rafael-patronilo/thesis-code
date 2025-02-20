
from typing import Protocol
from torch import nn
import torch


class WeightedLossFunction(Protocol):
    # noinspection PyShadowingBuiltins input
    def __call__(self, input : torch.Tensor, target : torch.Tensor, weight : torch.Tensor) -> torch.Tensor:
        ...

class WeightedTarget(nn.Module):
    """
    Wraps a weighted loss function to apply a different weight tensor for each target.
    The target tensor should be of shape (batch_size, 2, *)
    where target[:, 0] is the real target and target[:, 1] is the weight tensor.
    """
    def __init__(self, loss_fn : WeightedLossFunction):
        super().__init__()
        self.loss_fn = loss_fn

    # noinspection PyShadowingBuiltins input
    def forward(self, input : torch.Tensor, target : torch.Tensor) -> torch.Tensor:
        real_target = target[:, 0]
        target_weights = target[:, 1]
        return self.loss_fn(input=input, target=real_target, weight=target_weights)