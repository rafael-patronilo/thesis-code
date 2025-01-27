"""
Simple script to build convolutional perception networks; not meant to be trained on its own
"""

from core.util import conv_out_shape
from core import datasets
from typing import Literal, Sequence, assert_never, Optional
from torch import nn
import torch
from core.training import Trainer
import logging

logger = logging.getLogger(__name__)

class XtrainsPerceptionNetwork(nn.Module):
    def __init__(self, convolutional : nn.Module, linear : nn.Module):
        super().__init__()
        self.convolutional = convolutional
        self.linear = linear

    def forward(self, x):
        x = self.convolutional(x)
        maxes = nn.functional.adaptive_max_pool2d(x, (1, 1))
        avgs = nn.functional.adaptive_avg_pool2d(x, (1, 1))
        x = torch.cat([maxes.flatten(1), avgs.flatten(1)], 1)
        x = self.linear(x)
        return x

def make_model(
        input_shape: Sequence[int],
        first_conv_layers: Sequence[int],
        k1_conv_layers: Sequence[int],
        last_linear_output_size: int,
        first_kernel_size : int = 3,
        first_pool_kernel_size : int = 0,
        hidden_activations: Literal['relu'] | tuple[Literal['leaky_relu'], float] = 'relu'
):
    logger.debug(f"Creating convolutional perception network with kernel size 1 convolutions...")
    conv_layers: list[nn.Module] = []

    def hidden_activation():
        match hidden_activations:
            case 'relu':
                return nn.ReLU()
            case ('leaky_relu', alpha):
                return nn.LeakyReLU(alpha)
            case never:
                assert_never(never)

    in_channels = input_shape[0]
    in_shape = input_shape[1:]
    logger.debug(f"\t{in_channels} x {in_shape} -> ")
    for out_channels in first_conv_layers:
        conv_layers.append(nn.Conv2d(
            in_channels, out_channels, first_kernel_size,
            padding='same')
        )
        conv_layers.append(hidden_activation())
        in_channels = out_channels
        logger.debug(f"\tConv k = {first_kernel_size} {in_channels} x {in_shape} -> ")
    if first_pool_kernel_size > 1:
        in_shape = conv_out_shape(in_shape, first_pool_kernel_size,
                                  padding=0, stride=first_pool_kernel_size)
        conv_layers.append(nn.MaxPool2d(first_pool_kernel_size, padding=0))
        logger.debug(f"\tPool {first_pool_kernel_size} -> {in_channels} x {in_shape}")

    for out_channels in k1_conv_layers:
        conv_layers.append(nn.Conv2d(
            in_channels, out_channels, 1, padding=0)
        )
        conv_layers.append(hidden_activation())
        in_channels = out_channels
        logger.debug(f"\tConv k = 1 {in_channels} x {in_shape} -> ")

    return XtrainsPerceptionNetwork(
        nn.Sequential(*conv_layers),
        nn.Sequential(
            nn.LazyLinear(last_linear_output_size),
            nn.Sigmoid()
        )
    )

def create_trainer(dataset_name: str, **kwargs) -> Trainer:
    loss_fn = nn.BCELoss()
    dataset = datasets.get_dataset(dataset_name)
    input_shape = dataset.get_shape()[0]
    logger.debug(f"Input shape: {input_shape}")
    return Trainer(
        model=make_model(input_shape, **kwargs),
        loss_fn=loss_fn,
        optimizer=torch.optim.Adam,
        training_set=dataset,
        metric_loggers=[],
        stop_criteria=[],
        checkpoint_triggers=[],
        batch_size=64
    )