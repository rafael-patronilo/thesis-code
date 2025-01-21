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



def make_model(
        input_shape: Sequence[int],
        conv_layers: Sequence[int | tuple[Literal['pool'], int]],
        linear_layers: Sequence[int],
        num_concepts: int,
        hidden_activations: Literal['relu'] | tuple[Literal['leaky_relu'], float] = 'relu',
        dropout_last_layer : Optional[float] = None,
        kernel_size: int = 3
):
    logger.debug(f"Creating convolutional perception network...")
    same_padding = (kernel_size - 1) // 2
    pn_layers: list[nn.Module] = []

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
    for conv_layer in conv_layers:
        match conv_layer:
            case ('pool', pool_ksize):
                pn_layers.append(nn.MaxPool2d(pool_ksize, padding=0))
                in_shape = conv_out_shape(in_shape, pool_ksize, padding=0, stride=pool_ksize)
                logger.debug(f"\tpool {pool_ksize}")
            case out_channels:
                assert isinstance(out_channels, int)
                pn_layers.append(nn.Conv2d(in_channels, out_channels, kernel_size, padding=same_padding))
                pn_layers.append(hidden_activation())
                in_channels = out_channels
                logger.debug(f"\tconv")
        logger.debug(f"\t{in_channels} x {in_shape} -> ")

    last_conv_total_features = in_channels
    for dim_size in in_shape:
        last_conv_total_features *= dim_size
    in_features = last_conv_total_features
    pn_layers.append(nn.Flatten())
    logger.debug(f"\tFlatten {in_features} ->")

    for out_features in linear_layers:
        pn_layers.append(nn.Linear(in_features, out_features))
        pn_layers.append(hidden_activation())
        in_features = out_features
        logger.debug(f"\tLinear {out_features} ->")

    logger.debug(f"\tOutput {num_concepts}->")
    pn_layers.append(nn.Linear(in_features, num_concepts))
    pn_layers.append(nn.Sigmoid())
    if dropout_last_layer is not None:
        pn_layers.append(nn.Dropout(dropout_last_layer))

    return nn.Sequential(*pn_layers)


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