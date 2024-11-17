from core.util import conv_out_shape, transposed_conv_out_shape
from core import util, datasets
from core.metrics import get_metric
from core.stop_criteria import EarlyStop
from core.checkpoint_triggers import BestMetric
from typing import Literal, Sequence, Any, assert_never
from torch import nn
import torch
from core.nn.autoencoder import AutoEncoder
from core import Trainer, MetricsLogger, TrainingLogger
import torcheval.metrics as torch_metrics
import logging
from types import SimpleNamespace

from core.datasets import dataset_wrappers
logger = logging.getLogger(__name__)

EARLY_STOP = SimpleNamespace()
EARLY_STOP.patience = 10
EARLY_STOP.threshold = 0.0001

def make_model(
        input_shape : Sequence[int], 
        conv_layers : Sequence[int | tuple[Literal['pool'], int]], 
        linear_layers : Sequence[int], 
        encoding_size : int, 
        encoding_activation : Literal['relu', 'sigmoid'] = 'sigmoid',
        kernel_size : int = 3
    ):
    logger.debug(f"Creating convolutional autoencoder...")
    same_padding = (kernel_size - 1) // 2
    encoder_layers : list[nn.Module] = []
    deconv_layers : list[int | tuple[Literal['pool'], int, Sequence[int]]] = []
    in_channels = input_shape[0]
    in_shape = input_shape[1:]
    logger.debug(f"\t{in_channels} x {in_shape} -> ")
    for conv_layer in conv_layers:
        match conv_layer:
            case ('pool', pool_ksize):
                encoder_layers.append(nn.MaxPool2d(pool_ksize, padding=0))
                deconv_layers.append(('pool', pool_ksize, in_shape))
                in_shape = conv_out_shape(in_shape, pool_ksize, padding=0, stride=pool_ksize)
                logger.debug(f"\tpool {pool_ksize}")
            case out_channels:
                assert isinstance(out_channels, int)
                encoder_layers.append(nn.Conv2d(in_channels, out_channels, kernel_size, padding=same_padding))
                encoder_layers.append(nn.ReLU())
                deconv_layers.append(in_channels)
                in_channels = out_channels
                logger.debug(f"\tconv")
        logger.debug(f"\t{in_channels} x {in_shape} -> ")
    
    last_conv_total_features = in_channels
    for dim_size in in_shape:
        last_conv_total_features *= dim_size
    in_features = last_conv_total_features
    encoder_layers.append(nn.Flatten())
    logger.debug(f"\tFlatten {in_features} ->")

    for out_features in linear_layers:
        encoder_layers.append(nn.Linear(in_features, out_features))
        encoder_layers.append(nn.ReLU())
        in_features = out_features
        logger.debug(f"\tLinear {out_features} ->")
    
    logger.debug(f"\tEncoding {encoding_size}->")
    encoder_layers.append(nn.Linear(in_features, encoding_size))
    match encoding_activation:
        case 'relu':
            encoder_layers.append(nn.ReLU())
        case 'sigmoid':
            encoder_layers.append(nn.Sigmoid())
        case never:
            assert_never(never)
    in_features = encoding_size

    
    decoder_layers : list[nn.Module] = []
    for out_features in reversed(linear_layers):
        decoder_layers.append(nn.Linear(in_features, out_features))
        decoder_layers.append(nn.ReLU())
        in_features = out_features
        logger.debug(f"\tLinear {out_features} ->")
    decoder_layers.append(nn.Linear(in_features, last_conv_total_features))
    decoder_layers.append(nn.ReLU())
    logger.debug(f"\tLinear {last_conv_total_features} ->")

    decoder_layers.append(nn.Unflatten(1, [in_channels] + in_shape)) # type: ignore
    logger.debug(f"\tUnflatten {in_channels} x {in_shape} ->")
    for i, conv_layer in enumerate(reversed(deconv_layers)):
        match conv_layer:
            case ('pool', pool_ksize, target_shape):
                decoder_layers.append(nn.ConvTranspose2d(in_channels, in_channels, pool_ksize, padding=0, stride=pool_ksize))
                in_shape = transposed_conv_out_shape(in_shape, pool_ksize, padding=0, stride=pool_ksize)
                logger.debug(f"\tpool(conv transpose) {pool_ksize}")
                if in_shape != target_shape:
                    pad_right = target_shape[0] - in_shape[0]
                    pad_bottom = target_shape[1] - in_shape[1]
                    logger.debug(f"\tFixing discrepancy with padding ({pad_right, pad_bottom}) to get shape {target_shape}")
                    decoder_layers.append(nn.ReplicationPad2d((0, pad_right, 0, pad_bottom)))
                    in_shape = target_shape
            case out_channels:
                assert isinstance(out_channels, int)
                decoder_layers.append(nn.ConvTranspose2d(in_channels, out_channels, kernel_size, padding=same_padding))
                if i == len(deconv_layers) - 1:
                    decoder_layers.append(nn.Sigmoid())
                else:
                    decoder_layers.append(nn.ReLU())
                in_channels = out_channels
                logger.debug(f"\tconv transpose")
        logger.debug(f"\t{in_channels} x {in_shape} ->")
    logger.debug(f"End model")

    return AutoEncoder(
        encoder=nn.Sequential(*encoder_layers),
        decoder=nn.Sequential(*decoder_layers)
    )

def create_trainer(dataset_name : str, **kwargs) -> Trainer:
    loss_fn = nn.MSELoss()
    loss_metric = torch_metrics.MeanSquaredError
    dataset = datasets.get_dataset(dataset_name)
    dataset = dataset_wrappers.ForAutoencoder(dataset)
    input_shape = dataset.get_shape()[0]
    logger.debug(f"Input shape: {input_shape}")
    metrics = ['epoch_elapsed']
    metric_functions : dict = {}
    def train_loss_metric(loss):
        return loss
    for metric in metrics:
        metric_functions[metric] = get_metric(metric)
    #train_metrics = MetricsLogger(
    #    identifier='train',
    #    metric_functions={'loss':train_loss_metric}, #type:ignore
    #    dataset=dataset.for_training
    #)
    val_metrics = MetricsLogger(
        identifier='val',
        metric_functions=metric_functions | {'loss': loss_metric()}, #type:ignore
        dataset=dataset.for_validation
    )
    train_metrics = TrainingLogger(
        metric_functions=metric_functions | {'loss': loss_metric()}  #type:ignore
    )
    return Trainer(
        model=make_model(input_shape, **kwargs),
        loss_fn=loss_fn,
        optimizer=torch.optim.Adam,
        training_set=dataset.for_training(),
        metric_loggers=[train_metrics, val_metrics],
        stop_criteria=[EarlyStop(
            metric='loss', prefer='min', metrics_logger ='val', threshold=EARLY_STOP.threshold, patience=EARLY_STOP.patience)],
        checkpoint_triggers=[BestMetric(
            metric='loss', prefer='min', metrics_logger ='val', threshold=EARLY_STOP.threshold)],
        batch_size=64
    )