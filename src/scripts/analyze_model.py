import sys
from core import Trainer, ModelFileManager
from collections import OrderedDict
from dataclasses import dataclass
from typing import NamedTuple
import torch
from torch.utils.data import DataLoader
import logging
logger = logging.getLogger(__name__)

class Stats(NamedTuple):
    max: float
    min: float
    mean: float
    std: float

def analyze_weights(trainer: Trainer):
    logger.info("Analyzing model weights...")
    for name, param in trainer.model.named_parameters():
        logger.info(
            f"{name}:\n"
            f"\tMax: {param.max()}\n"
            f"\tMin: {param.min()}\n"
            f"\tMean: {param.mean()}\n"
            f"\tStd: {param.std()}"
        )

def analyze_layer_output(trainer : Trainer):
    @dataclass
    class LayerStats:
        sum: float = 0
        max: float = 0
        min: float = 0
        count: int = 0
    layer_stats : OrderedDict[torch.nn.Module, LayerStats] = OrderedDict()
    def forward_hook(module, input, output):
        stats = layer_stats.setdefault(module, LayerStats())
        stats.sum += output.sum().item()
        stats.count += output.numel()
        stats.max = max(stats.max, output.max().item())
        stats.min = min(stats.min, output.min().item())
    trainer.model.register_forward_hook(forward_hook)
    dataloader = DataLoader(
        trainer.try_get_validation_set()
    )


def main():
    path = sys.argv[1]
    with ModelFileManager(path) as file_manager:
        trainer = Trainer.load_checkpoint(file_manager)
        trainer.model.eval()
        with torch.no_grad():
            analyze_weights(trainer)