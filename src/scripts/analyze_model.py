
from pathlib import Path
from typing import TYPE_CHECKING, Literal
from core.init import DO_SCRIPT_IMPORTS
from typing import NamedTuple

from core.init.options_parsing import option, positional

from dataclasses import dataclass, field



if TYPE_CHECKING or DO_SCRIPT_IMPORTS:
    from core.training import Trainer
    from core.storage_management import ModelFileManager
    from collections import OrderedDict
    import torch
    from torch.utils.data import DataLoader
    import logging

    logger = logging.getLogger(__name__)


class Stats(NamedTuple):
    max: float
    min: float
    mean: float
    std: float

def analyze_weights(trainer: 'Trainer'):
    logger.info("Analyzing model weights...")
    for name, param in trainer.model.named_parameters():
        logger.info(
            f"{name}:\n"
            f"\tMax: {param.max()}\n"
            f"\tMin: {param.min()}\n"
            f"\tMean: {param.mean()}\n"
            f"\tStd: {param.std()}"
        )

def analyze_layer_output(trainer : 'Trainer'):
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


@dataclass
class Options:
    model: Path = field(
        metadata=positional(Path, help_="The model to load")
    )

    preferred_checkpoint : str = field(default='best',
                                       metadata=option(str, help_=
                                       "Either 'best' or 'last'. Defaults to 'best'. "
                                       "Specifies which checkpoint to prefer "
                                       "during checkpoint discovery. "
                                       "If the checkpoint option is specified, "
                                       "this option is ignored.")
                                       )

    checkpoint : Path | None = field(default=None,
                                metadata=option(Path, help_=
                                "The checkpoint file to load. If not specified, "
                                "the script will automatically decide which checkpoint to load, "
                                "in accordance to the preferred_checkpoint option."))

def main(options : Options):
    assert options.preferred_checkpoint in ['best', 'last']
    preferred_checkpoint: Literal['best', 'last']
    preferred_checkpoint = options.preferred_checkpoint  # type: ignore
    with ModelFileManager(options.model) as file_manager:
        trainer = Trainer.load_checkpoint(file_manager, options.checkpoint, preferred_checkpoint)
        trainer.model.eval()
        with torch.no_grad():
            analyze_weights(trainer)