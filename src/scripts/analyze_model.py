
from pathlib import Path
from typing import TYPE_CHECKING, Literal

from pandas.compat.pyarrow import pa
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
            f"\tShape: {param.shape}\n"
            f"\tMax:   {param.max()}\n"
            f"\tMin:   {param.min()}\n"
            f"\tMean:  {param.mean()}\n"
            f"\tStd:   {param.std()}"
        )

@dataclass
class LayerStats:
    shape : torch.Size
    sum: torch.Tensor
    max: torch.Tensor
    min: torch.Tensor
    count: int

    def mean(self) -> torch.Tensor:
        return self.sum / self.count

class LayerOutputHook:
    def __init__(self):
        self.layer_stats: OrderedDict[torch.nn.Module, LayerStats] = OrderedDict()

    @classmethod
    def _new_layer(cls, output_shape : torch.Size):
        num_features = output_shape[-1]
        return LayerStats(
            output_shape,
            sum=torch.zeros(num_features),
            max=torch.full((num_features,), float('-inf')),
            min=torch.full((num_features,), float('+inf')),
            count=0
        )

    def __call__(self, module, input, output : torch.Tensor):
        stats = self.layer_stats.setdefault(module, self._new_layer(output.shape))
        if output.shape != stats.shape:
            logger.warning(f"Layer {module} output shape changed from {stats.shape} to {output.shape}")
        output_channels = output.view(-1, output.shape[-1])
        stats.sum += output_channels.sum(dim=0).detach()
        stats.count += output_channels.size(0)
        stats.max = torch.maximum(stats.max, output_channels.max(dim=0)[0].detach())
        stats.min = torch.minimum(stats.min, output_channels.min(dim=0)[0].detach())

def analyze_backward(trainer : 'Trainer'):
    logger.info("Analyzing backward pass and gradients...")
    x, y = next(iter(DataLoader(trainer.try_get_validation_set())))
    logger.info(f"X shape: {x.shape}, Y shape: {y.shape}")
    x = x.to(torch.get_default_device())
    y = y.to(torch.get_default_device())
    trainer.model.train()
    forward_hook = LayerOutputHook()
    trainer.model.register_forward_hook(forward_hook)
    with torch.set_grad_enabled(True):
        pred = trainer.model(x)
        logger.info(f"Prediction shape: {pred.shape}")
        loss = trainer.model.loss(pred, y)
        logger.info(f"Loss: {loss.item()}")
        loss.backward()
        for name, param in trainer.model.named_parameters():
            grad = param.grad
            assert grad is not None
            logger.info(
                f"{name}:\n"
                f"\tMax grad:  {grad.max()}\n"
                f"\tMin grad:  {grad.min()}\n"
                f"\tMean grad: {grad.mean()}\n"
                f"\tStd grad:  {grad.std()}"
            )
    for module, stats in forward_hook.layer_stats.items():
        logger.info(f"Layer {module} output stats:\n"
            f"\tShape: {stats.shape}"
            f"\tMax:   {stats.max}\n"
            f"\tMin:   {stats.min}\n"
            f"\tMean:  {stats.mean()}\n"
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
    skip_weights : bool = field(default=False,
                                metadata=option(bool, help_=
                                "Whether to skip the weights analysis."))
    skip_backward : bool = field(default=False,
                                metadata=option(bool, help_=
                                "Whether to skip the backward pass analysis."))

def main(options : Options):
    assert options.preferred_checkpoint in ['best', 'last']
    preferred_checkpoint: Literal['best', 'last']
    preferred_checkpoint = options.preferred_checkpoint  # type: ignore
    with ModelFileManager(options.model) as file_manager:
        trainer = Trainer.load_checkpoint(file_manager, options.checkpoint, preferred_checkpoint)
        if not options.skip_weights:
            analyze_weights(trainer)
        if not options.skip_backward:
            analyze_backward(trainer)