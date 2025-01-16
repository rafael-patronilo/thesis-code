
from typing import Iterable
from torch.nn import Module, Parameter, Sequential

class PartiallyPretrained(Sequential):
    def __init__(self, *layers : tuple[Module, bool]):
        layers_only = [layer[0] for layer in layers]
        super().__init__(*layers_only)
        self.pretrained = [layer[1] for layer in layers]

    def pre_trained_parameters(self) -> Iterable[Parameter]:
        for layer, is_pretrained in zip(self, self.pretrained):
            if is_pretrained:
                yield from layer.parameters()

    def untrained_parameters(self) -> Iterable[Parameter]:
        for layer, is_pretrained in zip(self, self.pretrained):
            if not is_pretrained:
                yield from layer.parameters()
