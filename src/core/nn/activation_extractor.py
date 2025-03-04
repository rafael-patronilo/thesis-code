from typing import Callable, Optional, Protocol

import torch
from torch import nn
import logging
logger = logging.getLogger(__name__)

class ActivationExtractionHook:
    def __init__(self, outputs : dict[nn.Module, torch.Tensor]):
        self._outputs = outputs

    def __call__(self, module, input, output):
        self._outputs[module] = output

class NeuronSelector(Protocol):
    def __call__(self, tensor : torch.Tensor) -> torch.Tensor:
        ...

class ActivationExtractor(nn.Module):
    """
    Utility module that calls an inner module and outputs the activations of given hidden layers
    """
    def __init__(
            self,
            model : nn.Module,
            layers : list[nn.Module]
    ):
        super().__init__()
        self.inner_model = model
        self.layers = layers
        self.outputs : dict[nn.Module, torch.Tensor] = {}
        self.extraction_hook = ActivationExtractionHook(self.outputs)

        submodules = set(model.modules())
        for layer in self.layers:
            if layer not in submodules and layer is not model:
                logger.error(f"Layer is not a submodule of the model;\n"
                                f"Activation extraction will probably fail.\n"
                                f"Model:\n{model}"
                                f"Layer:\n{layer}")
            layer.register_forward_hook(self.extraction_hook)

    def forward(self, x):
        self.outputs.clear()
        _ = self.inner_model(x)
        batch_size = None
        output_tensors = []
        for layer in self.layers:
            layer_output = self.outputs.get(layer)
            if layer_output is None:
                raise ValueError(f"Layer {layer} emitted no output")
            if batch_size is None:
                batch_size = layer_output.shape[0]
            elif batch_size != layer_output.shape[0]:
                raise ValueError(f"All layers should have the same size at dimension 0 but layer {layer}")
            output_tensors.append(layer_output)
        return torch.hstack(output_tensors)