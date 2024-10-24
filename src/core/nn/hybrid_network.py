from typing import Iterator
import torch
from torch.nn import Module as NNModule
from torch.nn.parameter import Parameter

class HybridNetwork(NNModule):
    def __init__(
            self, 
            reasoning_network : NNModule, 
            perception_network : NNModule
    ):
        super(HybridNetwork, self).__init__()
        self.reasoning_network = reasoning_network
        self.perception_network = perception_network
    
    def forward(self, x):
        x = self.perception_network(x)
        x = self.reasoning_network(x)
        return x

    def parameters(self, recurse: bool = True, perception_only : bool = True):
        if perception_only:
            return self.perception_network.parameters(recurse)
        else:
            return super().parameters(recurse)
        
    def named_parameters(
            self, 
            prefix: str = '', 
            recurse: bool = True, 
            remove_duplicate: bool = True,
            perception_only : bool = True
        ) -> Iterator[tuple[str, Parameter]]:
        if perception_only:
            return self.perception_network.named_parameters(prefix, recurse, remove_duplicate)
        else:
            return super().named_parameters(prefix, recurse, remove_duplicate)
