from typing import Iterator
import torch
from torch.nn import Module as NNModule
from torch.nn.parameter import Parameter

class HybridNetwork(NNModule):
    def __init__(
            self,
            perception_network: NNModule,
            reasoning_network : NNModule,
            output_includes_concepts : bool = False
    ):
        super(HybridNetwork, self).__init__()
        self.perception_network = perception_network
        self.reasoning_network = reasoning_network
        self.output_includes_concepts = output_includes_concepts
    
    def forward(self, x):
        concepts = self.perception_network(x)
        classes = self.reasoning_network(concepts)
        if self.output_includes_concepts:
            return torch.hstack((concepts, classes))
        return classes

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
