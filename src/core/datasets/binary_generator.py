from typing import Optional, Sequence
import logging
import torch
import abc
import numpy as np
from . import random_dataset
from . import SplitDataset
from torch.utils.data import Dataset, Subset
import warnings

logger = logging.getLogger(__name__)

class BinaryASTNode(abc.ABC):
    """An abstract syntax tree node for binary data generation"""

    @abc.abstractmethod
    def __call__(self, generated : torch.Tensor) -> torch.Tensor:
        pass

    def __or__(self, value : 'BinaryASTNode') -> 'BinaryASTNode':
        return _BinaryOpNode(self, value, torch.bitwise_or, '|')
    
    def __and__(self, value : 'BinaryASTNode') -> 'BinaryASTNode':
        return _BinaryOpNode(self, value, torch.bitwise_and, '&')
    
    def __xor__(self, value : 'BinaryASTNode') -> 'BinaryASTNode':
        return _BinaryOpNode(self, value, torch.bitwise_xor, '^')
    
    def __invert__(self) -> 'BinaryASTNode':
        return _UnaryOpNode(self, torch.bitwise_not, '~')



class _GeneratedValue(BinaryASTNode):
    """A node that returns the respective generated value"""

    def __init__(self, idx):
        self.idx = idx
        self.name = None

    def __call__(self, generated : torch.Tensor) -> torch.Tensor:
        return generated[self.idx]

    def unnamed_repr(self):
        return f"GeneratedValue({self.idx})"

    def __repr__(self):
        return self.unnamed_repr() if self.name is None else self.name

    def __eq__(self, value: object) -> bool:
        if not isinstance(value, _GeneratedValue):
            return False
        return self.idx == value.idx

    def __hash__(self) -> int:
        return hash((self.__class__, self.idx))

class _BinaryOpNode(BinaryASTNode):
    """A node that performs a binary operation"""

    def __init__(self, left : BinaryASTNode, right : BinaryASTNode, func, symbol : str):
        self.left = left
        self.right = right
        self.func = func
    
    def __call__(self, generated : torch.Tensor) -> torch.Tensor:
        return self.func(self.left(generated), self.right(generated))

    def __repr__(self) -> str:
        return f"({self.left} {self.func.__name__} {self.right})"

    def __eq__(self, value: object) -> bool:
        if not isinstance(value, _BinaryOpNode):
            return False
        return (
            self.func == value.func and 
            self.left == value.left and 
            self.right == value.right
        )
    
    def __hash__(self) -> int:
        return hash((self.func, self.left, self.right))

class _UnaryOpNode(BinaryASTNode):
    """A node that performs a unary operation"""

    def __init__(self, child : BinaryASTNode, func, symbol : str):
        self.child = child
        self.func = func
        self.symbol = symbol

    def __call__(self, generated : torch.Tensor) -> torch.Tensor:
        return self.func(self.child(generated))

    def __repr__(self) -> str:
        return f"{self.symbol}{self.child}"
    
    def __eq__(self, value: object) -> bool:
        if not isinstance(value, _UnaryOpNode):
            return False
        return self.func == value.func and self.child == value.child

    def __hash__(self) -> int:
        return hash((self.func, self.child))

class BinaryGenerator:
    """Instructions on how to generate a set of variables."""

    def __init__(
            self, 
            to_generate : int, 
            features : list[BinaryASTNode], 
            labels : list[BinaryASTNode]
        ):
        self.to_generate = to_generate
        self.features : list[BinaryASTNode] = features
        self.labels : list[BinaryASTNode] = labels
    
    def generate_random(self, rng : torch.Generator) -> tuple[torch.Tensor, torch.Tensor]:
        return self.generate_samples(torch.randint(0, 2, (self.to_generate,), generator=rng))
    
    def generate_from_int(self, value : int) -> tuple[torch.Tensor, torch.Tensor]:
        bits = torch.tensor([int(x) for x in f"{value:0{self.to_generate}b}"])
        return self.generate_samples(bits)
    
    def generate_samples(self, attribution : torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = torch.stack([node(attribution) for node in self.features])
        labels = torch.stack([node(attribution) for node in self.labels])
        return features.to(torch.get_default_dtype()), labels.to(torch.get_default_dtype())

    def as_complete_dataset(self, split : tuple[float, float] | float = (0.8, 0.1), shuffle : bool = True, seed : Optional[int] = None) -> SplitDataset:
        class _CompleteDataset(Dataset):
            def __init__(self, generator : BinaryGenerator):
                self.generator = generator
            def __getitem__(self, index):
                return self.generator.generate_from_int(index)
        complete_dataset = _CompleteDataset(self)
        indices : Sequence[int]
        if shuffle:
            rng = np.random.default_rng(seed=seed)
            indices = rng.permutation(len(self)) # type: ignore
        else:
            indices = range(len(self))
        train_bound, val_bound = SplitDataset._split(len(self), SplitDataset._cast_splits(split))
        train_indices = indices[:train_bound]
        val_indices = indices[train_bound:val_bound]
        test_indices = indices[val_bound:]
        return SplitDataset(
            Subset(complete_dataset, train_indices),
            Subset(complete_dataset, val_indices),
            Subset(complete_dataset, test_indices)
        )


    def __len__(self) -> int:
        return 2**self.to_generate

class BinaryGeneratorBuilder:
    """A generator of binary data"""


    def __init__(self):
        self._to_generate = 0
        self.features : dict[str, BinaryASTNode] = {}
        self.labels : dict[str, BinaryASTNode] = {}
        pass

    def simplify(
        self,
        remove_unused : bool = True
    ):
        raise NotImplementedError("Simplification is not yet implemented")


    def gen_var(self) -> BinaryASTNode:
        """Define a new variable to generate

        Returns:
            BinaryASTNode: A reference to variable that will be generated
        """
        idx = self._to_generate
        self._to_generate += 1
        return _GeneratedValue(idx)
    
    def build(self) -> BinaryGenerator:
        """Builds the generator

        Returns:
            BinaryGenerator: The generator
        """
        sorted_features = [node for _, node in sorted(self.features.items(), key=lambda x: x[0])]
        sorted_labels = [node for _, node in sorted(self.labels.items(), key=lambda x: x[0])]
        
        return BinaryGenerator(self._to_generate, sorted_features, sorted_labels)
    
    def clone(self) -> 'BinaryGeneratorBuilder':
        clone = BinaryGeneratorBuilder()
        clone._to_generate = self._to_generate
        clone.features = self.features
        clone.labels = self.labels
        return clone

    def __repr__(self) -> str:
        def process_variables(variables):
            gen = []
            other = []
            for name, node in variables:
                if isinstance(node, _GeneratedValue):
                    gen.append((name, node))
                    node.name = name
                else:
                    other.append((name, node))
            return gen, other
        gen_feats, other_feats = process_variables(self.features.items())
        gen_labels, other_labels = process_variables(self.labels.items())
        def format_definitions(gen, other):
            defs = (
                [f"\t\t{name} = {node.unnamed_repr()}" for name, node in gen] +
                [f"\t\t{name} = {node}" for name, node in other]
            )
            return ",\n".join(defs)
        return (
f"""{self.__class__.__name__}(
\tto_generate={self._to_generate},
\tFeatures: 
{format_definitions(gen_feats, other_feats)},
\tLabels:
{format_definitions(gen_labels, other_labels)}
)""")

    def without_labels(self, *labels) -> 'BinaryGeneratorBuilder':
        """Removes the specified labels

        Returns:
            itself for chaining
        """
        for l in labels:
            if l not in self.labels:
                warnings.warn(f"Unrecognized label {l}")
        self.labels = {k : v for k, v in self.labels.items() if k not in labels}
        return self
    
    def without_features(self, *features) -> 'BinaryGeneratorBuilder':
        """Removes the specified features

        Returns:
            itself for chaining
        """
        for f in features:
            if f not in self.features:
                warnings.warn(f"Unrecognized feature {f}")
        self.features = {k : v for k, v in self.features.items() if k not in features}
        return self

    def with_labels(self, *labels) -> 'BinaryGeneratorBuilder':
        """Removes all but the specified labels

        Returns:
            itself for chaining
        """
        for l in labels:
            if l not in self.labels:
                warnings.warn(f"Unrecognized label {l}")
        self.labels = {k : v for k, v in self.labels.items() if k in labels}
        return self

    def with_features(self, *features) -> 'BinaryGeneratorBuilder':
        """Removes all but the specified features

        Returns:
            itself for chaining
        """
        for f in features:
            if f not in self.features:
                warnings.warn(f"Unrecognized feature {f}")
        self.features = {k : v for k, v in self.features.items() if k in features}
        
        return self

    def implied_by(self, node : BinaryASTNode) -> BinaryASTNode:
        """Utility function for defining a variable that is implied by another.
        It is a shorthand for `generator.gen_var() | node`

        Args:
            node (BinaryASTNode): The implying node

        Returns:
            The implied node
        """
        return self.gen_var() | node
