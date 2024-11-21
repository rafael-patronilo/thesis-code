from typing import Optional, Sequence
import logging
import torch
import abc
import numpy as np
from . import random_dataset
from . import SplitDataset, CollumnReferences
from torch.utils.data import Dataset, Subset
import warnings

logger = logging.getLogger(__name__)

class BinaryASTNode(abc.ABC):
    """An abstract syntax tree node for binary data generation"""

    @abc.abstractmethod
    def __call__(self, generated : torch.Tensor, force_valid : bool) -> torch.Tensor:
        pass

    def is_valid(self, generated : torch.Tensor) -> bool:
        return bool(self(generated, True) == self(generated, False))

    def __or__(self, value : 'BinaryASTNode') -> 'BinaryASTNode':
        return _BinaryOpNode(self, value, torch.bitwise_or, '|')
    
    def __and__(self, value : 'BinaryASTNode') -> 'BinaryASTNode':
        return _BinaryOpNode(self, value, torch.bitwise_and, '&')
    
    def __xor__(self, value : 'BinaryASTNode') -> 'BinaryASTNode':
        return _BinaryOpNode(self, value, torch.bitwise_xor, '^')
    
    def __invert__(self) -> 'BinaryASTNode':
        return _UnaryOpNode(self, torch.bitwise_not, '~')



class _FreeVariable(BinaryASTNode):
    """A node that returns the respective generated value"""

    def __init__(self, idx):
        self.idx = idx
        self.name = None

    def __call__(self, generated : torch.Tensor, force_valid : bool) -> torch.Tensor:
        return generated[self.idx]

    def unnamed_repr(self):
        return f"FreeVariable({self.idx})"

    def __repr__(self):
        return self.unnamed_repr() if self.name is None else self.name

    def __eq__(self, value: object) -> bool:
        if not isinstance(value, _FreeVariable):
            return False
        return self.idx == value.idx

    def __hash__(self) -> int:
        return hash((self.__class__, self.idx))

class _ImpliedVariable(BinaryASTNode):
    """A node that is implied by another"""

    def __init__(self, idx : int, node : BinaryASTNode):
        self.idx = idx
        self.node = node

    def __call__(self, generated : torch.Tensor, force_valid : bool) -> torch.Tensor:
        implication_value = self.node(generated, force_valid)
        if force_valid:
            return generated[self.idx] | implication_value
        else:
            return generated[self.idx]

    def __repr__(self) -> str:
        return f"ImpliedBy({self.idx}, {self.node})"

    def __eq__(self, value: object) -> bool:
        if not isinstance(value, _ImpliedVariable):
            return False
        return self.idx == value.idx and self.node == value.node

    def __hash__(self) -> int:
        return hash((self.__class__, self.idx, self.node))

class _BinaryOpNode(BinaryASTNode):
    """A node that performs a binary operation"""

    def __init__(self, left : BinaryASTNode, right : BinaryASTNode, func, symbol : str):
        self.left = left
        self.right = right
        self.func = func
        self.symbol = symbol
    
    def __call__(self, generated : torch.Tensor, force_valid : bool) -> torch.Tensor:
        return self.func(self.left(generated, force_valid), self.right(generated, force_valid))

    def __repr__(self) -> str:
        return f"({self.left} {self.symbol} {self.right})"

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

    def __call__(self, generated : torch.Tensor, force_valid : bool) -> torch.Tensor:
        return self.func(self.child(generated, force_valid))

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
            labels : list[BinaryASTNode],
            feature_names : Optional[list[str]] = None,
            label_names : Optional[list[str]] = None,
            validation_node : Optional[BinaryASTNode] = None,
            valid_label : str = 'valid'
        ):
        self.to_generate = to_generate
        self.features : list[BinaryASTNode] = features
        self.labels : list[BinaryASTNode] = labels
        self.feature_names = feature_names
        self.valid_label = valid_label
        self.label_names = label_names + [valid_label] if label_names is not None else None
        self.validation_node = validation_node

    def get_collumn_references(self) -> CollumnReferences:
        raise NotImplementedError("Collumn references not available")
    
    def _attach_collumn_references[T : SplitDataset](self, dataset : T) -> T:
        # TODO
        #col_refs = self.get_collumn_references()
        #if col_refs is not None:
        #    dataset.collumn_references = col_refs
        return dataset
    
    def generate_random(self, rng : torch.Generator) -> tuple[torch.Tensor, torch.Tensor]:
        return self.generate_samples(torch.randint(0, 2, (self.to_generate,), generator=rng, device='cpu'))
    
    def generate_from_int(self, value : int, force_valid : bool = True) -> tuple[torch.Tensor, torch.Tensor]:
        bits = torch.tensor([int(x) for x in f"{value:0{self.to_generate}b}"], device='cpu')
        return self.generate_samples(bits, force_valid)
    
    def _attach_valid_label(self, attribution : torch.Tensor, force_valid : bool, labels):
        valid = None
        if self.validation_node is not None:
            valid = self.validation_node(attribution, force_valid)
        if valid is not False and not force_valid:
            valid = (
                all(node.is_valid(attribution) for node in self.features) and 
                all(node.is_valid(attribution) for node in self.labels)
            )
        if valid is not None:
            labels.append(torch.tensor(1 if valid else 0, device='cpu'))
        return labels

    def generate_samples(self, attribution : torch.Tensor, force_valid : bool = True) -> tuple[torch.Tensor, torch.Tensor]:
        features = torch.stack([node(attribution, force_valid) for node in self.features])
        labels = torch.stack(self._attach_valid_label(
            attribution,
            force_valid,
            [node(attribution, force_valid) for node in self.labels]
        ))
        return features.to(torch.get_default_dtype()), labels.to(torch.get_default_dtype())

    def as_complete_dataset(
            self, 
            split : tuple[float, float] | float = (0.8, 0.1),
            force_valid : bool = True,
            shuffle : bool = True, 
            seed : Optional[int] = None
        ) -> SplitDataset:
        class _CompleteDataset(Dataset):
            def __init__(self, generator : BinaryGenerator):
                self.generator = generator
            def __getitem__(self, index):
                return self.generator.generate_from_int(index, force_valid)
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
        return self._attach_collumn_references(
            SplitDataset(
                Subset(complete_dataset, train_indices),
                Subset(complete_dataset, val_indices),
                Subset(complete_dataset, test_indices)
            )
        )

    def as_random_dataset(
            self, 
            sizes : tuple[int, int, int], 
            test_seed : int,
            val_seed : int,
            train_seed : Optional[int] = None,
            on_the_fly : bool = True) -> random_dataset.RandomDataset:
        return self._attach_collumn_references(
            random_dataset.RandomDataset(
                self.generate_random,
                sizes,
                test_seed,
                val_seed,
                train_seed,
                on_the_fly
            )
        )

    def __len__(self) -> int:
        return 2**self.to_generate

class BinaryGeneratorBuilder:
    """A generator of binary data"""


    def __init__(self):
        self._to_generate = 0
        self.features : dict[str, BinaryASTNode] = {}
        self.labels : dict[str, BinaryASTNode] = {}
        self.validation_node : Optional[BinaryASTNode] = None
        self.valid_label = 'valid'

    def simplify(
        self,
        remove_unused : bool = True
    ):
        # TODO implement simplification?
        raise NotImplementedError("Simplification is not yet implemented")

    def _gen_idx(self) -> int:
        idx = self._to_generate
        self._to_generate += 1
        return idx

    def free_variable(self) -> BinaryASTNode:
        """Define a new variable to generate

        Returns:
            BinaryASTNode: A reference to variable that will be generated
        """
        return _FreeVariable(self._gen_idx())
    
    
    def implied_by(self, node : BinaryASTNode) -> BinaryASTNode:
        """Defines a node that is logically implied by another.
        This is different from `free_variable() | node` because the implication may be broken 
            if generating invalid samples is allowed.

        Args:
            node (BinaryASTNode): The implying node

        Returns:
            The implied node
        """
        return _ImpliedVariable(self._gen_idx(), node)

    def require(self, node : BinaryASTNode):
        """Adds a node that must be valid. 
        If called multiple times it considers a conjunction of all requirements.

        Args:
            node (BinaryASTNode): The node to require

        """
        if self.validation_node is None:
            self.validation_node = node
        else:
            self.validation_node = self.validation_node & node
    
    def build(self) -> BinaryGenerator:
        """Builds the generator

        Returns:
            BinaryGenerator: The generator
        """
        sorted_feature = sorted(self.features.items(), key=lambda x: x[0])
        sorted_label = sorted(self.labels.items(), key=lambda x: x[0])
        sorted_feature_nodes = [node for _, node in sorted_feature]
        sorted_label_nodes = [node for _, node in sorted_label]
        sorted_feature_names = [name for name, _ in sorted_feature]
        sorted_label_names = [name for name, _ in sorted_label]
        
        return BinaryGenerator(
            self._to_generate, 
            sorted_feature_nodes, 
            sorted_label_nodes,
            feature_names=sorted_feature_names,
            label_names=sorted_label_names,
            validation_node=self.validation_node,
            valid_label=self.valid_label
        )
    
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
                if isinstance(node, _FreeVariable):
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
    
    def with_valid_label(self, label : str) -> 'BinaryGeneratorBuilder':
        """Sets the name of the valid label

        Returns:
            itself for chaining
        """
        self.valid_label = label
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