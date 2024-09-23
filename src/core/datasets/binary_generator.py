from types import UnionType
import logging
import torch
import abc
from random_dataset import RandomDataset
from . import SplitDataset
import warnings

logger = logging.getLogger(__name__)

class BinaryASTNode(abc.ABC):
    """An abstract syntax tree node for binary data generation"""

    @abc.abstractmethod
    def __call__(self, generated : torch.Tensor) -> torch.Tensor:
        pass


    def __or__(self, value : 'BinaryASTNode') -> 'BinaryASTNode':
        return _BinaryOpNode(self, value, torch.bitwise_or)
    
    def __and__(self, value : 'BinaryASTNode') -> 'BinaryASTNode':
        return _BinaryOpNode(self, value, torch.bitwise_and)
    
    def __xor__(self, value : 'BinaryASTNode') -> 'BinaryASTNode':
        return _BinaryOpNode(self, value, torch.bitwise_xor)
    
    def __invert__(self) -> 'BinaryASTNode':
        return _UnaryOpNode(self, torch.bitwise_not)
    
    def __eq__(self, value : 'BinaryASTNode') -> 'BinaryASTNode':
        return _BinaryOpNode(self, value, torch.eq)
    
    def __ne__(self, value : 'BinaryASTNode') -> 'BinaryASTNode':
        return _BinaryOpNode(self, value, torch.ne)



class _GeneratedValue(BinaryASTNode):
    """A node that returns the respective generated value"""

    def __init__(self, idx):
        self.idx = idx

    def __call__(self, generated : torch.Tensor) -> torch.Tensor:
        return generated[self.idx]

class _BinaryOpNode(BinaryASTNode):
    """A node that performs a binary operation"""

    def __init__(self, left : BinaryASTNode, right : BinaryASTNode, func):
        self.left = left
        self.right = right
        self.func = func
    
    def __call__(self, generated : torch.Tensor) -> torch.Tensor:
        return self.func(self.left(generated), self.right(generated))

class _UnaryOpNode(BinaryASTNode):
    """A node that returns the NOT of its child"""

    def __init__(self, child : BinaryASTNode, func):
        self.child = child
        self.func = func

    def __call__(self, generated : torch.Tensor) -> torch.Tensor:
        return self.func(self.child(generated))

class _CompleteBinaryGeneratedDataset(SplitDataset):

    def __init__(
            self, 
            splits : tuple[float, float] | float, 
            to_generate : int, 
            binary_generator : 'BinaryGeneratorBuilder',
            shuffle : bool = True,
            random_state = None
        ):
        super().__init__()
        self.splits : tuple[float, float] = self._cast_splits(splits)
        self.to_generate = to_generate
        if self.to_generate > 32:
            warnings.warn(f"Generating high number of random variables {self.to_generate}, may not work")
        self.total_samples = 2 ** to_generate
        self.binary_generator = binary_generator
        self.shuffle = shuffle
        self.random_state = random_state

    def bit_range(self) -> torch.Tensor:
        tensor = torch.empty((self.total_samples, self.to_generate), dtype=torch.bool)
        
        for i in range(self.total_samples):
            for j in range(self.to_generate):
                tensor[i] = i & (1 << j) > 0
        return tensor

    def _load(self):
        if self.loaded:
            return
        
        logger.info(f"Generating complete binary dataset with {self.to_generate} random variables ({self.total_samples:_} samples)")
        generated = self.bit_range()
        logger.debug("Random variables assigned")

        if self.shuffle:
            logger.debug("Shuffling dataset")
            rng = torch.Generator()
            if self.random_state is not None:
                rng.manual_seed(self.random_state)
            generated = generated[torch.randperm(self.total_samples, generator=rng)]
        
        logger.debug("Generating features and labels")
        self.data = self.binary_generator.generate_samples(generated)
        logger.info("Generation complete")

        train_bound, val_bound = self._split(self.to_generate, self.splits)
        logger.info(f"""
Splitting complete binary dataset with {self.to_generate} random variables ({self.total_samples:_} samples):
Training: \t [0, {train_bound}[ \t ({train_bound:_} samples, {self.splits[0] * 100}%)
Validation: \t [{train_bound}, {val_bound}[ \t ({val_bound - train_bound:_} samples, {self.splits[1] * 100}%)
Testing: \t [{val_bound}, {self.total_samples}[ \t ({self.total_samples - val_bound:_} samples, {(1 - self.splits[0] - self.splits[1]) * 100}%)
Num features: \t {len(self.binary_generator.features)}
Num target: \t {len(self.binary_generator.labels)}
Shuffle: \t {self.shuffle}
Seed: \t {self.random_state}""")

        #logger.info()
        self.loaded = True
    

class BinaryGeneratorBuilder:
    """A generator of binary data"""


    def __init__(self):
        self._to_generate = 0
        self.features = {}
        self.labels = {}
        pass

    def gen_var(self):
        idx = self._to_generate
        self._to_generate += 1
        return _GeneratedValue(idx)
    
    def generate_samples(self, generated : torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = torch.stack([node(generated) for node in self.features.values()], dim=1)
        labels = torch.stack([node(generated) for node in self.labels.values()], dim=1)
        return features, labels

    def to_random_dataset(self, samples : tuple[int, int, int], seed = None) -> RandomDataset:
        def generator(rng : torch.Generator):
            generated = torch.randint(0, 2, (self._to_generate,), generator=rng)
            return self.generate_samples(generated)
        return RandomDataset(generator, samples, seed)
    
    def clone(self) -> 'BinaryGeneratorBuilder':
        clone = BinaryGeneratorBuilder()
        clone._to_generate = self._to_generate
        clone.features = self.features
        clone.labels = self.labels
        return clone


