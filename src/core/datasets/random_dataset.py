from types import UnionType
from . import SplitDataset
import logging
from typing import Callable, Optional
import torch
from torch.utils.data import Dataset, TensorDataset

logger = logging.getLogger(__name__)

SampleGenerator = Callable[[torch.Generator], tuple[torch.Tensor, torch.Tensor]]

def _hashable_tensor_tuple(a : tuple[torch.Tensor, torch.Tensor]) -> tuple:
    return (tuple(a[0].tolist()), tuple(a[1].tolist()))

def _generate_sample_excluding(generator_function : SampleGenerator, rng : torch.Generator, exclude : set) -> tuple[torch.Tensor, torch.Tensor]:
    attempt = 0
    while True:
        sample = generator_function(rng)
        if _hashable_tensor_tuple(sample) not in exclude:
            return sample
        attempt += 1
        if attempt >= 10 and attempt % 10 == 0:
            logger.warning(f"Taking unusually long to generate a not excluded sample: {attempt} attempts")

class RandomDataset(SplitDataset):
    """A randomly generated dataset. 
    Ensures that samples do not repeat over different subsets (training, validation and testing).
    By default, the training dataset will be generated on the fly. 
    Generation on the fly without a set seed will produce different training samples each time.
    Note that in order to prevent repeated samples over different subsets, sample generation time may vary.
    """

    class _GeneratorSubset(Dataset):
             
            def __init__(
                self,
                generator_function : SampleGenerator,
                seed : Optional[int],
                samples : int,
                exclude : set
            ):
                super().__init__()
                self.generator_function = generator_function
                self.samples = samples
                self.seed = seed
                self.exclude = exclude
    
            def __len__(self):
                return self.samples
    
            def __getitem__(self, idx):
                rng = torch.Generator()
                if self.seed is not None:
                    rng = torch.Generator().manual_seed(self.seed + idx)
                return _generate_sample_excluding(self.generator_function, rng, self.exclude)

    def _generate(
        self,
        samples : int,
        seed : Optional[int],
        exclude : set
    ) -> tuple[tuple[torch.Tensor, torch.Tensor], set]:
        if seed is not None:
            rng = torch.Generator().manual_seed(seed)
        else:
            rng = torch.Generator()
        samples_set = set()
        first_sample = _generate_sample_excluding(self.generator_function, rng, exclude)
        features = torch.empty((samples, len(first_sample[0])))
        labels = torch.empty((samples, len(first_sample[1])))
        features[0], labels[0] = first_sample
        samples_set.add(_hashable_tensor_tuple(first_sample))
        for i in range(1, samples):
            sample = _generate_sample_excluding(self.generator_function, rng, samples_set)
            features[i], labels[i] = sample
            samples_set.add(_hashable_tensor_tuple(sample))
        return (features, labels), samples_set


    def __init__(
            self,
            generator_function : SampleGenerator,
            samples_per_set : tuple[int, int, int],
            val_seed : int,
            test_seed : int,
            train_seed : Optional[int] = None,
            on_the_fly : bool = True
        ):
        """

        Args:
            generator_function (Callable[[torch.Generator], tuple[torch.Tensor, torch.Tensor]]): 
                Called to generate each sample. Should return 2 1D tensors: features and labels.
            samples_per_set (tuple[int, int, int]): 
                The number of samples to generate for training, validation and testing.
            val_seed (int): 
                Seed for generating the validation set. Should be different from the other seeds.
            test_seed (int):
                Seed for generating the test set. Should be different from the other seeds.
            train_seed (Optional[int], optional): 
                Optional seed for generating the training set, making it reproducible if present.
                If present, should be different from the other seeds.
                If not present, the default generator will be used.
                Defaults to None.

            on_the_fly (bool, optional): 
                If True, training samples are generated on the fly as requested by the dataloader. Defaults to True.
        """
        super().__init__()
        self.generator_function = generator_function
        self.samples_per_set = samples_per_set
        self.train_seed = train_seed
        self.val_seed = val_seed
        self.test_seed = test_seed
        self.on_the_fly = on_the_fly

    def _load(self):
        if self.loaded:
            return
        logger.info(
f"""Generating random dataset ({sum(self.samples_per_set):_} samples):
Training: \t {self.samples_per_set[0]:_} samples \t seed = {self.train_seed}
Validation: \t {self.samples_per_set[1]:_} samples \t seed = {self.val_seed}
Testing: \t {self.samples_per_set[2]:_} samples \t seed = {self.test_seed}
""")
        logger.debug("Generating test set")
        test_tensors, exclude = self._generate(self.samples_per_set[2], self.test_seed, set())
        self.test_data = TensorDataset(*test_tensors)

        logger.debug("Generating validation set")
        val_tensors, exclude_val = self._generate(self.samples_per_set[1], self.val_seed, exclude)
        self.val_data = TensorDataset(*val_tensors)
        exclude.update(exclude_val)

        if self.on_the_fly:
            logger.debug("Configuring training set for on the fly generation")
            self.train_data = self._GeneratorSubset(
                self.generator_function, self.train_seed, self.samples_per_set[0], exclude)
        else:
            logger.debug("Generating training set")
            train_tensors, _ = self._generate(self.samples_per_set[0], self.train_seed, exclude)
            self.train_data = TensorDataset(*train_tensors)

        logger.info("Generation complete")
        self.loaded = True