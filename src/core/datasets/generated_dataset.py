from . import SplitDataset
import logging
from typing import Callable, Optional
import torch
from torch.utils.data import Dataset
import warnings

logger = logging.getLogger(__name__)

class GeneratedDataset(SplitDataset):
    """A dataset randomly generated on the fly. If a seed is not specified,
        the dataset will be different each time it is loaded.
    """

    class _GeneratorSubset(Dataset):
             
            def __init__(
                self,
                generator_function : Callable[[torch.Generator], tuple[torch.Tensor, torch.Tensor]],
                seed : Optional[int],
                samples : int
            ):
                super().__init__()
                self.generator_function = generator_function
                self.samples = samples
                self.seed = seed
    
            def __len__(self):
                return self.samples
    
            def __getitem__(self, idx):
                rng = torch.Generator()
                if self.seed is not None:
                    rng = torch.Generator().manual_seed(self.seed + idx)
                return self.generator_function(rng)
    
    def __init__(
            self,
            generator_function : Callable[[torch.Generator], tuple[torch.Tensor, torch.Tensor]],
            samples_per_set : tuple[int, int, int],
            seed : Optional[int] = None,
        ):
        """

        Args:
            generator_function (Callable[[torch.Generator], tuple[torch.Tensor, torch.Tensor]]): 
                The function that generates the data
            samples_per_set (tuple[int, int, int]): 
                Number of samples for training, validation and testing
            seed (Optional[int], optional): 
                If specified, makes the dataset reproducible. Defaults to None.
        """
        super().__init__()
        self.generator_function = generator_function
        self.samples_per_set = samples_per_set
        self.train_seed = None
        self.val_seed = None
        self.test_seed = None
        if seed is not None:
            self.train_seed = seed
            self.val_seed = seed + self.samples_per_set[0]
            self.test_seed = seed + self.samples_per_set[0] + self.samples_per_set[1]
        self.train_data = self._GeneratorSubset(self.generator_function, self.train_seed, self.samples_per_set[0])
        self.val_data = self._GeneratorSubset(self.generator_function, self.val_seed, self.samples_per_set[1])
        self.test_data = self._GeneratorSubset(self.generator_function, self.test_seed, self.samples_per_set[2])