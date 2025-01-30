"""
Implementation of andomized Input Sampling for Explanation

V. Petsiuk, A. Das, and K. Saenko. “RISE: Randomized Input Sampling for Expla-
nation of Black-box Models”. In: CoRR abs/1806.07421 (2018). arXiv: 1806.07421.
url: https://arxiv.org/abs/1806.07421
"""
from typing import Optional, Protocol

import torch
from torch import nn

class RISEMaskGenerator(Protocol):
    def __call__(self, n_masks : int, height : int, width : int, p : float) -> torch.Tensor:
        ...


class UniformMaskGenerator:
    def __init__(self, rng: Optional[torch.Generator] = None):
        self.rng = rng

    def __call__(self, n_masks: int, height: int, width: int, p : float) -> torch.Tensor:
        noise = torch.rand(n_masks, 1, height, width, generator=self.rng)
        return torch.where(noise < p, 1.0, 0.0)

class RISE:
    def __init__(
            self,
            image_height : int,
            image_width : int,
            mask_size_factor : float = 0.9,
            n_masks : int = 4000,
            batch_size : int = 32,
            generator : Optional[RISEMaskGenerator] = None,
            probability : float = 0.5
    ):
        self.mask_size_factor = mask_size_factor
        self.n_masks = n_masks
        self.image_height = image_height
        self.image_width = image_width
        self.mask_generator = generator or UniformMaskGenerator()
        self.probability = probability
        self.batch_size = batch_size

    def generate(
            self,
            model : nn.Module,
            images : torch.Tensor,
            output_index : Optional[int] = None) -> torch.Tensor:
        assert images.ndim == 4, "Expected images to have shape (n_images, n_channels, height, width)"
        assert images.shape[2] == self.image_height, f"Expected images to have height {self.image_height}"
        assert images.shape[3] == self.image_width, f"Expected images to have width {self.image_width}"
        i = 0
        num_images = images.shape[0]
        sums = torch.zeros(num_images, self.image_height, self.image_width)
        with torch.no_grad():
            while i < self.n_masks:
                batch_masks = num_images * min(self.batch_size, self.n_masks - i)
                masks = self.mask_generator(batch_masks, self.image_height, self.image_width, self.probability)
                masked_images = images * masks
                output = model(masked_images)
                if output_index is not None:
                    output = output[:, output_index]
                sums += output * masks
                i += batch_masks
        return sums / (self.n_masks * self.probability)





