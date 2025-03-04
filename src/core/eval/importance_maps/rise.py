"""
Implementation of andomized Input Sampling for Explanation

V. Petsiuk, A. Das, and K. Saenko. “RISE: Randomized Input Sampling for Expla-
nation of Black-box Models”. In: CoRR abs/1806.07421 (2018). arXiv: 1806.07421.
url: https://arxiv.org/abs/1806.07421
"""
from math import ceil
from typing import Optional, Protocol, final

from pandas.io.sql import overload
import torch
from torch import Tensor, device, nn
import torchvision

from core.util.progress_trackers import NULL_PROGRESS_TRACKER, ProgressTracker

class RISEMaskGenerator(Protocol):
    def gen_masks(self, n_masks : int, height : int, width : int, p : float) -> torch.Tensor:
        """
        Generate `n_masks` masks with shape (`height`, `width`).
        A mask is a tensor with shape (1, `height`, `width`) where each element is either
        1 with probability p or 0.

        :param n_masks: number of masks to generate
        :param height: height of the masks
        :param width: width of the masks
        :param p: probability of each mask cell being 1
        :return: a tensor of shape (n_masks, 1, height, width) containing the generated masks
        """
        ...

    def gen_identations(self, n : int, upper_bound : tuple[int, int] | torch.Tensor) -> torch.Tensor:
        """
        Generate `n` random indentations up to `upper_bound` (inclusive).
        Each indentation is a tensor between (0, 0) and (`upper_bound`[0], `upper_bound`[1]) with the y, x coordinates.
        :param n: the number of random indentations to generate
        :param upper_bound: the max y, x coordinates of each indentation (inclusive)
        :return: a tensor of shape (n, 2) containing the generated indentations
        """
        ...

class UniformMaskGenerator(RISEMaskGenerator):

    @overload
    def __init__(self):
        ...

    @overload
    def __init__(self, rng: torch.Generator) -> None:
        ...

    @overload
    def __init__(self, *, seed: int) -> None:
        ...


    def __init__(
            self,
            rng: Optional[torch.Generator] = None,
            seed : Optional[int] = None
    ):
        if seed is not None and rng is None:
            self.rng = torch.Generator(device=torch.get_default_device())
            self.rng.manual_seed(seed)
        else:
            self.rng = rng

    def gen_masks(self, n_masks: int, height: int, width: int, p : float) -> torch.Tensor:
        noise = torch.rand(n_masks, 1, height, width, generator=self.rng)
        return torch.where(noise < p, 1.0, 0.0)

    def gen_identations(self, n : int, upper_bound : tuple[int, int] | torch.Tensor) -> torch.Tensor:
        def gen_int(bound : int | Tensor) -> torch.Tensor:
            int_bound : int
            if isinstance(bound, Tensor):
                int_bound =  bound.int().item() # type: ignore
            else:
                int_bound = bound
            return torch.randint(0, int_bound, (n, 1), generator=self.rng)
        y = gen_int(upper_bound[0]).unsqueeze(-1)
        x = gen_int(upper_bound[1]).unsqueeze(-1)
        return torch.hstack([y, x])

class RISE:
    def __init__(
            self,
            mask_cell_size : int | tuple[int, int] = 32,
            n_masks : int = 4000,
            batch_size : int = 32,
            generator : Optional[RISEMaskGenerator] = None,
            probability : float = 0.5
    ):
        self.mask_cell_size : tuple[int, int] = (
            mask_cell_size
            if isinstance(mask_cell_size, tuple)
            else (mask_cell_size, mask_cell_size)
        )
        self.n_masks = n_masks
        self.mask_generator = generator or UniformMaskGenerator()
        self.probability = probability
        self.batch_size = batch_size


    def _generate_masks(
            self,
            n_images: int,
            n_masks: int,
            image_height : int,
            image_width : int
    ) -> torch.Tensor:
        """
        Generate masks by upsampling noise as described in the RISE paper
        :param n_images: number of images to generate masks for
        :param n_masks: number of masks to generate for each image
        :param image_height: the height of the image to mask
        :param image_width: the width of the image to mask
        :return: a mask tensor of dimensions (n_images, n_masks, 1, image_height, image_width)
        """
        total_masks = n_images * n_masks
        c_h, c_w = self.mask_cell_size
        h = ceil(image_height / c_h)
        w = ceil(image_width / c_w)
        small_masks = self.mask_generator.gen_masks(total_masks, h, w, self.probability)
        upsample_size = ((h+1) * c_h, (w+1) * c_w)
        upsampled = nn.functional.interpolate(
            small_masks, size=upsample_size, mode='bilinear'
        )
        cropped = torch.empty(total_masks, 1, image_height, image_width)
        idents = self.mask_generator.gen_identations(total_masks, (c_h, c_w))
        for i in range(total_masks):
            y, x = idents[i][0].item(), idents[i][1].item()
            cropped[i] = upsampled[i, :, y:y+image_height, x:x+image_width]

        return cropped.view(n_images, n_masks, 1, image_height, image_width)

    def generate(
            self,
            model : nn.Module,
            images : torch.Tensor,
            progress_tracker : ProgressTracker = NULL_PROGRESS_TRACKER
    ) -> torch.Tensor:
        """
        Generate importance maps for the given images using the given model
        :param model: the model to generate importance maps for
        :param images: the images to generate importance maps for, with shape (n_images, n_channels, height, width)
        :param normalize: whether to max min normalize the importance maps in the end
        :param progress_tracker: an optional progress tracker to use
        :return: the importance maps for the given images, in a tensor of shape (n_images, n_outputs, height, width)
        """
        assert images.ndim == 4, "Expected images to have shape (n_images, n_channels, height, width)"
        n_channels = images.shape[1]
        image_height = images.shape[2]
        image_width = images.shape[3]
        i = 0
        num_images = images.shape[0]
        sums : torch.Tensor | None = None
        mask_sums : torch.Tensor | None = None
        with torch.no_grad():
            while i < self.n_masks:
                batch_masks = min(self.batch_size, self.n_masks - i)
                masks = self._generate_masks(num_images, batch_masks,
                                             image_height, image_width)
                images_view = images.view(num_images, 1, n_channels, image_height, image_width)
                masked_images = images_view * masks
                masked_images = masked_images.view(num_images * batch_masks,
                                                   n_channels, image_height, image_width)
                output = model(masked_images)
                num_outputs = output.shape[1]
                output = output.view(num_images, batch_masks, num_outputs, 1, 1)
                local_sum = (output * masks).sum(dim=1, keepdim=False)
                local_mask_sum = masks.sum(dim=1, keepdim=False)
                if sums is None:
                    sums = local_sum
                    mask_sums = local_mask_sum
                else:
                    sums += local_sum
                    mask_sums += local_mask_sum
                progress_tracker.tick(batch_masks)
                i += batch_masks
        assert sums is not None

        result = sums / mask_sums
        for i in range(result.shape[0]):
            for j in range(result.shape[1]):
                min_ = result[i,j].min()
                max_ = result[i,j].max()
                result[i,j] = (result[i,j] - min_) / (max_ - min_)
        return result