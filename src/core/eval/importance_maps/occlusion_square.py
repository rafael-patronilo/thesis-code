from dataclasses import dataclass
from typing import Optional
import torch
from torch import nn

from core.util.progress_trackers import NULL_PROGRESS_TRACKER, ProgressTracker


class OcclusionSquare:
    def __init__(
            self,
            square_size : int = 16,
            stride : int = 1,
            same_padding : bool = True,
            square_color : torch.Tensor = torch.tensor([0.5, 0.5, 0.5])
    ):
        self.square_size = square_size
        self.stride = stride
        self.same_padding = same_padding
        self.square_color = square_color

    def generate(
            self,
            model : nn.Module,
            images : torch.Tensor,
            normalize : bool = True,
            progress_tracker : ProgressTracker = NULL_PROGRESS_TRACKER
    ) -> torch.Tensor:
        assert images.ndim == 4, \
            "Expected images to have shape (n_images, n_channels, height, width)"
        assert images.shape[1] == self.square_color.shape[0], \
            f"Mismatching number of channels between image ({images.shape[1]}) and square color ({self.square_color.shape[0]})"
        if self.same_padding:
            start = -self.square_size
            height_stop = images.shape[2]
            width_stop = images.shape[3]
        else:
            start = 0
            height_stop = images.shape[2] - self.square_size
            width_stop = images.shape[3] - self.square_size

        @dataclass
        class OcclusionMaps:
            sums: torch.Tensor
            counts: torch.Tensor
        occlusion_maps : Optional[OcclusionMaps] = None

        for i in range(start, height_stop, self.stride):
            for j in range(start, width_stop, self.stride):
                i_bounds = slice(max(0, i), min(i + self.square_size, images.shape[2]))
                j_bounds = slice(max(0, j), min(j + self.square_size, images.shape[3]))
                images_copy = images.clone()
                images_copy[:, :, i_bounds, j_bounds] = self.square_color

                with torch.no_grad():
                    output = model(images_copy)
                mask = torch.zeros(
                    images.shape[0], output.shape[1], images.shape[2], images.shape[3])
                mask[:, :, i_bounds, j_bounds] = 1.0
                if occlusion_maps is None:
                    occlusion_maps = OcclusionMaps(mask * output, mask)
                else:
                    occlusion_maps.sums += mask * output
                    occlusion_maps.counts += mask
                progress_tracker.tick()
        if occlusion_maps is None:
            raise ValueError("No occlusion maps were generated")
        result = occlusion_maps.sums / occlusion_maps.counts
        if normalize:
            result = result - result.min()
            result = result / result.max()
        return result


