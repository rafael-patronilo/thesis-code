
import torch

def overlay_heatmap(
        images : torch.Tensor,
        importance_maps : torch.Tensor,
        alpha : float = 0.5,
        max_color : torch.Tensor = torch.tensor([1.0, 0.0, 0.0]),
        min_color : torch.Tensor = torch.tensor([0.0, 0.0, 1.0])
):
    assert importance_maps.ndim == 4, "importance_maps should have shape (batch_size, 1, height, width)"
    assert images.ndim == 4, "images should have shape (batch_size, 1, height, width)"
    assert importance_maps.size(0) == images.size(0), "batch size must be the same for images and importance maps"
    assert (importance_maps.size(2) == images.size(2) and
            importance_maps.size(3) == images.size(3)), "width and height should be the same"

    max_color = max_color.expand_as(importance_maps)
    min_color = min_color.expand_as(importance_maps)

    return (
        images * (1.0-alpha) +
        torch.lerp(min_color, max_color, importance_maps) * alpha
    )