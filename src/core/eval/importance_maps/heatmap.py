import torch
from matplotlib import cm

def overlay_heatmaps(
        images : torch.Tensor,
        importance_maps : torch.Tensor,
        alpha : float = 0.5,
        colormap : str = 'plasma'
):
    heatmaps = create_heatmaps(importance_maps, colormap)
    assert images.ndim == 4, "images should have shape (batch_size, n_channels, height, width)"
    assert importance_maps.size(0) == images.size(0), "batch size must be the same for images and importance maps"
    assert (importance_maps.size(2) == images.size(2) and
            importance_maps.size(3) == images.size(3)), "width and height should be the same"

    return images * (1.0-alpha) + heatmaps * alpha

def create_heatmaps(
        importance_maps : torch.Tensor,
        colormap : str = 'plasma'
):
    assert importance_maps.ndim == 4, "importance_maps should have shape (batch_size, 1, height, width)"
    heatmaps = torch.empty(importance_maps.size(0), 3, importance_maps.size(2), importance_maps.size(3))
    cmap = cm.get_cmap(colormap)

    for i in range(importance_maps.size(0)):
        npa = importance_maps[i].squeeze(0).numpy(force=True)
        rgb = cmap(npa)[:,:, :3]
        heatmaps[i] = torch.from_numpy(rgb).permute(2, 0, 1)
    return heatmaps