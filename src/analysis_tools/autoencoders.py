import torch
import torchvision
from torch.utils import data as torch_data
from torch.nn import Module

from core.storage_management import ModelFileManager


def sample_images(
        model : Module,
        seed : int,
        num_images : int,
        file_manager : 'ModelFileManager',
        dataset : 'torch_data.Dataset'
    ):
    generator = torch.Generator(device=torch.get_default_device())
    generator.manual_seed(seed)
    loader = torch_data.DataLoader(
        dataset, batch_size=num_images, shuffle=True, generator=generator)
    batch, _ = next(iter(loader))
    batch = batch.to(torch.get_default_device())
    results = model(batch)
    torchvision.utils.save_image(batch, file_manager.results_dest.joinpath("original.png"))
    torchvision.utils.save_image(results, file_manager.results_dest.joinpath("reconstructed.png"))
