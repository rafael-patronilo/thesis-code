from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Optional
from core.init import DO_SCRIPT_IMPORTS
from core.init.options_parsing import option, positional, comma_split


if TYPE_CHECKING or DO_SCRIPT_IMPORTS:
    from core.training import Trainer
    from core.storage_management import ModelFileManager
    from core.datasets import dataset_wrappers, get_dataset
    from analysis_tools.xtrains_utils import CLASSES, log_short_class_correspondence
    import torch
    from torch import nn
    from torch.utils.data import DataLoader
    import torchvision
    from core.eval.importance_maps.rise import RISE
    from core.eval.importance_maps.heatmap import overlay_heatmaps
    import pandas
    import logging
    from datetime import timedelta
    from core.util.progress_trackers import LogProgressContextManager

    logger = logging.getLogger(__name__)
    progress_cm = LogProgressContextManager(logger, cooldown=timedelta(minutes=2))

IMAGES=[

]

@dataclass
class Options:
    model_name: str = field(
        metadata=positional(str, help_="Name of the model to explain"))
    image_batch_size : int = field(
        default=16, metadata=option(int, help_="Number of images to explain per batch"))
    image_batches : int = field(
        default=4, metadata=option(int, help_="Number of batches of images to explain")
    )
    seed : int = field(
        default=110225,
        metadata=option(int, help_="Random seed for image selection"))
    mask_cell_size : int = field(default=16,
        metadata=option(int, help_="Size of the cells in the mask"))
    num_masks : int = field(default=8000,
        metadata=option(int, help_="Number of masks to generate"))
    mask_p : float = field(default=0.5,
        metadata=option(float, help_="Probability of a cell not being masked"))
    overlay_alpha : float = field(default=0.5,
        metadata=option(float, help_="Alpha value for the overlay"))
    colormap : str = field(default='plasma',
        metadata=option(str, help_="Colormap to use for the heatmap"))
    target_activations : Optional[list[str]] = field(default=None,
        metadata=option(comma_split, help_="Target activations to explain"))

def save_images_and_preds(indices, images, labels,
                          perception_network, hybrid_network,
                          original_images_path, preds_path):
    for i, img_idx in enumerate(indices):
        image = images[i]
        img_path = original_images_path.joinpath(f'{img_idx}.png')
        torchvision.utils.save_image(image, img_path)

    pred_concepts = perception_network(images)
    pred_classes = hybrid_network(images)
    results = pandas.DataFrame(
        index=indices,
        data=torch.hstack((labels, pred_concepts, pred_classes)).numpy(force=True)
    )
    results.to_csv(preds_path,
                   mode='a', header=False, index=True)


def save_maps(options : Options, images, maps, dest_path):
    for output in range(maps.shape[1]):
        cls = CLASSES[output]
        cls_path = dest_path.joinpath(cls)
        cls_path.mkdir(parents=True, exist_ok=True)
        overlay = overlay_heatmaps(images, maps[:, [output]],
                                   alpha=options.overlay_alpha,
                                   colormap=options.colormap)
        for i in range(maps.shape[0]):
            image_path = cls_path.joinpath(f'{i}.png')
            logger.info(f"Saving importance map for image {i} for class {cls} to {image_path}")
            torchvision.utils.save_image(overlay[i], image_path)

def select_activations(trainer : 'Trainer', options: Options) -> tuple['nn.Module', list[str]]:

    raise NotImplementedError()


def main(options: Options):
    raise NotImplementedError("This script is not yet implemented")
    # log_short_class_correspondence(logger)
    # model_name = options.model_name
    # with torch.no_grad():
    #     with ModelFileManager(model_name) as file_manager:
    #         trainer = Trainer.load_checkpoint(file_manager, prefer='best')
    #         trainer.model.eval()
    #         hybrid_network = trainer.model
    #         perception_network = hybrid_network.perception_network
    #         dataset = get_dataset('xtrains_with_concepts')
    #
    #         label_indices = dataset.get_column_references().get_label_indices(CLASSES)
    #         selected_dataset = dataset_wrappers.SelectCols(dataset, select_y=label_indices)
    #
    #         dest_path = file_manager.results_dest.joinpath('rise')
    #         preds_file = dest_path.joinpath('predictions.csv')
    #         preds_file.write_text(
    #             ','.join(['image'] +
    #                 [f'true_{c}' for c in CLASSES] +
    #                 [f'pred_{c}' for c in CLASSES] +
    #                 ['other', 'valid']
    #             ) + '\n')
    #
    #         image_batch_size = options.image_batch_size
    #         batch = 0
    #
    #         rng = torch.Generator(device=torch.get_default_device())
    #         rng.manual_seed(options.seed)
    #         dataloader = DataLoader(
    #             selected_dataset.for_validation(),
    #             batch_size=image_batch_size,
    #             shuffle=True,
    #             generator=rng
    #         )
    #         original_images_path = dest_path.joinpath('original')
    #         original_images_path.mkdir(parents=True, exist_ok=True)
    #         rise = RISE(
    #             mask_cell_size=options.mask_cell_size,
    #             n_masks=options.num_masks,
    #             probability=options.mask_p,
    #         )
    #         batch = 0
    #         with progress_cm.track('Generating importance maps',
    #                                'masks', options.image_batches * options.num_masks) as progress_tracker:
    #             for images, labels in dataloader:
    #                 if batch >= options.image_batches:
    #                     break
    #                 first_image = batch * image_batch_size
    #                 indices = range(first_image, first_image + image_batch_size)
    #                 save_images_and_preds(indices, images, labels,
    #                                       perception_network, hybrid_network,
    #                                       original_images_path, preds_file)
    #                 maps = rise.generate(
    #                     perception_network, images, progress_tracker=progress_tracker)
    #                 save_maps(options, images, maps, dest_path)