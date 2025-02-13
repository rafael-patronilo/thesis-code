from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Optional




from core.init import DO_SCRIPT_IMPORTS
from core.init.options_parsing import option, positional


if TYPE_CHECKING or DO_SCRIPT_IMPORTS:
    from core.training import Trainer
    from core.storage_management import ModelFileManager
    from core.datasets import dataset_wrappers, get_dataset
    from analysis_tools.perception_network import evaluate_perception_network
    from analysis_tools.datasets import analyze_dataset
    from analysis_tools.xtrains_utils import CLASSES, SHORT_CLASSES, SHORT_CONCEPTS, log_short_class_correspondence
    import torch
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
    num_images : int = field(
        default=64, metadata=option(int, help_="Number of images to explain"))
    seed : int = field(
        default=110225,
        metadata=option(int, help_="Random seed for image selection"))
    mask_cell_size : int = field(default=16,
        metadata=option(int, help_="Size of the cells in the mask"))
    num_masks : int = field(default=8000,
        metadata=option(int, help_="Number of masks to generate"))
    mask_p : float = field(default=0.5,
        metadata=option(float, help_="Probability of a cell not being masked"))
    overlay_alpha : float = field(default=0.75,
        metadata=option(float, help_="Alpha value for the overlay"))
    colormap : str = field(default='plasma',
        metadata=option(str, help_="Colormap to use for the heatmap"))

def main(options: Options):
    log_short_class_correspondence(logger)
    model_name = options.model_name
    with torch.no_grad():
        with ModelFileManager(model_name) as file_manager:
            trainer = Trainer.load_checkpoint(file_manager, prefer='best')
            trainer.model.eval()
            hybrid_network = trainer.model
            perception_network = hybrid_network.perception_network
            dataset = get_dataset('xtrains_with_concepts')

            label_indices = dataset.get_column_references().get_label_indices(CLASSES)
            selected_dataset = dataset_wrappers.SelectCols(dataset, select_y=label_indices)

            dest_path = file_manager.results_dest.joinpath('rise')

            logger.info("Selecting samples")
            rng = torch.Generator(device=torch.get_default_device())
            rng.manual_seed(options.seed)
            dataloader = DataLoader(
                selected_dataset.for_validation(),
                batch_size=options.num_images,
                shuffle=True,
                generator=rng
            )
            samples = next(iter(dataloader))
            images = samples[0].to(torch.get_default_device())
            labels = samples[1].to(torch.get_default_device())
            original_images_path = dest_path.joinpath('original')
            original_images_path.mkdir(parents=True, exist_ok=True)
            for i in range(options.num_images):
                image = images[i]
                img_path = original_images_path.joinpath(f'{i}.png')
                logger.info(f"Saving image {i} to {img_path}")
                torchvision.utils.save_image(image, img_path)

            logger.info("Computing predictions")
            pred_concepts = perception_network(images)
            pred_classes = hybrid_network(images)
            results = pandas.DataFrame(
                index=range(options.num_images),
                columns=pandas.Index([f'true_{c}' for c in CLASSES] + [f'pred_{c}' for c in CLASSES] + ['other', 'valid']),
                data=torch.hstack((labels, pred_concepts, pred_classes)).numpy(force=True)
            )
            results.to_csv(dest_path.joinpath('predictions.csv'))
            logger.info(f"Saved predictions to {dest_path.joinpath('predictions.csv')}")

            logger.info("Computing RISE importance maps")

            rise = RISE(
                mask_cell_size=options.mask_cell_size,
                n_masks=options.num_masks,
                probability=options.mask_p,
            )
            with progress_cm.track('Rise generation', 'masks') as progress_tracker:
                masks = rise.generate(
                    perception_network, images, progress_tracker=progress_tracker)
            for output in range(masks.shape[1]):
                cls = CLASSES[output]
                cls_path = dest_path.joinpath(cls)
                cls_path.mkdir(parents=True, exist_ok=True)
                overlay = overlay_heatmaps(images, masks[:, [output]],
                                           alpha=options.overlay_alpha,
                                           colormap=options.colormap)
                for i in range(masks.shape[0]):
                    image_path = cls_path.joinpath(f'{i}.png')
                    logger.info(f"Saving importance map for image {i} for class {cls} to {image_path}")
                    torchvision.utils.save_image(overlay[i], image_path)
