from dataclasses import dataclass, field
from typing import TYPE_CHECKING
from core.init import DO_SCRIPT_IMPORTS
from core.init.options_parsing import positional


if TYPE_CHECKING or DO_SCRIPT_IMPORTS:
    from core.training import Trainer
    from core.storage_management import ModelFileManager
    from core.datasets import dataset_wrappers
    from analysis_tools.autoencoders import sample_images
    from analysis_tools.perception_network import evaluate_perception_network
    from analysis_tools.datasets import analyze_dataset
    from analysis_tools.xtrains_utils import CLASSES, SHORT_CLASSES, log_short_class_correspondence
    import torch
    from core.datasets import get_dataset
    import logging

    logger = logging.getLogger(__name__)

SEED = 3892
NUM_IMAGES = 16

@dataclass
class Options:
    model_name : str = field(
        metadata=positional(str, help_="Name of the model to evaluate"))


def main(options : Options):
    log_short_class_correspondence(logger)
    model_name = options.model_name
    with torch.no_grad():
        with ModelFileManager(model_name) as file_manager:
            trainer = Trainer.load_checkpoint(file_manager, prefer='best')
            trainer.model.eval()
            encoder = trainer.model.encoder
            dataset = get_dataset('xtrains_with_concepts')
            val_set = dataset.for_validation()

            sample_images(trainer.model, SEED, NUM_IMAGES, file_manager, val_set)
            logger.info("Sampling images done")

            label_indices = dataset.get_column_references().get_label_indices(CLASSES)
            selected_dataset = dataset_wrappers.SelectCols(dataset, select_y=label_indices)
            evaluate_perception_network(trainer, encoder, file_manager, selected_dataset,
                                        SHORT_CLASSES)
            logger.info("Encoding evaluation done")

            analyze_dataset(trainer, file_manager, selected_dataset.for_training(),
                            'train', SHORT_CLASSES)
            analyze_dataset(trainer, file_manager, selected_dataset.for_validation(),
                            'val', SHORT_CLASSES)