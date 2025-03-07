from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from core.init import DO_SCRIPT_IMPORTS
from core.init.options_parsing import option, positional

if TYPE_CHECKING or DO_SCRIPT_IMPORTS:
    from core.training import Trainer
    from core.storage_management import ModelFileManager
    from core.datasets import dataset_wrappers
    from analysis_tools.perception_network import evaluate_concept_correspondence
    from analysis_tools.xtrains_utils import CLASSES, SHORT_CLASSES, log_short_class_correspondence
    import torch
    from core.datasets import get_dataset
    import logging

    logger = logging.getLogger(__name__)


@dataclass
class Options:
    model_name: str = field(
        metadata=positional(str, help_="Name of the model to evaluate"))
    with_training_set : bool = field(default=False,
        metadata=option(bool, help_="Whether to evaluate on the training set as well."))
    expect_concepts : bool = field(default=True,
        metadata=option(bool, help_="Whether to expect concepts in the dataset."))
    normalize_first : bool = field(default=False,
        metadata=option(bool, help_="Whether to normalize the data before evaluating."))



def main(options: Options):
    log_short_class_correspondence(logger)
    model_name = options.model_name
    with torch.no_grad():
        with ModelFileManager(model_name) as file_manager:
            trainer = Trainer.load_checkpoint(file_manager, prefer='best')
            trainer.model.eval()
            model = trainer.model.perception_network
            dataset = get_dataset('xtrains_with_concepts')

            label_indices = dataset.get_column_references().get_label_indices(CLASSES)
            selected_dataset = dataset_wrappers.SelectCols(dataset, select_y=label_indices)
            if options.expect_concepts:
                expected_concepts = dict(enumerate(SHORT_CLASSES))
            else:
                expected_concepts = None
            evaluate_concept_correspondence(
                trainer, model, file_manager, selected_dataset,
                options.normalize_first, SHORT_CLASSES,
                expected_concepts=expected_concepts,
                with_training=options.with_training_set
            )

            logger.info("Perception network evaluation done")