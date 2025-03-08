from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from core.init import DO_SCRIPT_IMPORTS
from core.init.options_parsing import comma_split, parse_bool, option, positional

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
    attribution: list[str] = field(
        metadata=option(comma_split, help_="Concept attribution to use "
                                           "between the perception network and the reasoning network"))
    with_training_set : bool = field(default=False,
        metadata=option(parse_bool, help_="Whether to evaluate on the training set as well."))



def main(options: Options):
    log_short_class_correspondence(logger)
    model_name = options.model_name
    with torch.no_grad():
        with ModelFileManager(model_name) as file_manager:
            trainer = Trainer.load_checkpoint(file_manager, prefer='best')
            trainer.model.eval()
            pn = trainer.model.perception_network
            rn = trainer.model.reasoning_network
