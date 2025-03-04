from dataclasses import dataclass, field
from typing import TYPE_CHECKING
from core.init import DO_SCRIPT_IMPORTS
from core.init.options_parsing import option, positional


if TYPE_CHECKING or DO_SCRIPT_IMPORTS:
    from core.training import Trainer
    from core.storage_management import ModelFileManager
    from core.datasets import dataset_wrappers
    from analysis_tools.perception_network import evaluate_perception_network, select_all_linear_activations
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
    normalize_first : bool = field(default=False,
        metadata=option(bool, help_="Whether to normalize the data before evaluating."))
    include_reasoning : bool = field(default=True,
        metadata=option(bool, help_="Whether to include reasoning network when selecting activations."))



def main(options: Options):
    log_short_class_correspondence(logger)
    model_name = options.model_name
    with (torch.no_grad()):
        with ModelFileManager(model_name) as file_manager:
            trainer = Trainer.load_checkpoint(file_manager, prefer='best')
            trainer.model.eval()
            target_network = trainer.model
            if not options.include_reasoning:
                target_network = target_network.perception_network


            dataset = get_dataset('xtrains_with_concepts')
            label_indices = dataset.get_column_references().get_label_indices(CLASSES)

            x, _ = next(iter(trainer.make_loader(dataset.for_training())))

            activation_extractor, neuron_labels = select_all_linear_activations(target_network, x)
            logger.info(f"{len(neuron_labels)} activations selected")

            selected_dataset = dataset_wrappers.SelectCols(dataset, select_y=label_indices)

            evaluate_perception_network(trainer, activation_extractor, file_manager, selected_dataset,
                                        options.with_training_set,
                                        options.normalize_first,
                                        SHORT_CLASSES,
                                        neuron_labels=neuron_labels,
                                        cross_neurons=False,
                                        results_path=file_manager.results_dest.joinpath('activations_to_concepts'))
            logger.info("Activations evaluation done")