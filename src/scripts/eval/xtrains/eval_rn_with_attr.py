from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from core.init import DO_SCRIPT_IMPORTS
from core.init.options_parsing import comma_split, parse_bool, option, positional


if TYPE_CHECKING or DO_SCRIPT_IMPORTS:
    import torch
    from torch import nn
    from core.training import Trainer
    from core.storage_management import ModelFileManager
    from analysis_tools.xtrains_utils import log_short_class_correspondence, make_order_from_attribution
    import torch
    from core.util import progress_trackers
    from core.nn.hybrid_network import HybridNetwork
    from core.nn.layers import Reorder

    import logging

    logger = logging.getLogger(__name__)
    progress_cm = progress_trackers.LogProgressContextManager(logger)


@dataclass
class Options:
    model_name: str = field(
        metadata=positional(str, help_="Name of the model to evaluate"))
    attribution: list[str] = field(
        metadata=positional(comma_split, help_="Concept attribution to use "
                                           "between the perception network and the reasoning network"))
    with_training_set : bool = field(default=True,
        metadata=option(parse_bool, help_="Whether to evaluate on the training set as well."))

def main(options: Options):
    log_short_class_correspondence(logger)
    model_name = options.model_name
    with torch.no_grad():
        with ModelFileManager(model_name) as file_manager:
            trainer = Trainer.load_checkpoint(file_manager, prefer='best')
            trainer.model.eval()
            order = make_order_from_attribution(options.attribution)

            pn = trainer.model.perception_network
            rn = trainer.model.reasoning_network
            pn = nn.Sequential(pn, Reorder(order))
            trainer.model = HybridNetwork(pn, rn)
            if options.with_training_set:
                training_logger = trainer.train_logger
                if training_logger is None:
                    raise ValueError("No train logger found")
                loader = trainer.training_loader()
                training_logger.prepare_torch_metrics()
                with progress_cm.track('Training set evaluation', 'batches', len(loader)) as progress_tracker:
                    for x, y in loader:
                        x = x.to(torch.get_default_device())
                        y = y.to(torch.get_default_device())
                        z = trainer.model(x)
                        training_logger.update_torch_metrics(z, y)
                        progress_tracker.tick()
            record = trainer.eval_metrics(write=False)
            str_builder = ["Evaluation complete\n"]
            for metric_logger_id, metrics in record.items():
                str_builder.append(f"Metrics for {metric_logger_id}:\n")
                for metric, value in metrics.items():
                    str_builder.append(f"\t{metric}: {value}\n")
            logger.info(''.join(str_builder))