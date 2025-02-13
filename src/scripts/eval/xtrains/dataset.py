from dataclasses import dataclass, field
from math import log
from pathlib import Path
from typing import TYPE_CHECKING

from torcheval.metrics import BinaryRecall

from core.init import DO_SCRIPT_IMPORTS

from core.init.options_parsing import option

if TYPE_CHECKING or DO_SCRIPT_IMPORTS:
    from core.init import options as global_options
    from core.datasets import dataset_wrappers
    from analysis_tools.datasets import analyze_dataset
    from analysis_tools.xtrains_utils import CLASSES, SHORT_CLASSES, SHORT_CONCEPTS, log_short_class_correspondence
    import torch
    from core.datasets import get_dataset
    from core.eval.metrics import BinaryBalancedAccuracy, metric_wrappers, PearsonCorrelationCoefficient, \
        BinarySpecificity

    from core.eval.metrics_crosser import MetricCrosser
    from torch.utils.data import DataLoader
    import logging
    from torch.utils.data import DataLoader


    from datetime import timedelta
    from core.util.progress_trackers import LogProgressContextManager


    logger = logging.getLogger(__name__)
    progress_cm = LogProgressContextManager(logger, cooldown=timedelta(minutes=2))


@dataclass
class Options:
    dataset_name: str = field(default='xtrains_with_concepts',
                              metadata=option(str, help_="Name of the dataset to evaluate"))
    batch_size: int = field(default=64,
                            metadata=option(int, help_="Batch size for data loaders"))
    destination : Path = field(default=Path('storage/analysis_results'),
                        metadata=option(Path, help_="Destination for analysis results"))



def cross_classes(
        crosser : 'MetricCrosser',
        loader: 'DataLoader',
        destination : 'Path'
):
    destination.mkdir(parents=True, exist_ok=True)
    crosser.reset()
    with progress_cm.track('Cross class evaluation', 'batches', loader) as progress_tracker:
        for _, y in loader:
            crosser.update(y, y)
            progress_tracker.tick(y.shape[0])
    logger.info("Computing cross class metrics...")
    for k, v in crosser.compute().items():
        v.to_csv(destination.joinpath(f'{k}_correlation.csv'))

def main(options: Options):
    def make_loader(dataset):
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=options.batch_size,
            shuffle=False
        )
    log_short_class_correspondence(logger)
    destination = options.destination.joinpath(options.dataset_name)
    destination.mkdir(parents=True, exist_ok=True)
    dataset = get_dataset(options.dataset_name)
    if hasattr(dataset, 'skip_image_loading'):
        dataset.skip_image_loading = True # type: ignore

    label_indices = dataset.get_column_references().get_label_indices(CLASSES)
    selected_dataset = dataset_wrappers.SelectCols(dataset, select_y=label_indices)

    crosser = MetricCrosser(
        SHORT_CLASSES,
        SHORT_CLASSES,
        {
            'correlation': PearsonCorrelationCoefficient,
            'balanced_accuracy': lambda: metric_wrappers.ToDtype(
                BinaryBalancedAccuracy(), torch.int32, apply_to_pred=False),
            'recall': lambda: metric_wrappers.ToDtype(
                BinaryRecall(), torch.int32, apply_to_pred=False),
            'specificity': lambda: metric_wrappers.ToDtype(
                BinarySpecificity(), torch.int32, apply_to_pred=False),
        }
    )

    logger.info('On training set')
    cross_classes(
        crosser, make_loader(selected_dataset.for_training()), destination.joinpath('train'))
    logger.info('On validation set')
    cross_classes(
        crosser,  make_loader(selected_dataset.for_validation()), destination.joinpath('val'))

    analyze_dataset(make_loader(selected_dataset.for_training()),
                    destination,
                    'train', SHORT_CLASSES)
    analyze_dataset(make_loader(selected_dataset.for_validation()),
                    destination,
                    'val', SHORT_CLASSES)