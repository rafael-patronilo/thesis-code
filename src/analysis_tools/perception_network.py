from datetime import timedelta
from typing import Optional

import pandas as pd
import torch
from torch.utils import data as torch_data
from torcheval import metrics
from torcheval.metrics.classification import BinaryRecall

from core.datasets import SplitDataset
from core.eval.metrics import PearsonCorrelationCoefficient, \
    metric_wrappers as metric_wrappers, BinaryBalancedAccuracy, \
    BinarySpecificity

from core.eval.metrics_crosser import MetricCrosser
from core.eval.plotting import CrossBinaryHistogram
from core.nn import layers
from core.storage_management import ModelFileManager
from core.training import Trainer
from core.util.progress_trackers import LogProgressContextManager
import logging
logger = logging.getLogger(__name__)

progress_cm = LogProgressContextManager(logger, cooldown=timedelta(minutes=2))

def evaluate_perception_network_on_set(
        perception_network : 'torch.nn.Module',
        file_manager : 'ModelFileManager',
        dataloader : 'torch_data.DataLoader',
        dataset_description : str,
        crosser : 'MetricCrosser',
        encoding_crosser : 'MetricCrosser',
        encoding_labels : list[str],
        class_labels : list[str]
        ):
    histogram = CrossBinaryHistogram(encoding_labels, class_labels)

    crosser.reset()
    encoding_crosser.reset()
    with progress_cm.track(f'Encoding evaluation on {dataset_description} set', 'batches', dataloader) as progress_tracker:
        for x, y in dataloader:
            x = x.to(torch.get_default_device())
            y = y.to(torch.get_default_device())
            z = perception_network(x)
            crosser.update(z, y)
            encoding_crosser.update(z, z)
            histogram.update(z, y)
            progress_tracker.tick()
    results = crosser.compute()
    encoding_results = encoding_crosser.compute()
    results_path = file_manager.results_dest.joinpath(f"concept_cross_metrics").joinpath(dataset_description)
    results_path.mkdir(parents=True, exist_ok=True)
    for metric, result in results.items():
        logger.info(f"{dataset_description} {metric} results:\n{result}")
        dest_file = results_path.joinpath(f"{metric}.csv")
        logger.info(f"Saving {metric} results to {dest_file}")
        result.to_csv(dest_file)
        summarized = pd.DataFrame(
            columns=result.columns,
            index=['max', 'max_encoding', 'min', 'min_encoding'] # type: ignore # Although not annotated, index can be list[str]
        )
        summarized.loc['max'] = result.max()
        summarized.loc['max_encoding'] = result.idxmax()
        summarized.loc['min'] = result.min()
        summarized.loc['min_encoding'] = result.idxmin()
        summarized.to_csv(results_path.joinpath(f"{metric}_summarized.csv"))
    encoding_results_path = results_path.joinpath("encoding")
    encoding_results_path.mkdir(parents=True, exist_ok=True)
    for metric, result in encoding_results.items():
        dest_file = encoding_results_path.joinpath(f"{metric}.csv")
        logger.info(f"Saving {metric} encoding results to {dest_file}")
        logger.info(f"Abs max of {metric}:\n {result.abs().max()}")
        result.to_csv(dest_file)

    torch.save(histogram.histogram, results_path.joinpath("histogram.pt"))
    histogram.create_figure(mode='overlayed').savefig(results_path.joinpath("densities_overlayed.png"))
    histogram.create_figure(mode='stacked').savefig(results_path.joinpath("densities_stacked.png"))


def evaluate_perception_network(
        trainer : 'Trainer',
        perception_network : 'torch.nn.Module',
        file_manager : 'ModelFileManager',
        dataset : 'SplitDataset',
        with_training : bool,
        min_max_normalize : bool,
        class_labels : list[str],
        encoding_labels : Optional[list[str]] = None
    ):
    if min_max_normalize:
        normalizer = layers.MinMaxNormalizer.fit(perception_network, trainer.make_loader(dataset.for_training()), progress_cm=progress_cm)
        logger.info(f"Normalizer min: {normalizer.min}")
        logger.info(f"Normalizer max: {normalizer.max}")
        model = layers.add_layers(perception_network, normalizer)
    else:
        model = perception_network
    if encoding_labels is None:
        encoding_labels = [f"E{i}" for i in range(dataset.get_shape()[1][0])]

    crosser = MetricCrosser(
        encoding_labels,
        class_labels,
        {
            'correlation': PearsonCorrelationCoefficient,
            'balanced_accuracy': lambda: metric_wrappers.ToDtype(
                BinaryBalancedAccuracy(), torch.int32, apply_to_pred=False),
            'recall' : lambda : metric_wrappers.ToDtype(
                BinaryRecall(), torch.int32, apply_to_pred=False),
            'specificity' : lambda : metric_wrappers.ToDtype(
                BinarySpecificity(), torch.int32, apply_to_pred=False),
            'auc': metrics.BinaryAUROC,
        }
    )
    encoding_crosser = MetricCrosser(
        encoding_labels,
        encoding_labels,
        {
            'correlation': PearsonCorrelationCoefficient,
        }
    )

    crosser.to(torch.get_default_device())
    encoding_crosser.to(torch.get_default_device())

    if with_training:
        logger.info("Computing cross metrics on training set")
        evaluate_perception_network_on_set(
            model,
            file_manager,
            trainer.make_loader(dataset.for_training()),
            'train',
            crosser,
            encoding_crosser,
            encoding_labels,
            class_labels
        )

    logger.info("Computing cross metrics on validation set")
    evaluate_perception_network_on_set(
        model,
        file_manager,
        trainer.make_loader(dataset.for_validation()),
        'val',
        crosser,
        encoding_crosser,
        encoding_labels,
        class_labels
    )