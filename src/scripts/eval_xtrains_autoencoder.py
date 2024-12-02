from collections import OrderedDict
import sys
from typing import Callable, Iterable, Sequence
import math

from core.util.debug import debug_tensor
from core import Trainer, ModelFileManager
from core.datasets import SplitDataset, dataset_wrappers
from core.util.progress_trackers import log_cooldown

from core.metrics import metric_wrappers as metric_wrappers

from core.metrics.metrics_crosser import MetricCrosser
from core.nn import layers
import torch
import torchvision
import torch.utils.data as torch_data
import os
from core.metrics_logger import dataloader_worker_init_fn
from core.datasets import get_dataset
import logging
from logging_setup import NOTIFY
from torcheval import metrics
import json

logger = logging.getLogger(__name__)
SEED = 3892
NUM_IMAGES = 16

CLASSES = [
    'PassengerCar',
    'FreightWagon',
    'EmptyWagon',
    'LongWagon',
    'ReinforcedCar',
    'LongPassengerCar',
    'AtLeast2PassengerCars',
    'AtLeast2FreightWagons',
    'AtLeast3Wagons'
]


def evaluate_encoding_on_set(
        encoder : torch.nn.Module,
        file_manager : ModelFileManager,
        dataloader : torch_data.DataLoader,
        dataset_description : str,
        crosser : MetricCrosser
        ):

    crosser.reset()
    for x, y in dataloader:
        x = x.to(torch.get_default_device())
        y = y.to(torch.get_default_device())
        z = encoder(x)
        if not (z.min() >=0 and z.max() <= 1):
            logger.error(f"Encoding out of bounds: {z.min()} - {z.max()}")
        crosser.update(encoder(x), y)
    results = crosser.compute()
    results_path = file_manager.results_dest.joinpath(f"concept_cross_metrics").joinpath(dataset_description)
    results_path.mkdir(parents=True, exist_ok=True)
    for metric, result in results.items():
        logger.info(f"{dataset_description} {metric} results:\n{result}")
        dest_file = results_path.joinpath(f"{metric}.json")
        logger.info(f"Saving {metric} results to {dest_file}")
        result.to_csv(results_path.joinpath(f"{metric}.csv"))


def sample_images(
        trainer : Trainer, 
        file_manager : ModelFileManager, 
        dataset : torch_data.Dataset
    ):
    generator = torch.Generator(device=torch.get_default_device())
    generator.manual_seed(SEED)
    loader = torch_data.DataLoader(
        dataset, batch_size=NUM_IMAGES, shuffle=True, generator=generator)
    batch, _ = next(iter(loader))
    batch = batch.to(torch.get_default_device())
    results = trainer.model(batch)
    torchvision.utils.save_image(batch, file_manager.results_dest.joinpath("original.png"))
    torchvision.utils.save_image(results, file_manager.results_dest.joinpath("reconstructed.png"))

def evaluate_encoding(
        trainer : Trainer,
        file_manager : ModelFileManager,
        dataset : SplitDataset
    ):
    label_indices = dataset.get_collumn_references().get_label_indices(CLASSES)
    dataset = dataset_wrappers.SelectCols(dataset, select_y=label_indices)
    model : torch.nn.Module = trainer.model.encoder
    logger.info("Fitting min max normalizer to training set")
    normalizer = layers.MinMaxNormalizer.fit(model, trainer.make_loader(dataset.for_training()))
    logger.info(f"Normalizer min: {normalizer.min}")
    logger.info(f"Normalizer max: {normalizer.max}")
    model = layers.add_layers(model, normalizer)
    crosser = MetricCrosser(
        ('E', normalizer.min.size(0)),
        CLASSES,
        {
            'accuracy' : metrics.BinaryAccuracy,
            'f1' : metrics.BinaryF1Score,
            'precision' : metrics.BinaryPrecision,
            'recall' : metrics.BinaryRecall,
            'entropy' : lambda: (
                metric_wrappers.ToDtype(metrics.BinaryNormalizedEntropy(), torch.float32)
            )
        }
    )

    logger.info("Computing cross metrics on training set")
    evaluate_encoding_on_set(
        model, 
        file_manager, 
        trainer.make_loader(dataset.for_training()), 
        'train', 
        crosser
    )

    logger.info("Computing cross metrics on validation set")
    evaluate_encoding_on_set(
        model, 
        file_manager, 
        trainer.make_loader(dataset.for_validation()), 
        'val', 
        crosser
    )

def main():
    model_name = sys.argv[1]
    model_path = None
    if len(sys.argv) > 2:
        model_path = sys.argv[2]
    with torch.no_grad():
        with ModelFileManager(model_name, model_path) as file_manager:
            trainer = Trainer.load_checkpoint(file_manager)
            trainer.model.eval()
            dataset = get_dataset('xtrains_with_concepts')
            val_set = dataset.for_validation()
            sample_images(trainer, file_manager, val_set)
            logger.info("Sampling images done")
            evaluate_encoding(trainer, file_manager, dataset)
            logger.log(NOTIFY, 'Done')

