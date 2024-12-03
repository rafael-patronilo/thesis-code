from collections import OrderedDict
import sys
from typing import Callable, Iterable, Sequence
import math

from core.util.debug import debug_tensor
from core import Trainer, ModelFileManager
from core.datasets import SplitDataset, dataset_wrappers
from core.util.progress_trackers import LogProgressContextManager
from datetime import timedelta

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
from core.util.plotting import CrossBinaryHistogram
import json


logger = logging.getLogger(__name__)
progress_cm = LogProgressContextManager(logger, cooldown=timedelta(minutes=2))
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

SHORT_CLASSES = [
    'Passenger',
    'Freight',
    'Empty',
    'Long',
    'Reinforced',
    'LongPassenger',
    '2Passenger',
    '2Freight',
    '3Wagons'
]


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

def evaluate_encoding_on_set(
        encoder : torch.nn.Module,
        file_manager : ModelFileManager,
        dataloader : torch_data.DataLoader,
        dataset_description : str,
        crosser : MetricCrosser,
        encoding_labels : list[str]
        ):
    histogram = CrossBinaryHistogram(encoding_labels, SHORT_CLASSES)

    crosser.reset()
    with progress_cm.track(f'Encoding evaluation on {dataset_description} set', 'batches', dataloader) as progress_tracker:
        for x, y in dataloader:
            x = x.to(torch.get_default_device())
            y = y.to(torch.get_default_device())
            z = encoder(x)
            crosser.update(z, y)
            histogram.update(z, y)
            progress_tracker.tick()
    results = crosser.compute()
    results_path = file_manager.results_dest.joinpath(f"concept_cross_metrics").joinpath(dataset_description)
    results_path.mkdir(parents=True, exist_ok=True)
    for metric, result in results.items():
        logger.info(f"{dataset_description} {metric} results:\n{result}")
        dest_file = results_path.joinpath(f"{metric}.json")
        logger.info(f"Saving {metric} results to {dest_file}")
        result.to_csv(results_path.joinpath(f"{metric}.csv"))
    torch.save(histogram.histogram, results_path.joinpath("histogram.pt"))
    histogram.create_figure(mode='overlayed').savefig(results_path.joinpath("densities_overlayed.png"))
    histogram.create_figure(mode='stacked').savefig(results_path.joinpath("densities_stacked.png"))

def evaluate_encoding(
        trainer : Trainer,
        file_manager : ModelFileManager,
        dataset : SplitDataset
    ):
    model : torch.nn.Module = trainer.model.encoder
    normalizer = layers.MinMaxNormalizer.fit(model, trainer.make_loader(dataset.for_training()), progress_cm=progress_cm)
    logger.info(f"Normalizer min: {normalizer.min}")
    logger.info(f"Normalizer max: {normalizer.max}")
    model = layers.add_layers(model, normalizer)
    encoding_labels = [f"E{i}" for i in range(normalizer.min.size(0))]

    crosser = MetricCrosser(
        encoding_labels,
        SHORT_CLASSES,
        {
            'accuracy' : metrics.BinaryAccuracy,
            'f1' : metrics.BinaryF1Score,
            'precision' : lambda: metric_wrappers.ToDtype(
                metrics.BinaryPrecision(), torch.long, apply_to_pred=False),
            'recall' : lambda: metric_wrappers.ToDtype(
                metrics.BinaryRecall(), torch.long, apply_to_pred=False),
            'entropy' : lambda: metric_wrappers.ToDtype(
                metrics.BinaryNormalizedEntropy(), torch.float32),
            'auc' : metrics.BinaryAUROC,
            
        }
    )
    crosser.to(torch.get_default_device())

    logger.info("Computing cross metrics on training set")
    evaluate_encoding_on_set(
        model, 
        file_manager, 
        trainer.make_loader(dataset.for_training()), 
        'train', 
        crosser,
        encoding_labels
    )

    logger.info("Computing cross metrics on validation set")
    evaluate_encoding_on_set(
        model, 
        file_manager, 
        trainer.make_loader(dataset.for_validation()), 
        'val', 
        crosser,
        encoding_labels
    )

def analyze_dataset(trainer : Trainer, file_manager : ModelFileManager, dataset : torch_data.Dataset, dataset_description : str):
    loader = trainer.make_loader(dataset)
    hist = torch.zeros(len(CLASSES), 2)
    with progress_cm.track(f'Analyzing dataset {dataset_description}', 'batches', loader) as progress_tracker:
        for _, y in loader:
            for j in range(len(CLASSES)):
                col : torch.Tensor= y[:, j]
                pos = y[:, j].sum(0)
                hist[j][1] += pos
                hist[j][0] += col.size(0) - pos
            progress_tracker.tick()
    logger.info(f"Dataset {dataset_description} histogram:\n{hist}")
    densities = hist / hist.sum(-1, keepdim=True)
    logger.info(f"Dataset {dataset_description} densities:\n{densities}")
    import matplotlib.pyplot as plt
    import numpy as np
    fig, ax = plt.subplots(layout='constrained')
    hist = hist.numpy(force=True)
    densities = densities.numpy(force=True)
    p = ax.bar(SHORT_CLASSES, hist[:, 1], label='Positive', color='g')
    ax.bar_label(p, labels=[f'{d:.2%}' for d in densities[:,1]], label_type='center')
    p = ax.bar(SHORT_CLASSES, hist[:, 0], bottom=hist[:, 1], label='Negative', color='r')
    ax.bar_label(p, labels=[f'{d:.2%}' for d in densities[:,0]], label_type='center')
    ax.set_title(f"{dataset_description} concept histogram")
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
    ax.legend()
    fig.savefig(file_manager.results_dest.joinpath(f"{dataset_description}_concept_histogram.png"))
    

def analyze_selected_dataset(trainer : Trainer, file_manager : ModelFileManager, selected_dataset : SplitDataset):
    analyze_dataset(trainer, file_manager, selected_dataset.for_training(), 'train')
    analyze_dataset(trainer, file_manager, selected_dataset.for_validation(), 'val')

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
            
            label_indices = dataset.get_collumn_references().get_label_indices(CLASSES)
            selected_dataset = dataset_wrappers.SelectCols(dataset, select_y=label_indices)
            evaluate_encoding(trainer, file_manager, selected_dataset)
            logger.info("Encoding evaluation done")
            analyze_selected_dataset(trainer, file_manager, selected_dataset)
            
            logger.log(NOTIFY, 'Done')

