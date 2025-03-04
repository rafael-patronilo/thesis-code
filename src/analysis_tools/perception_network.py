from datetime import timedelta
from pathlib import Path
from typing import Optional, Literal, Type, Any
from collections import OrderedDict
import pandas as pd
import torch
from torch import nn
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

from core.nn.activation_extractor import ActivationExtractor
logger = logging.getLogger(__name__)

progress_cm = LogProgressContextManager(logger, cooldown=timedelta(minutes=2))

def select_all_linear_activations(
        model : 'nn.Module',
        x : torch.Tensor,
        target_module_type : type[nn.Module] = nn.Linear
    ) -> tuple[ActivationExtractor, list[str]]:
    """
    Selects all layers after a dense linear layer
    :param model:
    :param target_module_type: The target module type that
        we want to select the activation (nn.Linear by default)
    :param x:

    :return:
    """
    layer_names : list[str] = []
    target_layers : list[nn.Module] = []
    was_target : bool = False
    for name, layer in model.named_modules():
        if was_target:
            layer_names.append(name)
            target_layers.append(layer)
        was_target = isinstance(layer, target_module_type)
    logger.debug(f"Selected layers: {layer_names}")
    activation_extractor = ActivationExtractor(model, target_layers)
    activation_extractor(x.to(torch.get_default_device()))
    neuron_labels : list[str] = []
    for name, layer in zip(layer_names, target_layers):
        for i in range(activation_extractor.outputs[layer].shape[1]):
            neuron_labels.append(f"{name}[{i}]")
    return activation_extractor, neuron_labels



def evaluate_perception_network_on_set(
        perception_network : 'torch.nn.Module',
        dataloader : 'torch_data.DataLoader',
        dataset_description : str,
        crosser : 'MetricCrosser',
        neuron_crosser : 'MetricCrosser',
        neuron_labels : list[str],
        class_labels : list[str],
        results_path : Path
        ):
    histogram = CrossBinaryHistogram(neuron_labels, class_labels)
    pred_histogram = torch.zeros(len(neuron_labels), 2)
    total_preds = 0

    crosser.reset()
    neuron_crosser.reset()
    with progress_cm.track(f'Encoding evaluation on {dataset_description} set', 'batches', dataloader) as progress_tracker:
        for x, y in dataloader:
            x = x.to(torch.get_default_device())
            y = y.to(torch.get_default_device())
            z = perception_network(x)
            crosser.update(z, y)
            neuron_crosser.update(z, z)
            histogram.update(z, y)
            pred_histogram += torch.where(
                (z > 0.5).unsqueeze(-1),
                torch.tensor([0, 1], device=z.device),
                torch.tensor([1, 0], device=z.device)
            ).sum(dim=0)
            total_preds += z.shape[0]
            progress_tracker.tick()
    results = crosser.compute()
    neuron_results = neuron_crosser.compute()
    results_path = results_path.joinpath(dataset_description)
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
    for metric, result in neuron_results.items():
        dest_file = encoding_results_path.joinpath(f"{metric}.csv")
        logger.info(f"Saving {metric} encoding results to {dest_file}")
        logger.info(f"Abs max of {metric}:\n {result.abs().max()}")
        result.to_csv(dest_file)

    pd_pred_hist = pd.DataFrame(
        pred_histogram.numpy(force=True),
        index=pd.Index(neuron_labels),
        columns=pd.Index(['negative', 'positive'])
    )
    pd_pred_hist.to_csv(results_path.joinpath("pred_histogram.csv"))
    (pd_pred_hist / total_preds).to_csv(results_path.joinpath("pred_densities.csv"))

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
        neuron_labels : Optional[list[str]] = None,
        cross_neurons : bool = True,
        results_path : Optional[Path] = None,
        with_validation : bool = True,
        with_test : bool = False
    ):
    if min_max_normalize:
        normalizer = layers.MinMaxNormalizer.fit(
            perception_network, trainer.make_loader(dataset.for_training()), progress_cm=progress_cm)
        logger.info(f"Normalizer min: {normalizer.min}")
        logger.info(f"Normalizer max: {normalizer.max}")
        model = layers.add_layers(perception_network, normalizer)
    else:
        model = perception_network
    if neuron_labels is None:
        neuron_labels = [f"N[{i}]" for i in range(dataset.get_shape()[1][0])]
    if results_path is None:
        results_path = file_manager.results_dest.joinpath('concept_cross_metrics')

    crosser = MetricCrosser(
        neuron_labels,
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
    neuron_crosser = MetricCrosser(
        neuron_labels,
        neuron_labels,
        {
            'correlation': PearsonCorrelationCoefficient,
        } if cross_neurons else {}
    )

    crosser.to(torch.get_default_device())
    neuron_crosser.to(torch.get_default_device())

    if with_training:
        logger.info("Computing cross metrics on training set")
        evaluate_perception_network_on_set(
            model,
            trainer.make_loader(dataset.for_training()),
            'train',
            crosser,
            neuron_crosser,
            neuron_labels,
            class_labels,
            results_path
        )
    if with_validation:
        logger.info("Computing cross metrics on validation set")
        evaluate_perception_network_on_set(
            model,
            trainer.make_loader(dataset.for_validation()),
            'val',
            crosser,
            neuron_crosser,
            neuron_labels,
            class_labels,
            results_path
        )
    if with_test:
        logger.info("Computing cross metrics on test set")
        test_set = dataset.for_testing()
        if test_set is None:
            logger.warning("No test set")
        else:
            evaluate_perception_network_on_set(
                model,
                trainer.make_loader(test_set),
                'test',
                crosser,
                neuron_crosser,
                neuron_labels,
                class_labels,
                results_path
            )