from pathlib import Path
from typing import TypedDict, Required
import torcheval
from core import datasets
from core.datasets import dataset_wrappers
from core.eval.objectives import Maximize
from core.eval import metrics
from core.eval import metrics
from core.eval.metrics import metric_wrappers
from core.training import Trainer, MetricsRecorder, TrainingRecorder
from core.storage_management import ModelFileManager
from core.nn.hybrid_network import HybridNetwork
from core.nn import PartiallyPretrained
import torch
from torcheval.metrics import Mean
from torch import nn

from core.nn.autoencoder import AutoEncoder
import logging

from core.training.checkpoint_triggers import BestMetric
from core.training.stop_criteria import EarlyStop

logger = logging.getLogger(__name__)

class ModelLoadPath(TypedDict, total=False):
    model_name : Required[str]
    model_path : str
    checkpoint_preference : str
    checkpoint_path : str



def load_network(load_path : ModelLoadPath) -> nn.Module:
    model_name = load_path['model_name']
    model_path = load_path.get('model_path', None)
    checkpoint_preference = load_path.get('checkpoint_preference', 'best')
    checkpoint_path = load_path.get('checkpoint_path', None)
    checkpoint_path = Path(checkpoint_path) if checkpoint_path is not None else None
    with ModelFileManager(model_name, model_path) as file_manager:
        trainer = Trainer.load_checkpoint(file_manager, checkpoint_path,
                                          prefer=checkpoint_preference) # type: ignore
        return trainer.model

def modify_perception_network(
        perception_network : nn.Module,
        num_concepts : int
):
    if isinstance(perception_network, AutoEncoder):
        logger.info("Selecting encoder")
        encoder = perception_network.encoder
        return modify_perception_network(encoder, num_concepts)
    elif isinstance(perception_network, nn.Sequential):
        layers = [(layer, True) for layer in perception_network.layers]
        layers.pop()
        if layers[-1][0].out_features != num_concepts:
            logger.info("Replacing last linear with correct size")
            layers.pop()
            layers.append((nn.Linear(perception_network.layers[-2].in_features, num_concepts), False))
        logger.info("Setting last activation to sigmoid")
        layers.append((nn.Sigmoid(), False))
        return PartiallyPretrained(*layers)
    else:
        logger.warning("No modification applied.")
        return perception_network




def create_model(
        num_concepts: int,
        perception_network_path : ModelLoadPath,
        reasoning_network_path : ModelLoadPath,
        load_perception_weights : bool = True,
    ) -> HybridNetwork:
    perception_network = load_network(perception_network_path)
    reasoning_network = load_network(reasoning_network_path)

    perception_network = modify_perception_network(perception_network, num_concepts)
    return HybridNetwork(
        reasoning_network=reasoning_network,
        perception_network=perception_network
    )



def create_trainer(
        dataset_name : str,
        pre_trained_learning_rate : float = 0.00001,
        untrained_learning_rate : float = 0.001,
        **kwargs) -> Trainer:
    dataset = dataset_wrappers.ConcatConst(datasets.get_dataset(dataset_name), 1, 'y')
    dataset.for_training() # make sure it is loaded
    col_refs = dataset.get_column_references()
    num_labels = len(col_refs.labels)
    valid_col = num_labels
    weights = torch.ones(num_labels + 1)
    weights[valid_col] = 4
    loss_fn = nn.BCELoss(weights)
    patience = kwargs.pop('patience', 10)
    metric_functions : dict = {
        'elapsed' : metrics.Elapsed,
        'mean_valid' : lambda: metric_wrappers.SelectCol(metric_wrappers.Unary(Mean()), valid_col)
    }
    classes : list[str] = dataset.get_column_references().labels.columns_to_names + ['valid']
    metric_functions.update(**metric_wrappers.SelectCol.col_wise(classes, {
        'balanced_accuracy' : metrics.BinaryBalancedAccuracy,
    }, reduction='min'))
    val_metrics = MetricsRecorder(
        identifier='val',
        metric_functions=metric_functions,
        dataset=dataset.for_validation
    )
    train_metrics = TrainingRecorder(
        metric_functions=metric_functions
    )
    objective = Maximize('val', 'balanced_accuracy')

    model = create_model(**kwargs)

    def optimizer(_) -> torch.optim.Optimizer:
        pn = model.perception_network
        if isinstance(pn, PartiallyPretrained):
            return torch.optim.Adam([
                {'params': pn.pre_trained_parameters(), 'lr': pre_trained_learning_rate},
                {'params': pn.untrained_parameters(), 'lr': untrained_learning_rate}])
        else:
            return torch.optim.Adam(pn.parameters(), lr = pre_trained_learning_rate)

    return Trainer(
        model=model,
        loss_fn=loss_fn,
        optimizer=optimizer,
        training_set=dataset,
        metric_loggers=[train_metrics, val_metrics],
        stop_criteria=[EarlyStop(objective, patience=patience)],
        checkpoint_triggers=[BestMetric(objective)],
        objective=objective,
        batch_size=64
    )