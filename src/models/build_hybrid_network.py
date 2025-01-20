import json
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
from torcheval.metrics import Mean, metric
from torch import nn

from core.nn.autoencoder import AutoEncoder
import logging

from core.training.checkpoint_triggers import BestMetric
from core.training.stop_criteria import EarlyStop
from core.training.metrics_recorder import EvaluationResult

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
        layers = [(layer, True) for layer in perception_network]
        layers.pop()
        if layers[-1][0].out_features != num_concepts:
            logger.info("Replacing last linear with correct size")
            removed, _ = layers.pop()
            layers.append((nn.Linear(removed.in_features, num_concepts), False))
        logger.info("Setting last activation to sigmoid")
        layers.append((nn.Sigmoid(), False))
        return PartiallyPretrained(*layers)
    else:
        logger.warning("No modification applied.")
        return perception_network




def create_model(
        num_concepts: int,
        perception_network_path : ModelLoadPath,
        reasoning_network_path : ModelLoadPath
    ) -> HybridNetwork:
    perception_network = load_network(perception_network_path)
    reasoning_network = load_network(reasoning_network_path)

    perception_network = modify_perception_network(perception_network, num_concepts)
    return HybridNetwork(
        perception_network=perception_network,
        reasoning_network=reasoning_network
    )

def pn_evaluator(model : 'HybridNetwork', x, y):
    return EvaluationResult(model.perception_network(x), y)

def create_trainer(
        dataset_name : str,
        concept_dataset_name : str,
        concepts : list[str],
        pre_trained_learning_rate : float = 0.00001,
        untrained_learning_rate : float = 0.001,
        valid_col_weight : float = 1,
        **kwargs) -> Trainer:
    dataset = dataset_wrappers.ConcatConst(datasets.get_dataset(dataset_name), 1, 'y')
    dataset.for_training() # make sure it is loaded
    concept_dataset = datasets.get_dataset(concept_dataset_name)
    concept_dataset.for_validation() # make sure it is loaded
    col_refs = dataset.get_column_references()
    logger.info(f"Column info for dataset: {col_refs}")
    concept_col_refs = concept_dataset.get_column_references()
    logger.info(f"Column info for concept dataset: {concept_col_refs}")
    num_labels = len(col_refs.labels.columns_to_names)
    valid_col = num_labels - 1
    weights = torch.ones(num_labels)
    weights[valid_col] = valid_col_weight
    logger.info(f"Loss weights: {weights}")
    loss_fn = nn.BCELoss(weights)
    patience = kwargs.pop('patience', 10)

    classes : list[str] = col_refs.labels.columns_to_names
    def metric_functions():
        metric_functions_ : dict = {
            'elapsed': metrics.Elapsed(),
            'mean_valid': metric_wrappers.SelectCol(metric_wrappers.Unary(Mean()), valid_col)
        }
        metric_functions_.update(**metric_wrappers.SelectCol.col_wise(classes, {
            'balanced_accuracy' : metric_wrappers.to_int(metrics.BinaryBalancedAccuracy),
        }, reduction='min'))
        return metric_functions_
    val_metrics = MetricsRecorder(
        identifier='val',
        metric_functions=metric_functions(),
        dataset=dataset.for_validation
    )
    train_metrics = TrainingRecorder(
        metric_functions=metric_functions()
    )
    pn_metrics = MetricsRecorder(
        identifier='pn_val',
        metric_functions={
            f"balanced_accuracy_{concept}" : metric_wrappers.SelectCol(
                metric_wrappers.to_int(metrics.BinaryBalancedAccuracy)(),
                concept_col_refs.labels.names_to_column[concept]
            ) for concept in concepts
        },
        dataset=concept_dataset.for_validation,
        evaluator=pn_evaluator
    )
    objective = Maximize('val', 'balanced_accuracy')

    model = create_model(len(concepts), **kwargs)

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
        metric_loggers=[train_metrics, val_metrics, pn_metrics],
        stop_criteria=[EarlyStop(objective, patience=patience)],
        checkpoint_triggers=[BestMetric(objective)],
        objective=objective,
        batch_size=64
    )