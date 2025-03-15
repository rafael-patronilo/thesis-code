from typing import Literal, Optional

import torcheval.metrics

from core import datasets
from core.datasets import dataset_wrappers
from core.eval.objectives import Maximize
from core.eval import metrics
from core.eval.metrics import metric_wrappers
from core.training import Trainer, MetricsRecorder, TrainingRecorder
from core.nn.hybrid_network import HybridNetwork
from core.nn import PartiallyPretrained
import torch
from torcheval.metrics import Mean
from torch import nn

from core.nn.autoencoder import AutoEncoder
import logging

from core.training.checkpoint_triggers import BestMetric
from core.training.stop_criteria import EarlyStop
from core.training.metrics_recorder import EvaluationResult
from core.training.trainer import TrainerConfig, ModelLoadPath


logger = logging.getLogger(__name__)

def modify_perception_network(
        perception_network : nn.Module,
        num_concepts : int,
        dropout_last_pn : Optional[float],
        activation : Literal['sigmoid', 'relu'] = 'sigmoid',
):
    if isinstance(perception_network, AutoEncoder):
        logger.info("Selecting encoder")
        encoder = perception_network.encoder
        return modify_perception_network(encoder, num_concepts, dropout_last_pn)
    elif isinstance(perception_network, nn.Sequential):
        layers = [(layer, True) for layer in perception_network]
        layers.pop()
        if layers[-1][0].out_features != num_concepts:
            logger.info("Replacing last linear with correct size")
            removed, _ = layers.pop()
            layers.append((nn.Linear(removed.in_features, num_concepts), False))
        if activation == 'sigmoid':
            logger.info("Setting last activation to sigmoid")
            layers.append((nn.Sigmoid(), False))
        else:
            logger.info("Setting last activation to ReLU")
            layers.append((nn.ReLU(), False))
        if dropout_last_pn is not None:
            logger.info(f"Adding dropout to last layer with p={dropout_last_pn}")
            layers.append((nn.Dropout(dropout_last_pn), False))
        return PartiallyPretrained(*layers)
    else:
        if dropout_last_pn is not None:
            logger.info(f"Adding dropout to last layer with p={dropout_last_pn}")
            return nn.Sequential(perception_network, nn.Dropout(dropout_last_pn))
        logger.warning("No modification applied.")
        return perception_network




def create_model(
        num_concepts: int,
        reasoning_network_config: ModelLoadPath | TrainerConfig,
        perception_network_config : ModelLoadPath | TrainerConfig,
        activation : Literal['sigmoid', 'relu'] = 'sigmoid',
        dropout_last_pn : Optional[float] = None,
    ) -> HybridNetwork:
    perception_network = Trainer.model_from_path_or_config(perception_network_config)
    perception_network = modify_perception_network(perception_network, num_concepts, dropout_last_pn, activation)
    reasoning_network = Trainer.model_from_path_or_config(reasoning_network_config)
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
        rn_learning_rate : float | None = None,
        skip_pn_eval : bool = False,
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
    patience = kwargs.pop('patience', 20)
    threshold = kwargs.pop('threshold', 0.01)

    classes : list[str] = col_refs.labels.columns_to_names
    def metric_functions():
        metric_functions_ : dict = {
            'elapsed': metrics.Elapsed(),
            'mean_valid': metric_wrappers.SelectCol(metric_wrappers.Unary(Mean()), valid_col)
        }
        metric_functions_.update(**metric_wrappers.SelectCol.col_wise(classes, {
            'balanced_accuracy' : metric_wrappers.to_int(metrics.BinaryBalancedAccuracy),
            'accuracy': metric_wrappers.to_int(torcheval.metrics.BinaryAccuracy),
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
    metric_recorders = [train_metrics, val_metrics]
    if not skip_pn_eval:
        pn_metric_functions = {}
        for concept in concepts:
            pn_metric_functions.update(
                {
                    f"balanced_accuracy_{concept}": metric_wrappers.SelectCol(
                        metric_wrappers.to_int(metrics.BinaryBalancedAccuracy)(),
                        concept_col_refs.labels.names_to_column[concept]
                    ),
                    f"correlation_{concept}": metric_wrappers.SelectCol(
                        metrics.PearsonCorrelationCoefficient(),
                        concept_col_refs.labels.names_to_column[concept]
                    )
                }
            )
        pn_metrics = MetricsRecorder(
            identifier='pn_val',
            metric_functions=pn_metric_functions,
            dataset=concept_dataset.for_validation,
            evaluator=pn_evaluator
        )
        metric_recorders.append(pn_metrics)
    objective = Maximize('val', 'balanced_accuracy', threshold=threshold)

    model = create_model(len(concepts), **kwargs)

    def optimizer(_) -> torch.optim.Optimizer:
        pn = model.perception_network
        param_groups = []
        if isinstance(pn, PartiallyPretrained):
            param_groups.append({'params': pn.pre_trained_parameters(), 'lr': pre_trained_learning_rate})
            param_groups.append({'params': pn.untrained_parameters(), 'lr': untrained_learning_rate})
        else:
            param_groups.append({'params': pn.parameters(), 'lr' : untrained_learning_rate})
        if rn_learning_rate is not None:
            param_groups.append({'params': model.reasoning_network.parameters(), 'lr': rn_learning_rate})
        return torch.optim.Adam(param_groups)

    return Trainer(
        model=model,
        loss_fn=loss_fn,
        optimizer=optimizer,
        training_set=dataset,
        metric_loggers=metric_recorders,
        stop_criteria=[EarlyStop(objective, patience=patience)],
        checkpoint_triggers=[BestMetric(objective)],
        objective=objective,
        batch_size=64
    )