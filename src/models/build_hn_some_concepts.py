from typing import Literal, Optional
from core import datasets
from core.eval import metrics
from core.eval.metrics import metric_wrappers
from core.training import Trainer, MetricsRecorder, TrainingRecorder
from core.nn.hybrid_network import HybridNetwork
from core.nn import PartiallyPretrained
import torch
from torcheval.metrics import Mean, Metric
from torch import nn

from core.nn.autoencoder import AutoEncoder
import logging

from core.training.metrics_recorder import EvaluationResult
from core.training.trainer import TrainerConfig, ModelLoadPath
import numpy as np

from torch.utils.data import Dataset

from core.nn.loss_wrappers import WeightedTarget
from core.datasets.dataset_wrappers import SelectCols
from core.eval.objectives import Maximize
from core.training.checkpoint_triggers.best_metric import BestMetric


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
        reasoning_network=reasoning_network,
        output_includes_concepts=True
    )

def sub_sample_order(total_samples : int, seed : int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    order = np.arange(total_samples)
    rng.shuffle(order)
    return order

def select_sub_samples(total_samples : int, sub_samples : int, seed : int) -> set[int]:
    order = sub_sample_order(total_samples, seed)
    return set(order[:sub_samples])

class SomeConceptsDatasetWrapper(Dataset):
    def __init__(
            self,
            inner : Dataset,
            class_indices : list[int],
            concept_indices : list[int],
            concept_samples: set[int],
            class_weights: Optional[torch.Tensor] = None,
            concept_weights : Optional[torch.Tensor] = None
    ):
        self.inner = inner
        self.class_indices = class_indices
        self.concept_indices = concept_indices
        self.concept_samples = concept_samples
        self.class_weights = class_weights if class_weights is not None else torch.ones(len(class_indices) + 1)
        self.concept_weights = concept_weights if concept_weights is not None else torch.ones(len(concept_indices))
        # Weight tensors must be in the cpu because of multiprocessing dataloaders
        self.class_weights = self.class_weights.cpu()
        self.concept_weights = self.concept_weights.cpu()

    def __getitem__(self, index):
        x, y = self.inner[index]
        y_classes = y[self.class_indices]

        if index in self.concept_samples:
            y_concepts = y[self.concept_indices]
            y_concept_weights = self.concept_weights
        else:
            y_concepts = torch.zeros_like(y[self.concept_indices])
            y_concept_weights = torch.zeros_like(self.concept_weights)
        new_y = torch.cat((
            y_concepts,
            y_classes,
            torch.tensor([1.0]) # valid column
        ))
        y_weights = torch.cat((
            y_concept_weights,
            self.class_weights
        ))
        return x, torch.vstack((new_y, y_weights))

    def __len__(self):
        return len(self.inner) # type: ignore

    def __repr__(self):
        return f"< {self.inner} where {len(self.concept_samples)} contain concepts >"

def pn_evaluator(model : 'HybridNetwork', x, y):
    return EvaluationResult(model.perception_network(x), y)

def create_trainer(
        dataset_name : str,
        concepts : list[str],
        classes : list[str],
        num_samples_with_concepts : int,
        sample_selection_seed : int,
        pre_trained_learning_rate : float = 0.00001,
        untrained_learning_rate : float = 0.001,
        rn_learning_rate : float | None = None,
        valid_col_weight : float = 1,
        **kwargs) -> Trainer:
    dataset = datasets.get_dataset(dataset_name)
    training_set = dataset.for_training() # make sure it is loaded
    training_set_size = len(training_set) # type: ignore

    col_refs = dataset.get_column_references()
    logger.info(f"Column info for dataset: {col_refs}")

    class_indices = [col_refs.labels.names_to_column[c] for c in classes]
    valid_col_idx = len(class_indices)
    weights = torch.ones(len(class_indices) + 1)
    weights[valid_col_idx] = valid_col_weight
    logger.info(f"Loss class weights: {weights}")

    concept_indices = [col_refs.labels.names_to_column[c] for c in concepts]

    training_set = SomeConceptsDatasetWrapper(
        training_set,
        class_indices = class_indices,
        concept_indices = concept_indices,
        concept_samples = select_sub_samples(
            training_set_size, num_samples_with_concepts, sample_selection_seed),
        class_weights = weights
    )

    def metric_functions(*, include_concepts : bool, target_has_weights : bool):
        def target_selector(x : Metric):
            if not target_has_weights:
                return metric_wrappers.SelectCol(x, 1, apply_to_preds=False)
            else:
                return x

        metric_functions_ : dict = {
            'elapsed': metrics.Elapsed(),
            'mean_valid': target_selector(metric_wrappers.SelectCol(metric_wrappers.Unary(Mean()), valid_col_idx))
        }
        per_col_metrics = {
            'balanced_accuracy': lambda: target_selector(metric_wrappers.to_int(
                metrics.BinaryBalancedAccuracy)()),
        }

        metric_functions_.update(**metric_wrappers.SelectCol.col_wise(
                             ([None] * len(concepts)) + classes,
                             per_col_metrics,
                             reduction='min'))
        if include_concepts:
            metric_functions_.update(**metric_wrappers.SelectCol.col_wise(
                concepts,
                per_col_metrics,
                reduction=None
            ))
        return metric_functions_
    val_metrics = MetricsRecorder(
        identifier='val',
        metric_functions=metric_functions(include_concepts=True, target_has_weights=False),
        dataset=SelectCols(dataset, select_y=concept_indices + class_indices).for_validation
    )
    train_metrics = TrainingRecorder(
        metric_functions=metric_functions(include_concepts=False, target_has_weights=True)
    )
    metric_recorders = [train_metrics, val_metrics]

    model = create_model(len(concepts), **kwargs)
    objective = Maximize('val', 'balanced_accuracy', threshold=0.01)

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

    loss_fn = WeightedTarget(nn.functional.binary_cross_entropy)

    return Trainer(
        model=model,
        loss_fn=loss_fn,
        optimizer=optimizer,
        training_set=training_set,
        metric_loggers=metric_recorders,
        objective=objective,
        checkpoint_triggers=[BestMetric(objective)],
        batch_size=64
    )