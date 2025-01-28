import logging
import torcheval.metrics
from torcheval.metrics import Metric

from core import datasets
from core.training import Trainer, MetricsRecorder
from core.training.checkpoint_triggers.best_metric import BestMetric
from core.training.stop_criteria import GoalReached
import core.eval.metrics
from core.eval.metrics import Elapsed
from core.eval.metrics import metric_wrappers
from torch import nn
import torcheval
import torch

from core.training.stop_criteria import EarlyStop
from core.eval.objectives import Maximize, Minimize
from core.training.metrics_recorder import TrainingRecorder


def create_model(layer_sizes : list[int], num_outputs : int) -> nn.Module:
    layers = []
    for size in layer_sizes:
        layers.append(nn.LazyLinear(size))
        layers.append(nn.ReLU())
    layers.append(nn.LazyLinear(num_outputs))
    layers.append(nn.Sigmoid())
    return nn.Sequential(*layers)

def create_trainer(layer_sizes : list[int], num_outputs : int, dataset_name : str) -> Trainer:
    dataset = datasets.get_dataset(dataset_name)
    metrics_per_class = {
        'balanced_accuracy': metric_wrappers.to_int(core.eval.metrics.BinaryBalancedAccuracy),
    }

    def metrics_factory() -> dict[str, torcheval.metrics.Metric]:
        metrics : dict[str, Metric] = {'epoch_elapsed' : Elapsed()}
        metric_wrappers.SelectCol.col_wise(dataset, metrics_per_class,
                                           reduction='min', out_dict=metrics)
        return metrics
    
    train_metrics = TrainingRecorder(
        metric_functions=metrics_factory()
    )

    objective = Maximize('train', 'balanced_accuracy', threshold=0.01)
    patience_objective = Minimize('train', 'loss', threshold=0.001)

    return Trainer(
        model=create_model(layer_sizes, num_outputs),
        loss_fn=torch.nn.BCELoss(),
        optimizer=torch.optim.Adam,
        training_set=dataset,
        checkpoint_each=25,
        objective=objective,
        metric_loggers=[train_metrics],
        stop_criteria=[EarlyStop(patience_objective, patience=20), GoalReached(1.0)],
        checkpoint_triggers=[BestMetric(objective)],
    )