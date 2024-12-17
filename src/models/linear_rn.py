import torcheval.metrics
from core import Trainer, MetricsRecorder, datasets
from core.training.checkpoint_triggers.best_metric import BestMetric
import core.eval.metrics
from core.eval.metrics import Elapsed
from core.eval.metrics import metric_wrappers
from torch import nn
import torcheval
import torch

from core.training.stop_criteria import EarlyStop


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
        'accuracy': metric_wrappers.to_int(torcheval.metrics.BinaryAccuracy),
        'f1_score': metric_wrappers.to_int(torcheval.metrics.BinaryF1Score),
        'precision': metric_wrappers.to_int(torcheval.metrics.BinaryPrecision),
        'recall': metric_wrappers.to_int(torcheval.metrics.BinaryRecall),
        'auc': metric_wrappers.to_int(torcheval.metrics.BinaryAUROC),
        'balanced_accuracy': metric_wrappers.to_int(core.metrics.BinaryBalancedAccuracy),
        'specificity': metric_wrappers.to_int(core.metrics.BinarySpecificity),
        'pos_rate': metric_wrappers.to_int(core.metrics.BinaryPositiveRate),
    }

    def metrics_factory() -> dict[str, torcheval.metrics.Metric]:
        return (
                metric_wrappers.SelectCol.col_wise(dataset, metrics_per_class) |
                {
                f'min_{name}' : metric_wrappers.MinOf.foreach_class(dataset, metric) 
                for name, metric in metrics_per_class.items()
            } |
                {'epoch_elapsed' : Elapsed()})
    
    train_metrics = MetricsRecorder(
        identifier='train',
        metric_functions=metrics_factory(),
        dataset=dataset.for_training_eval
    )
    val_metrics = MetricsRecorder(
        identifier='val',
        metric_functions=metrics_factory(),
        dataset=dataset.for_validation
    )

    return Trainer(
        model=create_model(layer_sizes, num_outputs),
        loss_fn=torch.nn.BCELoss(),
        optimizer=torch.optim.Adam,
        training_set=dataset,
        checkpoint_each=25,
        metric_loggers=[train_metrics, val_metrics],
        stop_criteria=[EarlyStop(
            metric='min_balanced_accuracy', prefer='max', metrics_logger ='val', threshold=0.001, patience=20)],
        checkpoint_triggers=[BestMetric(
            metric='min_balanced_accuracy', prefer='max', metrics_logger ='val', threshold=0.01)],
        display_metrics=['min_balanced_accuracy', 'balanced_accuracy', 'auc', 'balanced_accuracy_valid', 'auc_valid', 'pos_rate_valid', 'f1_score', 'accuracy', 'epoch_elapsed'],
    )