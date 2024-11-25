import torcheval.metrics
from core import Trainer, MetricsLogger, util, datasets
from core.metrics import Elapsed, metric_wrappers
from torch import nn
import torcheval
import torch

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
    metrics_per_clas = {
        'accuracy':torcheval.metrics.BinaryAccuracy,
        'f1_score':torcheval.metrics.BinaryF1Score
    }

    def metrics_factory() -> dict[str, torcheval.metrics.Metric]:
        return metric_wrappers.SelectCol.col_wise(dataset, metrics_per_clas) | {'epoch_elapsed':Elapsed()}
    
    train_metrics = MetricsLogger(
        identifier='train',
        metric_functions=metrics_factory(),
        dataset=dataset.for_training_eval
    )
    val_metrics = MetricsLogger(
        identifier='val',
        metric_functions=metrics_factory(),
        dataset=dataset.for_validation
    )
    return Trainer(
        model=create_model(layer_sizes, num_outputs),
        loss_fn=torch.nn.BCELoss(),
        optimizer=torch.optim.Adam,
        training_set=dataset,
        metric_loggers=[train_metrics, val_metrics],
    )