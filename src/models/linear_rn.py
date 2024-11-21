from core import Trainer, MetricsLogger, util, datasets
from core.metrics import get_metric
from torch import nn
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
    loss_fn = torch.nn.BCELoss()
    loss_metric = util.DecoratedTorchMetric(loss_fn) #TODO metrics here need to be updated
    dataset = datasets.get_dataset(dataset_name)
    metrics = ['accuracy', 'f1_score', 'epoch_elapsed']
    metric_functions : dict = {
        'loss': loss_metric,
    }
    for metric in metrics:
        metric_functions[metric] = get_metric(metric)
    train_metrics = MetricsLogger(
        identifier='train',
        metric_functions=metric_functions,
        dataset=dataset.for_training_eval
    )
    val_metrics = MetricsLogger(
        identifier='val',
        metric_functions=metric_functions,
        dataset=dataset.for_validation
    )
    return Trainer(
        model=create_model(layer_sizes, num_outputs),
        loss_fn=torch.nn.BCELoss(),
        optimizer=torch.optim.Adam,
        training_set=dataset,
        metric_loggers=[train_metrics, val_metrics],
    )