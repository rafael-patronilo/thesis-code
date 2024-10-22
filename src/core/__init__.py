from typing import NamedTuple, Any, Optional, Literal, Iterable

from .storage_management.model_file_manager import ModelFileManager
from .trainer import Trainer, ModelDetails
from .metrics_logger import MetricsLogger, NamedMetricFunction, MetricFunction
from .study_manager import StudyManager



from . import modules
from . import datasets
from torch.utils.data import DataLoader

def _replace_loss(loss_fn : str, metrics : Iterable[NamedMetricFunction]) -> Iterable[NamedMetricFunction]:
    loss_metric_added = False
    for x in metrics:
        if isinstance(x, str) and x.lower() == 'loss':
            if not loss_metric_added:
                yield (
                    'loss', 
                    modules.metrics.DecoratedTorchMetric(modules.get_loss_function(loss_fn))
                )
                loss_metric_added = True
        else:
            yield x

def prepare_new_model(
        model_name : str,
        model_identifier : str,
        model : Any,
        dataset_name : str,
        optimizer : str,
        loss_fn : str,
        val_metrics : Optional[Iterable[NamedMetricFunction]] = None,
        train_metrics : Optional[Iterable[NamedMetricFunction] | Literal['copy']] = 'copy',
        metric_loggers : Optional[Iterable[MetricsLogger]] = None,
        batch_size : int = 32,
    ):
    raise Exception("This function is deprecated.")
    assert modules.loss_function_exists(loss_fn)
    assert modules.optimizer_exists(optimizer)
    if metric_loggers is None:
        metric_loggers = []
    else:
        metric_loggers = list(metric_loggers)
    if train_metrics == 'copy':
        train_metrics = val_metrics
    dataset = datasets.get_dataset(dataset_name)
    loss_metric : NamedMetricFunction = (
        'loss', 
        modules.metrics.DecoratedTorchMetric(modules.get_loss_function(loss_fn)))
    if val_metrics is not None:
        metric_loggers.append(MetricsLogger(
            identifier = 'val',
            metric_functions = list(_replace_loss(loss_fn, val_metrics)),
            dataset = dataset.for_validation,
        ))
    if train_metrics is not None:
        metric_loggers.append(MetricsLogger(
            identifier = 'train',
            metric_functions = list(_replace_loss(loss_fn, train_metrics)),
            dataset = dataset.for_training_eval,
        ))
    x, _ = next(iter(DataLoader(dataset.for_training())))
    model(x) # initialize lazy layers
    with ModelFileManager(model_name, model_identifier, conflict_strategy='new') as file_manager:
        model_details = ModelDetails(
            architecture=model,
            optimizer=optimizer,
            loss_fn=loss_fn,
            dataset=dataset,
            metrics=list(metric_loggers),
            batch_size=batch_size,
        )
        # file_manager.save_model_config(model_details)

