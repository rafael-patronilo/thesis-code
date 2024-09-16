from typing import NamedTuple, Any, Optional
ModelDetails = NamedTuple(
    "ModelDetails",
    [
        ("architecture", Any),
        ("optimizer", str),
        ("loss_fn", str),
        ("dataset", str),
        ("metrics", list[str]),
        ("batch_size", int),
        ("train_metrics", Optional[list[str]]),
    ]
)

from .trainer import Trainer
from .metrics_logger import MetricsLogger
from .model_file_manager import ModelFileManager

from . import modules
from . import datasets
from torch.utils.data import DataLoader


def prepare_new_model(
        model_name : str,
        model_identifier : str,
        model : Any,
        dataset : str,
        optimizer : str,
        loss_fn : str,
        metrics : list[str],
        train_metrics : Optional[list[str]] = None,
        batch_size : int = 32,
    ):
    assert modules.loss_function_exists(loss_fn)
    assert modules.optimizer_exists(optimizer)
    assert all(modules.metrics.metric_exists(metric) for metric in metrics)
    x, _ = next(iter(DataLoader(datasets.dataset_registry[dataset].for_training())))
    model(x) # initialize lazy layers
    with ModelFileManager(model_name, model_identifier, conflict_strategy='new') as file_manager:
        model_details = ModelDetails(
            architecture=model,
            optimizer=optimizer,
            loss_fn=loss_fn,
            dataset=dataset,
            metrics=metrics,
            batch_size=batch_size,
            train_metrics=train_metrics
        )
        file_manager.save_model_details(model_details)