from typing import NamedTuple, Any, Optional, Literal, Iterable

from .storage_management.model_file_manager import ModelFileManager
from .trainer import Trainer, ModelDetails
from .metrics_logger import MetricsLogger, TrainingLogger, NamedMetricFunction, MetricFunction
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

