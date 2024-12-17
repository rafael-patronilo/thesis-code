from typing import Literal, Optional, Callable, Any
from core import Trainer, MetricsRecorder
import logging

logger = logging.getLogger(__name__)


class BestMetric:
    def __init__(self, 
                 metric : str ='loss', 
                 prefer : Literal['max', 'min'] = 'min',
                 metrics_logger : str = 'val',
                 threshold : float = 0.005) -> None:
        self.best_value = None
        self.best_epoch = None
        self.metric = metric
        self.prefer : Callable[[Any, Any], bool] = (
            (lambda x, y : x > y) if prefer == 'max' else (lambda x, y : x < y)
        )
        self.metrics_logger = metrics_logger
        self.threshold = threshold

    def _select_metric(self, trainer : Trainer) -> Optional[float]:
        metrics_logger : MetricsRecorder = [x for x in trainer.metric_loggers if x.identifier == self.metrics_logger][0]
        if metrics_logger.last_record is None:
            return None
        return metrics_logger.last_record[self.metric]

    def __call__(self, trainer : Trainer) -> bool:
        value = self._select_metric(trainer)
        if value is None:
            return False
        if self.best_value is None:
            self.best_value = value
            self.best_epoch = trainer.epoch
            return False # No need to save the model on first epoch
        elif self.prefer(value, self.best_value) and abs(value - self.best_value) > self.threshold:
            self.best_value = value
            self.best_epoch = trainer.epoch
            logger.info(f"New best value for {self.metrics_logger} {self.metric} at epoch {self.best_epoch}: {self.best_value}")
            return True
        else:
            return False