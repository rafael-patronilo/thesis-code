from typing import Literal, Optional, Callable, Any
#from core.training import Trainer, MetricsRecorder
import logging

logger = logging.getLogger(__name__)

class EarlyStop:
    def __init__(self, 
                 metric : str ='loss', 
                 prefer : Literal['max', 'min'] = 'min',
                 metrics_logger : str = 'val',
                 threshold : Optional[float] = None,
                 patience : int = 10) -> None:
        self.best_value = None
        self.best_epoch = None
        self.metric = metric
        self.prefer : Callable[[Any, Any], bool] = (
            (lambda x, y : x > y) if prefer == 'max' else (lambda x, y : x < y)
        )
        self.threshold = threshold
        self.metrics_logger = metrics_logger
        self.patience = patience
    
    def state_dict(self):
        return dict(
            best_value = self.best_value,
            best_epoch = self.best_epoch
        )
    
    def load_state_dict(self, state):
        self.best_value = state['best_value']
        self.best_epoch = state['best_epoch']
    

    def _select_metric(self, trainer : Trainer) -> Optional[float]:
        metrics_logger : MetricsRecorder = [x for x in trainer.metric_loggers if x.identifier == self.metrics_logger][0]
        if metrics_logger.last_record is None:
            return None
        return metrics_logger.last_record[self.metric]
    
    def __call__(self, trainer : Trainer) -> bool:
        value = self._select_metric(trainer)
        if value is None:
            return False
        if self.best_value is None or self.best_epoch is None:
            self.best_value = value
            self.best_epoch = trainer.epoch
            logger.info(f"Initializing early stop with first value of {self.metrics_logger} {self.metric}: {self.best_value}")
            return False
        else:
            if self.prefer(value, self.best_value):
                if self.threshold is None or abs(self.best_value - value) >= self.threshold:
                    self.best_value = value
                    self.best_epoch = trainer.epoch
                    logger.info(f"New best value for {self.metrics_logger} {self.metric} at epoch {self.best_epoch}: {self.best_value}")
                    return False
                else:
                    logger.debug("Improvement under the threshold, not updating.")
            if trainer.epoch - self.best_epoch >= self.patience:
                logger.info(
                    f"Early stopping at epoch {trainer.epoch} with {self.metrics_logger} {self.metric}\n"
                    f"Best value recorded was {self.best_value} at epoch {self.best_epoch}\n"
                    f"Training failed to improve on the value by {self.threshold} with patience {self.patience}")
                return True
        return False