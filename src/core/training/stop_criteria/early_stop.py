from math import log
from typing import Literal, Optional, Callable, Any
from core.training import Trainer, MetricsRecorder
import logging
from core.eval.objectives import Objective
from core.training.trainer import ResultsDict

logger = logging.getLogger(__name__)

class EarlyStop:
    def __init__(self, 
                 objective : Objective,
                 patience : int = 10) -> None:
        self.best_result : ResultsDict | None = None
        self.best_epoch : int | None = None
        self.objective = objective
        self.patience = patience
    
    def state_dict(self):
        return dict(
            best_result = self.best_result,
            best_epoch = self.best_epoch
        )
    
    def load_state_dict(self, state):
        self.best_result = state['best_result']
        self.best_epoch = state['best_epoch']
    
    def __call__(self, trainer : Trainer) -> bool:
        results = trainer.get_results_dict()
        if results is None:
            return False
        value = self.objective.select_value(results)
        if self.best_result is None or self.best_epoch is None:
            if not trainer.first_epoch:
                self.best_result = results
                self.best_epoch = trainer.epoch
                logger.info(f"Initializing early stop with objective {self.objective}: {value = }")
            return False
        else:
            if self.objective.compare(results, self.best_result):
                self.best_result = results
                self.best_epoch = trainer.epoch
                logger.info(f"New best result at epoch {self.best_epoch} with objective {self.objective}: {value}")
            else:
                logger.debug('No improvement')
            if trainer.epoch - self.best_epoch >= self.patience:
                logger.info(
                    f"Early stopping at epoch {trainer.epoch} with objective {self.objective}\n"
                    f"Best value recorded was {self.objective.select_value(self.best_result)} at epoch {self.best_epoch}\n"
                    f"Training failed to improve on the value with patience {self.patience}")
                return True
        return False