from typing import Literal, Optional, Callable, Any
from core.training import Trainer, MetricsRecorder
import logging

from core.eval.objectives import Objective
from core.training import ResultsDict

logger = logging.getLogger(__name__)


class BestMetric:
    def __init__(self, 
                 objective : Objective) -> None:
        self.objective = objective
        self.best_result  : ResultsDict | None = None
        self.best_epoch : int | None = None

    def __call__(self, trainer : Trainer) -> bool:
        results = trainer.get_results_dict()
        value = self.objective.select_value(results)
        if results is None:
            return False
        if self.best_result is None or self.best_epoch is None:
            self.best_result = results
            self.best_epoch = trainer.epoch
            return False
        else:
            if self.objective.compare(results, self.best_result):
                self.best_result = results
                self.best_epoch = trainer.epoch
                logger.info(f"Triggering checkpoint: New best result at epoch {self.best_epoch} with objective {self.objective}: {value}")
                return True
            else:
                logger.debug('No improvement')
        return False