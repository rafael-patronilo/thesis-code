import logging
from typing import TYPE_CHECKING, Optional
if TYPE_CHECKING:
    from core.training.trainer import Trainer
    from core.eval.objectives import Objective

logger = logging.getLogger(__name__)

class GoalReached:
    def __init__(self, value, objective : Optional['Objective'] = None):
        self.value = value
        self.objective = objective
        self._disabled = False

    def __call__(self, trainer : 'Trainer') -> bool:
        if self._disabled:
            return False
        objective = self.objective if self.objective is not None else trainer.objective
        if objective is None:
            logger.warning("No objective set, BestValueReached will always return False")
            self._disabled = True
            return False
        else:
            results = trainer.get_results_dict()
            if results is None:
                return False
            return objective.select_value(results) == self.value