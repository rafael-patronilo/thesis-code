import logging
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from core.training.trainer import Trainer

logger = logging.getLogger(__name__)

class GoalReached:
    def __init__(self, value):
        self.value = value
        self._disabled = False

    def __call__(self, trainer : 'Trainer') -> bool:
        if self._disabled:
            return False
        if trainer.objective is None:
            logger.warning("No objective set, BestValueReached will always return False")
            self._disabled = True
            return False
        else:
            results = trainer.get_results_dict()
            if results is None:
                return False
            return trainer.objective.select_value(results) == self.value