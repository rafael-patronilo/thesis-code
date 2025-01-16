
from .early_stop import EarlyStop
from .goal_reached import GoalReached
from core.training import Trainer
__all__=[
    'EarlyStop',
    'StopAtEpoch',
    'GoalReached'
]

class StopAtEpoch:
    def __init__(self, max_epochs : int) -> None:
        self.max_epochs = max_epochs
    
    def __call__(self, trainer : Trainer) -> bool:
        return trainer.epoch >= self.max_epochs