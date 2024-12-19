
from .early_stop import EarlyStop
from core.training import Trainer

class StopAtEpoch:
    def __init__(self, max_epochs : int) -> None:
        self.max_epochs = max_epochs
    
    def __call__(self, trainer : Trainer) -> bool:
        return trainer.epoch >= self.max_epochs