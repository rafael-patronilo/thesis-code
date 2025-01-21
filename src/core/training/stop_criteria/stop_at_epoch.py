
from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from core.training import Trainer

class StopAtEpoch:
    def __init__(self, max_epoch: int):
        self.max_epoch = max_epoch

    def __call__(self, trainer : 'Trainer') -> bool:
        return trainer.epoch >= self.max_epoch

    def total_epochs(self, trainer : 'Trainer') -> int:
        if trainer.first_epoch:
            return self.max_epoch
        else:
            return self.max_epoch - trainer.epoch