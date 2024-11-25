

from typing import Any, Iterable
import time
from torcheval.metrics import Metric


class Elapsed(Metric):

    def __init__(self) -> None:
        super().__init__()
        self.last_time = None

    def reset(self) -> None:
        self.last_time = time.time()

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        return self.compute()
        
    def compute(self) -> float:
        if self.last_time is None:
            self.last_time = time.time()
            return float('nan')
        else:
            current_time = time.time()
            elapsed = current_time - self.last_time
            self.last_time = current_time
            return elapsed
        
    def update(self, *_, **__) -> None:
        pass

    def merge_state(self, metrics: Iterable['Elapsed']) -> 'Elapsed':
        raise NotImplementedError("Merging not implemented")