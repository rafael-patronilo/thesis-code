

from typing import Any
import time


class EpochTime:

    def __init__(self) -> None:
        self.last_time = None

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        if self.last_time is None:
            self.last_time = time.time()
            return float('nan')
        else:
            current_time = time.time()
            elapsed = current_time - self.last_time
            self.last_time = current_time
            return elapsed