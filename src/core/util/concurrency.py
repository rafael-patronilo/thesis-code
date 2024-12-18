import threading
import time
from datetime import datetime, timedelta


class RateLimiter:
    def __init__(self, rate : int, period : timedelta | float, logarithmic_reset : bool = False):
        self.rate = rate
        if isinstance(period, timedelta):
            period = period.total_seconds()
        self.period : float = period
        self._lock = threading.RLock()
        self.count = 0
        self.reset_at = time.time() + period
        self.logarithmic_reset = logarithmic_reset
        self.kill_switch = threading.Event()

    def reset(self, now : float | None = None):
        if now is None:
            now = time.time()
        with self._lock:
            if self.logarithmic_reset:
                self.count = self.count // 2
            else:
                self.count = 0
            self.reset_at = now + self.period

    def will_wait(self) -> float:
        with self._lock:
            now = time.time()
            if self.count >= self.rate - 1:
                return self.reset_at - now
            return 0.0

    def limit(self):
        wait : float | None = None
        with self._lock:
            self.count += 1
            now = time.time()
            if now >= self.reset_at:
                self.reset(now)
            elif self.count >= self.rate:
                wait = self.reset_at - now
        if wait is not None:
            self.kill_switch.wait(wait)