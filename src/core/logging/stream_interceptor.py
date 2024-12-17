import logging
import time
from datetime import timedelta, datetime
from io import TextIOWrapper
import threading


class StreamInterceptor(TextIOWrapper):
    WARNING_COOLDOWN = timedelta(minutes=5)
    BUFFER_TIMER = timedelta(seconds=1)

    def __init__(self, stream, stream_name : str, logger : logging.Logger, level):
        self.logger = logger
        self.level = level
        self.stream_name = stream_name
        self.last_warning = None
        self._buffer : list[str] = []
        self._buffer_lock = threading.Lock()
        self._msg_event = threading.Event()
        self._flush_thread = threading.Thread(target=self._flush_worker, daemon=True)
        super().__init__(stream)

    def _flush_worker(self):
        while True:
            self._msg_event.wait()
            time.sleep(self.BUFFER_TIMER.total_seconds())
            self.flush()

    def write(self, message) -> int:
        now = datetime.now()
        if self.last_warning is None or now - self.last_warning >= self.WARNING_COOLDOWN:
            self.logger.warning(
                f"Intercepted unexpected write to {self.stream_name}, please replace with proper logging\n"
                f"This warning will be suppressed for {self.WARNING_COOLDOWN}",
                exc_info=True
            )
            self.last_warning = now
        with self._buffer_lock:
            self._buffer.append(message)
            self._msg_event.set()
        return len(message)

    def flush(self):
        with self._buffer_lock:
            if self._buffer:
                self.logger.log(self.level, "".join(self._buffer))
                self._buffer.clear()
            self._msg_event.clear()
