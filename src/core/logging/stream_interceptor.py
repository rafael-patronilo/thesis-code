import logging
import time
from datetime import timedelta, datetime
from io import TextIOWrapper
import threading


class WriteInterceptor:
    WARNING_COOLDOWN = timedelta(minutes=5)
    BUFFER_TIMER = timedelta(seconds=5)

    def __init__(
            self,
            stream_name: str,
            logger: logging.Logger,
            level,
            expected: bool = False
    ):
        self.logger = logger
        self.level = level
        self.stream_name = stream_name
        self.last_warning = None
        self._buffer: list[str] = []
        self._buffer_lock = threading.Lock()
        self._msg_event = threading.Event()
        self._flush_thread = threading.Thread(target=self._flush_worker, daemon=True)
        self._expected = expected

    def write(self, message, /) -> int:
        now = datetime.now()
        if not self._expected and (self.last_warning is None or now - self.last_warning >= self.WARNING_COOLDOWN):
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

    def _flush_worker(self):
        while True:
            self._msg_event.wait()
            time.sleep(self.BUFFER_TIMER.total_seconds())
            self.flush()

class StreamInterceptor(TextIOWrapper):
    def __init__(
            self,
            stream,
            stream_name: str,
            logger: logging.Logger,
            level,
            expected: bool = False):
        super().__init__(stream)
        self._write_interceptor = WriteInterceptor(stream_name, logger, level, expected)
    def write(self, message, /) -> int:
        return self._write_interceptor.write(message)

    def flush(self):
        self._write_interceptor.flush()