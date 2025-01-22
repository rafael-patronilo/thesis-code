import http.client
import queue
import threading
from collections import deque
from copy import copy
from urllib.parse import urlparse
import logging
from logging import LogRecord
from typing import Callable, NamedTuple, override

from numpy import rec
from core.util.concurrency import RateLimiter
import json
from queue import Queue
from datetime import datetime, timedelta
import core.init


DISCORD_FORMAT = '`%(asctime)s` [%(levelname)s]\n-# (%(name)s|%(threadName)s)\n```%(message)s```' # noqa

MESSAGE_MAX_LENGTH = 1900 # actual max is 2000 but there seems to be some discrepancy
LOG_RECORD_MAX_LENGTH = 1500
RECORD_EXPIRY = timedelta(seconds=10)
ERROR_COOLDOWN = timedelta(minutes=5)

# discord rate limit seems to be 5 messages per 2 seconds
DISCORD_RATE_LIMIT : dict = dict(rate=5, period=2, logarithmic_reset = True)
CLOSE_TIMEOUT = timedelta(minutes=1)

class DiscordWebhookClient:
    def __init__(self, webhook_url: str) -> None:
        self.webhook_url = webhook_url
        parsed_url = urlparse(webhook_url)
        self.connection =  http.client.HTTPSConnection(parsed_url.netloc)
        self.webhook_path = parsed_url.path

    def _make_request(self, path : str, method: str, payload: str | None):
        self.connection.connect()
        headers = {}
        response_str = None
        if payload is not None:
            headers["Content-Type"] = "application/json"
        self.connection.request(method, path, payload, headers)
        response = self.connection.getresponse()
        try:
            response_str = str(response.read())
            if not (200 <= response.status < 300):
                raise Exception(f"Unexpected response code: {response.status}\n{response.msg}\n{response_str}"
                                f"\nPayload: {payload}")
        finally:
            if response_str is None:
                response.read()
            response.close()

    def send(self, content : str):
        """
        Incomplete implementation of https://discord.com/developers/docs/resources/webhook#execute-webhook

        :param content: the message contents (up to 2000 characters)

        :return:
        """
        payload = json.dumps({"content": content})
        self._make_request(
            self.webhook_path, "POST", payload)

    def close(self):
        self.connection.close()


class PendingRecord(NamedTuple):
    msg : str
    level : int
    timestamp : datetime

class DiscordLogConsumer:
    def __init__(self,
                 webhook_url : str,
                 should_discard : Callable[[PendingRecord], bool],
                 logger : logging.Logger):
        # Called by other threads
        self.pending : Queue[PendingRecord | None] = Queue()
        self._local_pending : deque[PendingRecord] = deque()
        self.rate_limiter = RateLimiter(**DISCORD_RATE_LIMIT)
        self.client = DiscordWebhookClient(webhook_url)
        self.should_discard = should_discard
        self.kill_switch = threading.Event()
        self.soft_kill_switch = threading.Event()
        self.complete = threading.Event()
        self.logger = logger
        self.thread = threading.Thread(
            name=self.__class__.__name__, target=self.run, daemon=True)

    def soft_kill(self):
        """
        Signal end of program so consumer thread tries to go over tasks faster.
        """
        # Called by other threads
        self.logger.debug("Soft killing discord log consumer")
        self.soft_kill_switch.set()
        self.pending.put(None)

    def kill(self):
        """
        Kill the consumer thread immediately and close the client
        """
        # Called by other threads
        self.logger.debug("Killing discord log consumer")
        if self.thread.is_alive():
            self.kill_switch.set()
            self.soft_kill_switch.set()
            self.rate_limiter.kill_switch.set()
            self.pending.put(None)
        self.client.close()

    def _get_discarding(self, block : bool) -> PendingRecord:
        # called by self.thread
        discarded = 0
        def suppressed_record():
            return PendingRecord(
                f"***(suppressed {discarded} log records)***",
                logging.CRITICAL,
                datetime.now()
            )
        while not self.kill_switch.is_set():
            try:
                if len(self._local_pending) > 0:
                    record = self._local_pending.pop()
                else:
                    record = self.pending.get(block=block)
                block = False
                if record is None:
                    raise StopIteration("None record found")
                if self.should_discard(record):
                    discarded += 1
                else:
                    if discarded > 0:
                        self._local_pending.append(record)
                        return suppressed_record()
                    else:
                        return record
            except queue.Empty:
                if discarded > 0:
                    return suppressed_record()
                else:
                    raise
        if discarded > 0:
            return suppressed_record()
        raise StopIteration("No record found")


    def _accum_records(self) -> list[PendingRecord]:
        # Called by self.thread
        records = []
        size_sum = 0
        block = not self.soft_kill_switch.is_set()
        while not self.kill_switch.is_set():
            try:
                record = self._get_discarding(block=block)
            except (queue.Empty, StopIteration):
                if len(records) == 0:
                    raise
                break
            block = False
            # message should already be truncated,
            # however if the message is too long this may loop endlessly
            assert len(record.msg) <= MESSAGE_MAX_LENGTH
            if size_sum + len(record.msg) > MESSAGE_MAX_LENGTH:
                self._local_pending.append(record)
                break
            records.append(record)
            size_sum += len(record.msg)
            if len(records) > 1:
                size_sum += 1
        return records

    def _rate_limit(self):
        # Called by self.thread
        if not (
                self.soft_kill_switch.is_set() and
                self.pending.qsize() == 0 and
                len(self._local_pending) == 0
        ):
            self.rate_limiter.limit()

    def run(self):
        # Called by self.thread
        while not self.kill_switch.is_set():
            try:
                records = self._accum_records()
                if len(records) == 0:
                    break # Soft kill
                content = "\n".join(rec.msg for rec in records)
                self._rate_limit()
                self.client.send(content)
            except (queue.Empty, StopIteration) as e:
                if self.kill_switch.is_set():
                    self.logger.debug("Killed", exc_info=True)
                    break
                elif self.soft_kill_switch.is_set():
                    self.logger.debug("Soft killed", exc_info=True)
                    break
                else:
                    self.logger.error("Unexpected Empty/StopIteration, "
                                      f" disabling discord logging for {ERROR_COOLDOWN}", exc_info=True)
                    self.soft_kill_switch.wait(ERROR_COOLDOWN.total_seconds())
            except Exception as e: # noqa
                self.logger.error(f"Error sending log message to discord,"
                                  f" disabling discord logging for {ERROR_COOLDOWN}",
                                  exc_info=True)
                if self.soft_kill_switch.is_set():
                    break
                self.soft_kill_switch.wait(ERROR_COOLDOWN.total_seconds())
        if not (self.pending.qsize() == 0 and len(self._local_pending) == 0):
            try:
                self.client.send("# _(LOGGING KILLED PREMATURELY)_")
            except: pass # noqa # ignore any errors
        self.complete.set()

    def join(self, timeout : float | None = None):
        # Called by other threads
        self.complete.wait(timeout)

    def start(self):
        # Called by other threads
        self.thread.start()


class DiscordFormatter(logging.Formatter):
    def __init__(self, fmt: str, level_map : dict[str, str]):
        self.level_map = level_map
        super().__init__(fmt)

    @override
    def formatMessage(self, record: LogRecord) -> str:
        record = copy(record)
        # noinspection SpellCheckingInspection
        record.levelname = self.level_map.get(record.levelname, record.levelname)
        record.message = record.message.replace("\t", "    ") # discord doesn't like tabs
        max_length = LOG_RECORD_MAX_LENGTH + 3
        trunc_msg = ""
        trunc = len(record.message) - max_length
        if trunc > 0:
            record.message = record.message[:max_length] + '...'
            trunc_msg = f"\n-# _(truncated {trunc} characters)_"
        result = super().formatMessage(record)
        return result+trunc_msg

    @override
    def format(self, record: LogRecord) -> str:
        """
        Copied from logging.Formatter.format with modifications
        to include the exception text in the message.
        Stack is purposefully ignored.

        :param record:

        :return:
        """
        record.message = record.getMessage()
        if self.usesTime():
            # noinspection SpellCheckingInspection
            record.asctime = self.formatTime(record, self.datefmt)
        if record.exc_info:
            # Cache the traceback text to avoid converting it multiple times
            # (it's constant anyway)
            if not record.exc_text:
                record.exc_text = self.formatException(record.exc_info)
        if record.exc_text:
            if record.message[-1] != "\n":
                record.message = record.message + "\n"
            record.message = record.message + record.exc_text
        return self.formatMessage(record)


class DiscordWebhookLogHandler(logging.Handler):
    def __init__(self, webhook_url: str, never_suppress : int = logging.INFO + 1):
        self.consumer = DiscordLogConsumer(
            webhook_url,
            self.discard_filter,
            logging.getLogger("discord_webhook_log_handler")
        )
        self.never_suppress = never_suppress
        self.consumer.start()
        super().__init__()

    def discard_filter(self, pending_record : PendingRecord) -> bool:
        if pending_record.level > self.never_suppress:
            return False
        return datetime.now() - pending_record.timestamp > RECORD_EXPIRY


    @override
    def emit(self, record : LogRecord):
        if self.consumer.soft_kill_switch.is_set() or record.thread == self.consumer.thread.ident:
            return
        msg = self.format(record)
        self.consumer.pending.put(PendingRecord(msg, record.levelno, datetime.now()), block=False)

    @override
    def close(self):
        if self.consumer.soft_kill_switch.is_set():
            self.consumer.kill()
        else:
            self.consumer.soft_kill()
        self.consumer.join(CLOSE_TIMEOUT.total_seconds())
        self.consumer.kill()
        super().close()

def add_handler(logger : logging.Logger):
    if len(core.init.options.discord_webhook_url.strip()) == 0:
        logger.warning("Discord webhook logging is disabled: "
                       "No webhook url specified. "
                       "Run with --help for how to enable discord logging.")
        return
    formatter = DiscordFormatter(DISCORD_FORMAT, core.init.options.discord_level_map)

    handler = DiscordWebhookLogHandler(core.init.options.discord_webhook_url)
    handler.setFormatter(formatter)
    handler.consumer.pending.put(PendingRecord("# LOG BREAK", logging.CRITICAL, datetime.now()), block=False)
    handler.setLevel(max(logging.INFO, logger.getEffectiveLevel()))
    logger.addHandler(handler)