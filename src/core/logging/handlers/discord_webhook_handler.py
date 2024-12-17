import http.client
import queue
import threading
from copy import copy
from urllib.parse import urlparse
import logging
from logging import LogRecord
from typing import Callable, NamedTuple, override
import json
from queue import Queue
from datetime import datetime, timedelta
from core.init import options


DISCORD_FORMAT = '-# %(asctime)s [%(levelname)s] (%(name)s|%(threadName)s)\n```%(message)s```' # noqa

MESSAGE_MAX_LENGTH = 2000
LOG_RECORD_MAX_LENGTH = 1500
RECORD_EXPIRY = timedelta(minutes=1)
ERROR_TIMEOUT = timedelta(minutes=5)
MESSAGE_COOLDOWN = timedelta(seconds=5)

class DiscordWebhookClient:
    def __init__(self, webhook_url: str) -> None:
        self.webhook_url = webhook_url
        parsed_url = urlparse(webhook_url)
        self.connection =  http.client.HTTPSConnection(parsed_url.netloc)
        self.webhook_path = parsed_url.path

    def _make_request(self, path : str, method: str, payload: str | None):
        self.connection.connect()
        self.connection.request(method, path, payload)
        response = self.connection.getresponse()
        try:
            if not response.status >= 200 and response.status < 300:
                raise Exception(f"Unexpected response code: {response.status}\n{response.msg}\n")
        finally:
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

class DiscordConsumerThread(threading.Thread):
    def __init__(self,
                 webhook_url : str,
                 discard_filter : Callable[[PendingRecord], bool],
                 logger : logging.Logger):
        self.pending : Queue[PendingRecord | None] = Queue()
        self.buffered_record : PendingRecord | None = None
        self.client = DiscordWebhookClient(webhook_url)
        self.discard_filter = discard_filter
        self.kill_switch = threading.Event()
        self.soft_kill_switch = threading.Event()
        self.logger = logger
        super().__init__(name = self.__class__.__name__)

    def soft_kill(self):
        self.soft_kill_switch.set()
        self.pending.put(None)

    def kill(self):
        self.kill_switch.set()
        self.pending.put(None)

    def _get_discarding(self, block : bool) -> PendingRecord:
        ignored = 0
        while not self.kill_switch.is_set():
            if self.buffered_record is not None:
                record = self.buffered_record
                self.buffered_record = None
            else:
                record = self.pending.get(block)
            if record is None:
                raise StopIteration()
            if self.discard_filter(record):
                ignored += 1
                self.pending.task_done()
            else:
                if ignored > 0:
                    self.buffered_record = record
                    return PendingRecord(f"***(suppressed {ignored} log records)***", logging.CRITICAL, datetime.now())
                return record
        raise StopIteration()


    def _get_records(self) -> list[PendingRecord]:
        records = []
        size_sum = 0
        record = self._get_discarding(block=self.soft_kill_switch.is_set())
        size_sum += len(record.msg)
        try:
            while not self.kill_switch.is_set():
                record = self._get_discarding(block=False)
                if size_sum + len(record.msg) > MESSAGE_MAX_LENGTH:
                    self.buffered_record = record
                    break
                records.append(record)
                size_sum += len(record.msg)
        except queue.Empty:
            pass
        return records

    @override
    def run(self):
        while not self.kill_switch.is_set():
            try:
                records = self._get_records()
                content = "\n".join(rec.msg for rec in records)
                self.kill_switch.wait(MESSAGE_COOLDOWN.total_seconds())
                if self.kill_switch.is_set():
                    break
                self.client.send(content)
                for _ in records:
                    self.pending.task_done()
            except (queue.Empty, StopIteration):
                break # Soft kill
            except Exception as e:
                self.logger.error(f"Error sending log message to discord,"
                                  f" disabling discord logging for {ERROR_TIMEOUT}\n{e}")
                self.soft_kill_switch.wait(ERROR_TIMEOUT.total_seconds())
                pass

class DiscordFormatter(logging.Formatter):
    def __init__(self, fmt: str, level_map : dict[str, str]):
        self.level_map = level_map
        super().__init__(fmt)

    @override
    def formatMessage(self, record: LogRecord) -> str:
        record = copy(record)
        # noinspection SpellCheckingInspection
        record.levelname = self.level_map.get(record.levelname, record.levelname)
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
            if record.message[-1:] != "\n":
                record.message = record.message + "\n"
            record.message = record.message + record.exc_text
        return self.formatMessage(record)


class DiscordWebhookLogHandler(logging.Handler):
    def __init__(self, webhook_url: str, never_suppress : int = logging.INFO + 1):
        self.client = DiscordWebhookClient(webhook_url)
        self.thread = DiscordConsumerThread(
            webhook_url,
            self.discard_filter,
            logging.getLogger("discord_webhook_log_handler")
        )
        self.never_suppress = never_suppress
        self.thread.start()
        super().__init__()

    def discard_filter(self, pending_record : PendingRecord) -> bool:
        if pending_record.level > self.never_suppress:
            return False
        return datetime.now() - pending_record.timestamp > RECORD_EXPIRY


    @override
    def emit(self, record : LogRecord):
        if record.thread == self.thread.ident:
            return
        msg = self.format(record)
        self.thread.pending.put(PendingRecord(msg, record.levelno, datetime.now()))

    @override
    def close(self):
        if self.thread.soft_kill_switch.is_set():
            self.thread.kill()
        else:
            self.thread.soft_kill()
        self.thread.join()
        self.client.close()
        super().close()

def add_handler(logger : logging.Logger):
    if len(options.discord_webhook_url):
        logger.warning("Discord webhook logging is disabled: "
                       "No webhook url specified. "
                       "Run with --help for how to enable discord logging.")
        return
    formatter = DiscordFormatter(DISCORD_FORMAT, options.discord_level_map)

    handler = DiscordWebhookLogHandler(options.discord_webhook_url)
    handler.setFormatter(formatter)
    handler.thread.pending.put(PendingRecord("# LOG BREAK", logging.CRITICAL, datetime.now()))
    #handler.setLevel(max(logging.INFO, logger.getEffectiveLevel()))
    logger.addHandler(handler)