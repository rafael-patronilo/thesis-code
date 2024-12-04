"""
This module handles discord notifications.
It uses discord webhooks (http api is handled directly, no extra libraries required). 
"""
from contextlib import suppress
import os
import http.client
import logging
import json
import queue
import threading
from typing import NamedTuple
from urllib.parse import urlparse

logger = logging.getLogger(__name__)

MSG_MAX_LENGTH = 2000

class DiscordMessageBufferer:
    """
    Utility class to buffer messages and split them into multiple messages if they exceed the maximum length.
    """
    Block = NamedTuple("Block", [("header", str), ("footer", str)])
    CODE_BLOCK = Block("```\n", "```")
    DIFF_BLOCK = Block("```diff\n", "```")
    EMPTY_BLOCK = Block("", "")

    def __init__(self) -> None:
        self.lock = threading.RLock()
        self.buffer = queue.Queue()
        self.message = ""
        self.extras = {}
        self.block = self.EMPTY_BLOCK
        self.prefix = ""
        self.closed = False
    
    def __append_buffering(self, text: str):
        with self.lock:
            if len(self.message) + len(text) + len(self.block.footer) > MSG_MAX_LENGTH:
                self.break_msg()
            msg_split = False
            while len(self.message) + len(text) + len(self.block.footer) > MSG_MAX_LENGTH:
                msg_split = True
                excess = len(self.message) + len(text) + len(self.block.footer) - MSG_MAX_LENGTH
                self.message += text[:-excess]
                self.break_msg()
                if text[-excess-1] == "\n":
                    text = text[-excess:]
                else:
                    text = self.prefix + text[-excess:]
            self.message += text
            if msg_split:
                self.break_msg()

    def _assert_not_closed(self):
        if self.closed:
            raise ValueError("Buffer is closed.")
    
    def append(self, text: str):
        with self.lock:
            self._assert_not_closed()
            if text.endswith("\n"):
                text = text[:-1].replace("\n", "\n" +  self.prefix) + "\n"
            else:
                text = text.replace("\n", "\n" +  self.prefix)
            if self.message.endswith("\n"):
                text = self.prefix + text
            self.__append_buffering(text)
    
    def appendln(self, text: str):
        self.append(text + "\n")
    
    def get_queue(self):
        return self.buffer
    
    def begin_code_block(self):
        self.begin_block(self.CODE_BLOCK)
        
    def end_code_block(self):
        with self.lock:
            if self.block != self.CODE_BLOCK:
                raise ValueError("Code block not opened.")
            self.end_block()

    def begin_diff_block(self):
        self.begin_block(self.DIFF_BLOCK)
        
    def end_diff_block(self):
        with self.lock:
            if not self.block == self.DIFF_BLOCK:
                raise ValueError("Diff block not opened.")
            self.end_block()
    
    def begin_block(self, block):
        with self.lock:
            self._assert_not_closed()
            if self.block != self.EMPTY_BLOCK and block != self.EMPTY_BLOCK:
                raise ValueError("Block already opened.")
            self.block = block
            self.append(block.header)
    
    def end_block(self):
        with self.lock:
            self._assert_not_closed()
            self.append(self.block.footer)
            block = self.block
            self.block = self.EMPTY_BLOCK
            return block
        
    def break_msg(self):
        with self.lock:
            self._assert_not_closed()
            if self.message != "" and self.message != self.block.header:
                self.message += self.block.footer
                payload = {'content':self.message}
                payload.update(self.extras)
                self.buffer.put(payload)
                self.message = self.block.header
                self.extras = {}
    
    def begin_prefix(self, prefix: str):
        with self.lock:
            if self.prefix != "":
                raise ValueError("Prefix already set.")
            self.prefix = prefix
        
    def end_prefix(self):
        with self.lock:
            prefix = self.prefix
            self.prefix = ""
            return prefix
    
    def mention_everyone(self):
        block = self.end_block()
        prefix = self.end_prefix()
        self.append("@everyone\n")
        self.break_msg()
        self.begin_block(block)
        self.begin_prefix(prefix)

    def close(self):
        with self.lock:
            self.buffer.put(None)


class DiscordWebhookHandler(logging.Handler):
    ERROR_COOLDOWN = 60
    CONSUMER_COOLDOWN = 0.05
    """
    A logging handler that sends logs to a discord webhook.
    """
    def __init__(self, 
                webhook_url: str,
                mention_everyone_min_level = logging.ERROR,
                mention_everyone_levels = None,
                buffer_flush_interval = 20
                ):
        self.webhook_url = webhook_url
        parsed_url = urlparse(webhook_url)
        self.client = http.client.HTTPSConnection(parsed_url.netloc)
        self.webhook_path = parsed_url.path
        self.message_buffer = DiscordMessageBufferer()
        self.message_buffer.begin_diff_block()
        self.buffer_flush_interval = buffer_flush_interval
        self.mention_everyone_min_level = mention_everyone_min_level
        self.mention_everyone_levels = mention_everyone_levels or []
        self.__kill_switch = threading.Event()
        self.__buffer_consumer_thread = threading.Thread(
            target=self._buffer_consumer, 
            name='discord_webhook_log_handler')
        super().__init__()
        if self._try_send_message({"content": "# LOG BREAK"}):
            self.__kill_switch.wait(self.CONSUMER_COOLDOWN)
            self.__buffer_consumer_thread.start()
        else:
            logger.error("Failed to send initial message to discord webhook. Discord logging will be disabled.")
    
    def _try_send_message(self, payload) -> bool:
        try:
            try:
                self.client.connect()
                self.client.request("POST", self.webhook_path, json.dumps(payload),
                        {"Content-Type": "application/json"})
                response = self.client.getresponse()
                if response.status != 204:
                    raise Exception(f"Unexpected response code: {response.status}\n{response.msg}\n")
                return True
            finally:
                response.read()
                response.close()
        except Exception as e:
            logger.warning( 
                    "Failed to send message to discord webhook\n"
                    f"Request payload: {payload}",
                    exc_info=True
                )
            return False
        
    def _buffer_consumer(self):
        msg_queue = self.message_buffer.get_queue()
        try:
            while not self.__kill_switch.is_set():
                try:
                    payload = msg_queue.get(timeout=self.buffer_flush_interval)
                    if payload is None:
                        return
                    if self._try_send_message(payload):
                        self.__kill_switch.wait(self.CONSUMER_COOLDOWN)
                    else:
                        logger.warning(f"Disabling discord logging for {self.ERROR_COOLDOWN} seconds")
                        self.__kill_switch.wait(self.ERROR_COOLDOWN)
                except queue.Empty:
                    self.message_buffer.break_msg()
        finally:
            # Try to flush remaining messages with no cooldown
            with suppress(queue.Empty):
                with self.message_buffer.lock:
                    self.message_buffer.break_msg()
                    self.__kill_switch.set()
                    while msg_queue.not_empty:
                        payload = msg_queue.get(block=False)
                        if payload is None or not self._try_send_message(payload):
                            break

    def flush(self) -> None:
        self.message_buffer.break_msg()
    
    def emit(self, record):
        if self.__kill_switch.is_set() or record.threadName == self.__buffer_consumer_thread.getName():
            return
        with self.message_buffer.lock:
            if self.__kill_switch.is_set():
                return
            log_entry = self.format(record)
            if record.levelno >= logging.WARNING:
                self.message_buffer.begin_prefix("- ")
            else:
                self.message_buffer.begin_prefix("+ ")
            self.message_buffer.appendln(log_entry)
            self.message_buffer.end_prefix()
            if (
                record.levelno >= self.mention_everyone_min_level
                or record.levelno in self.mention_everyone_levels
            ):
                self.message_buffer.mention_everyone()
    
    def close(self):
        with self.message_buffer.lock:
            self.__kill_switch.set()
            self.message_buffer.close()
        if self.__buffer_consumer_thread.is_alive():
            self.__buffer_consumer_thread.join(timeout=10)
        self.client.close()
        super().close()