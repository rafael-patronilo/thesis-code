"""
This module handles discord notifications.
It uses discord webhooks (no extra pip packages required). 
The Webhook id and token should be specified as the environment variables DISCORD_WEBHOOK_ID and DISCORD_WEBHOOK_TOKEN.
"""
import os
import http.client
import logging
import json
import queue
import threading
import time

logger = logging.getLogger(__name__)

MESSAGE_MAX_LENGTH = 2000
CODE_HEADER = "```diff\n"
CODE_FOOTER = "```"
CODE_POSITIVE_PREFIX = "+ "
CODE_NEGATIVE_PREFIX = "- "

client = None
webhook_path = None
tasks = queue.Queue()

def __worker():
    while True:
        task = tasks.get()
        task()

worker_thread = threading.Thread(target=__worker, daemon=True)

def __connect() -> bool:
    global client, webhook_path
    if webhook_path is None:
        webhook_path = os.getenv('DISCORD_WEBHOOK_PATH')
        if webhook_path is None:
            logger.warning("DISCORD_WEBHOOK_PATH not set. Discord notifications will not be sent.")
            return False
    if client is None:
        client = http.client.HTTPSConnection("discord.com")
    if not worker_thread.is_alive():
        worker_thread.start()
    return True



class DiscordLoggerHandler(logging.Handler):
    def __init__(self, 
                level = logging.INFO, 
                mention_everyone_min_level = logging.ERROR,
                buffer_flush_interval = 300
                ):
        super().__init__()
        self.message_buffer = ""
        self.lock = threading.Lock()
        self.__buffer_flush_interval = buffer_flush_interval
        self.setLevel(level)
        self.__buffer_flush_worker_thread = threading.Thread(target=self.__buffer_flush_worker, daemon=True)
        self.__buffer_flush_worker_thread.start()
        
        
    def __buffer_flush_worker(self):
        while True:
            time.sleep(self.buffer_flush_interval)
            if len(self.message_buffer) > 0:
                self.lock.acquire()
                if len(self.message_buffer) > 0:
                    send(self.message_buffer)
                    self.message_buffer = ""
                self.lock.release()
        
    def emit(self, record):
        if not __connect():
            return
        self.lock.acquire()
        log_entry = self.format(record)
        prefix = CODE_NEGATIVE_PREFIX if record.levelno >= logging.WARNING else CODE_POSITIVE_PREFIX
        if record.levelno >= logging.WARNING:
            prefix = CODE_NEGATIVE_PREFIX
        if len(self.message_buffer) + len(log_entry) + len(CODE_HEADER) + len(CODE_FOOTER) > MESSAGE_MAX_LENGTH:
            send(self.message_buffer)
            self.message_buffer = ""
        self.message_buffer += log_entry + "\n"
        #...
        self.lock.release()

def send(message: str):
    if not __connect():
        return
    def task():
        payload = {
            "content": message
        }
        headers = {
            "Content-Type": "application/json"
        }
        client.request("POST", f"/{webhook_path}", body=json.dumps(payload), headers=headers)
        response = client.getresponse()
        if response.status != 204:
            logger.warning(f"Failed to send discord notification. Status code: {response.status}")
        else:
            logger.debug("Discord notification sent.")
        response.read()
        response.close()
    tasks.put(task)