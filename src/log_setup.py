import logging
import datetime
from utils import discord_webhook
import os
from pathlib import Path
from copy import deepcopy, copy
import warnings as python_warnings
from io import TextIOWrapper
import sys
import time

LOG_LEVEL = logging.getLevelName(os.getenv("LOG_LEVEL") or 'INFO')
LOG_DIR =  os.getenv("LOG_DIR") or "logs"
FORMAT = os.getenv("LOG_FORMAT") or '%(asctime)s [%(levelname)s] (%(name)s|%(threadName)s) %(message)s'
COLOR_FORMAT = os.getenv("LOG_COLOR_FORMAT") or '\033[34m%(asctime)s\033[0m [%(levelcolor)s%(levelname)s\033[0m] (%(name)s|%(threadName)s) %(message)s'

NOTIFY = logging.INFO + 1


class AnsiColorStreamHandler(logging.StreamHandler):
    DEFAULT_COLORS = {
        logging.CRITICAL: "\033[1;31m",
        logging.ERROR: "\033[1;31m",
        logging.WARNING: "\033[1;33m",
        logging.INFO: "\033[1;32m",
        logging.DEBUG: "\033[1;30m",
        NOTIFY: "\033[1;34m"
    }
    #RESET = "\033[0m"

    def __init__(self, stream = None, colors = None):
        super().__init__(stream=stream)
        self.colors = deepcopy(self.DEFAULT_COLORS)
        if colors is not None:
            self.colors.update(colors)
    
    def emit(self, record):
        try:
            record = copy(record)
            record.levelcolor = self.colors[record.levelno]
            super().emit(record)
        except Exception:
            self.handleError(record)

class MultiLineFormatter(logging.Formatter):
    def __init__(self, formatter, prefix = "\t"):
        self.formatter = formatter
        self.prefix = prefix
    
    def format(self, record):
        message = self.formatter.format(record)
        return message.replace("\n", "\n" + self.prefix)

class StreamInterceptor(TextIOWrapper):
    WARNING_COOLDOWN = 5

    def __init__(self, stream, stream_name : str, logger : logging.Logger, level):
        self.logger = logger
        self.level = level
        self.stream_name = stream_name
        self.last_warning = None
        super().__init__(stream)

    def write(self, message):
        now = time.time()
        if self.last_warning is None or now - self.last_warning >= self.WARNING_COOLDOWN:
            self.logger.warning(
                f"Intercepted unexpected write to {self.stream_name}, please replace with proper logging\n"
                f"This warning will be supressed for {self.WARNING_COOLDOWN} seconds",
                exc_info=True
            )
            self.last_warning = now
        self.logger.log(self.level, message)

python_warnings_logger = logging.getLogger("warnings.py")

def showwarning_hook(message, category, filename, lineno, file=None, line=None):
    text = python_warnings.formatwarning(message, category, filename, lineno, line)
    python_warnings_logger.warning(text)

def setup_logging():
    logging.addLevelName(NOTIFY, "NOTIFY")
    formatter=MultiLineFormatter(logging.Formatter(FORMAT))
    logger = logging.getLogger()
    logger.setLevel(os.getenv("LOG_LEVEL") or LOG_LEVEL)
    
    # intercept unexpected writes to stdout and stderr
    true_stdout = sys.stdout
    true_stderr = sys.stderr

    sys.stdout = StreamInterceptor(true_stdout, "stdout", logging.getLogger("stdout"), logging.INFO)
    sys.stderr = StreamInterceptor(true_stderr, "stderr", logging.getLogger("stderr"), logging.ERROR)

    # Setup console logging
    console_handler = AnsiColorStreamHandler(stream=true_stdout)
    console_handler.setFormatter(MultiLineFormatter(logging.Formatter(COLOR_FORMAT)))
    logger.addHandler(console_handler)
    
    # Setup file logging
    log_dir = Path(os.getenv("LOG_DIR") or LOG_DIR)
    log_dir.mkdir(parents=True, exist_ok=True)
    logfile = log_dir.joinpath(datetime.datetime.now().isoformat()).with_suffix(".log")
    file_handler = logging.FileHandler(logfile)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # setup discord webhook logging (if available)
    discord_webhook_url = os.getenv("DISCORD_WEBHOOK_URL")
    if discord_webhook_url is not None:
        discord_webhook_handler = discord_webhook.DiscordWebhookHandler(
            webhook_url=discord_webhook_url,
            mention_everyone_levels=[NOTIFY]
        )
        discord_webhook_handler.setFormatter(formatter)
        discord_webhook_handler.setLevel(max(logging.INFO, LOG_LEVEL))
        logger.addHandler(discord_webhook_handler)
    else:
        logger.warning("Discord webhook url not specified, discord logging disabled")

    # setup warnings logging
    python_warnings.showwarning = showwarning_hook


    
    logger.info(
f"""Start of logging
Time: {datetime.datetime.now().astimezone().isoformat()}
Level: {LOG_LEVEL if type(LOG_LEVEL) == str else logging.getLevelName(LOG_LEVEL)}
Active Handlers: {", ".join(type(handler).__name__ for handler in logger.handlers)}
Log file: {logfile}"""
)
    for handler in logger.handlers:
        handler.flush()