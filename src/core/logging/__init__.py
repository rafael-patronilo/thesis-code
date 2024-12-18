"""
Logging configuration and utilities
"""
import logging
import datetime
import os
from pathlib import Path
import sys
from datetime import datetime
import subprocess
import core.init

from .formatting_utils import MultiLineFormatter
from .stream_interceptor import StreamInterceptor
from .handlers import discord_webhook_handler, stdout_log_handler
from typing import Literal

NOTIFY = logging.INFO + 5
logging.addLevelName(NOTIFY, "NOTIFY")

log_file : Path | None = None

true_stderr = sys.stderr
true_stdout = sys.stdout

__all__=[
    "log_file",
    "NOTIFY",
    "log_break",
    "log_version_info",
    "setup"
]


FORMAT = '%(asctime)s [%(levelname)s] (%(name)s|%(threadName)s) %(message)s' # noqa

def log_break(msg = "LOG BREAK"):
    """
    Logs a break in stdout
    :param msg:
    :return:
    """
    try:
        cols, lines = os.get_terminal_size()
        half_width = (cols - len(msg)) // 2
        print(f"\n\n\033[34m{'='*half_width}{msg}{'='*half_width}\033[0m\n\n", file=true_stdout)
    except OSError:
        pass # ignore if terminal size cannot be determined

def log_version_info(logger, path = Path("version.txt")):
    """
    Log version info
    :param logger: the logger to use
    :param path: file that contains version info
    :return:
    """
    sb = ["Version Info"]
    if path.exists():
        sb.append(path.read_text())
    else:
        try:
            status = subprocess.run(
                ["git", "status"],
                capture_output=True,
                text=True
            )
            commit_info = subprocess.run(
                ["git", "log", "-1"],
                capture_output=True,
                text=True
            )
            sb.append(f"Git status:\n{status.stdout}")
            sb.append(f"Last commit:\n{commit_info.stdout}")
        except BaseException as e:
            logger.warning(f"Failed to get git info: {e}")
            return
    logger.info("\n".join(sb))

def trap_stdout():
    global true_stdout
    sys.stdout = StreamInterceptor(true_stdout, "stdout",
                                   logging.getLogger("stdout"), logging.INFO)

def trap_stderr():
    global true_stderr
    sys.stderr = StreamInterceptor(true_stderr, "stderr",
                                   logging.getLogger("stderr"), logging.ERROR)


def format_log_file_name(timespec : Literal['seconds', 'microseconds'] = 'seconds') -> str:
    file_name = core.init.start_time.isoformat(timespec=timespec).replace(':', '_')
    if timespec == 'microseconds':
        file_name = file_name.replace('.', '_')
    return file_name


def setup(version_info : bool = True, stdout_only : bool = False):
    """
    Setup logging for the application
    :param version_info: Whether to print version info (default True)
    :param stdout_only: Whether to log only to stdout (default False)
    :return:
    """
    global log_file
    formatter = MultiLineFormatter(logging.Formatter(FORMAT))
    logger = logging.getLogger()
    logger.setLevel(core.init.options.log_level)
    
    # intercept unexpected writes to stdout and stderr
    trap_stdout()
    trap_stderr()

    # Setup console logging
    stdout_log_handler.add_handler(logger, true_stdout)

    if stdout_only:
        logger.warning("Only logging to stdout; File and external logging disabled")
    else:
        # Setup file logging
        log_dir = core.init.options.log_dir
        log_dir.mkdir(parents=True, exist_ok=True)
        log_file = log_dir.joinpath(format_log_file_name()).with_suffix(".log")
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

        # Setup discord logging
        discord_webhook_handler.add_handler(logger)

    # setup warnings logging
    logging.captureWarnings(True)

    log_break()
    logger.info(
f"""Start of logging
Time: {datetime.now().astimezone().isoformat()}
Level: {core.init.options.log_level}
Active Handlers: {", ".join(type(handler).__name__ for handler in logger.handlers)}
Log file: {log_file}"""
)
    if version_info:
        log_version_info(logger)
    for handler in logger.handlers:
        handler.flush()
