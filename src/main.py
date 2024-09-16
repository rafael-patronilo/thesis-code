

import threading
#finished = threading.Event()

import logging
import log_setup
log_setup.setup_logging()
logger = logging.getLogger()



logger.info("Importing torch")
import torch

import datasets # register datasets
import functools

import signal
import sys


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info("Using torch device: %s", device)

def signal_handler(sig, frame):
    logger.warning("Received signal %s, attempting graceful exit", sig)
    exit_gracefully(0)

def exit_gracefully(code : int = 0):
    logger.info("Exitting gracefully")
    #finished.set()
    sys.exit(code)

signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)
logger.info("Signal handlers set")

def main_thread_wrapper(main_function):
    @functools.wraps(main_function)
    def wrapper(*args, **kwargs):
        try:
            main_function(*args, **kwargs)
        except BaseException as e:
            logger.exception("Uncaught exception in main thread: %s", e)
            exit_gracefully(-1)
        exit_gracefully(0)
        

    return wrapper