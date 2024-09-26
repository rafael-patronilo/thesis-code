

import threading

import logging
import log_setup
log_setup.setup_logging()
logger = logging.getLogger()
graceful_exit = False
def signal_handler(sig, frame):
    if graceful_exit:
        logger.error(f"Received signal {sig} during graceful exit, forcing exit")
        sys.exit(-1)
    logger.warning(f"Received signal {sig}, attempting graceful exit")
    exit_gracefully(0)

def exit_gracefully(code : int = 0):
    global graceful_exit
    if graceful_exit:
        logger.warning("Already exiting gracefully, ignoring call")
        return
    graceful_exit = True
    logger.info("Exitting gracefully")
    logging.shutdown()
    sys.exit(code)

import signal

signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)
logger.info("Signal handlers set")

import functools

import sys

try:
    logger.info("Importing torch")
    import torch
except BaseException as e:
    logger.exception("Failed to import torch: %s", e)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_default_device(device)
logger.info("Using torch device: %s", device)



#register datasets
try:
    import datasets
except BaseException as e:
    logger.exception("Failed to register datasets: %s", e)
    exit_gracefully(-1)

def main_thread_wrapper(main_function):
    @functools.wraps(main_function)
    def wrapper(*args, **kwargs):
        logger.info("Loading complete, initializing main thread")
        try:
            main_function(*args, **kwargs)
        except BaseException as e:
            logger.exception("Uncaught exception in main thread: %s", e)
            exit_gracefully(-1)
        exit_gracefully(0)
        

    return wrapper

if __name__ == "__main__":
    logger.info("Loading complete.")
    logger.warning("""Running main.py currently does nothing except loading.
Choose one of the other scripts.""")
    exit_gracefully(0)