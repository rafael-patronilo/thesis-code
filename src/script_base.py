

# set up logging
import logging
import logging_setup
logging_setup.setup_logging()
logger = logging.getLogger()

graceful_exit = False
def exit_gracefully(code : int = 0, impatient = False):
    global graceful_exit
    if graceful_exit:
        if impatient:
            logger.error("Forcefully exiting while already exiting gracefully")
            sys.exit(-1)
        else:
            logger.debug("Already exiting gracefully, ignoring call")
            return
    graceful_exit = True
    logger.info("Exitting gracefully")
    logging.shutdown()
    logging_setup.log_break(msg="END LOG")
    sys.exit(code)

# trap interrup signals for graceful exit
def signal_handler(sig, frame):
    logger.warning(f"Received signal {sig}, attempting graceful exit")
    exit_gracefully(0, impatient=True)

import signal
signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)
logger.info("Signal handlers set")




import functools
import sys
import os

#import torch and set device
try:
    logger.info("Importing torch")
    import torch
except BaseException as e:
    logger.exception("Failed to import torch: %s", e)
    exit_gracefully(-1)
PREFERRED_DEVICE = os.getenv("DEVICE", "cuda")
force_device = False
if PREFERRED_DEVICE.startswith("force "):
    PREFERRED_DEVICE = PREFERRED_DEVICE[len("force "):]
    force_device = True
device = torch.device("cuda" if torch.cuda.is_available() and PREFERRED_DEVICE=="cuda" else "cpu")
if force_device and device.type != PREFERRED_DEVICE:
    logger.error("Failed to force device to {PREFFERED_DEVICE}")
    exit_gracefully(-1)
torch.set_default_device(device)
logger.info("Using torch device: %s", device)



#register datasets
try:
    logger.info("Registering datasets")
    import datasets
except BaseException as e:
    logger.exception("Failed to register datasets: %s", e)
    exit_gracefully(-1)

def main_wrapper(main_function):
    @functools.wraps(main_function)
    def wrapper(*args, **kwargs):
        logger.info("Loading complete, initializing main thread")
        try:
            main_function(*args, **kwargs)
        except BaseException as e:
            logger.exception("Uncaught exception in main thread: %s", e)
            exit_gracefully(-1)
        if not graceful_exit:
            logger.info("Main thread finished, exiting gracefully")
            exit_gracefully(0)
        

    return wrapper

if __name__ == "__main__":
    logger.info("Loading complete.")
    logger.warning("""Running script_base.py does nothing except loading.
Choose one of the other scripts.""")
    exit_gracefully(0)