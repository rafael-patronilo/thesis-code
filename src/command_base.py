

# set up logging
import logging
import logging_setup
logging_setup.setup_logging()
logger = logging.getLogger()
import functools
import sys
import os

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

# preferred torch device
PREFERRED_DEVICE = os.getenv("DEVICE", "cuda")
force_device = False
if PREFERRED_DEVICE.startswith("force "):
    PREFERRED_DEVICE = PREFERRED_DEVICE[len("force "):]
    force_device = True

#import torch and set device
try:
    logger.info("Importing torch and setting device")
    import torch
    device = torch.device("cuda" if torch.cuda.is_available() and PREFERRED_DEVICE=="cuda" else "cpu")
    if force_device and device.type != PREFERRED_DEVICE:
        logger.error("Failed to force device to {PREFFERED_DEVICE}")
        exit_gracefully(-1)
    torch.set_default_device(device)
    logger.info("Using torch device: %s", device)
    from collections import deque
    torch.serialization.add_safe_globals([deque])
except BaseException as e:
    logger.exception("Failed to import torch: %s", e)
    exit_gracefully(-1)


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
        except (KeyboardInterrupt, SystemExit):
            logger.info("Interrupted main thread, exiting gracefully")
            exit_gracefully(0)
        except BaseException as e:
            logger.exception("Uncaught exception in main thread: %s", e)
            exit_gracefully(-1)
        if not graceful_exit:
            logger.info("Main thread finished, exiting gracefully")
            exit_gracefully(0)
    return wrapper

if __name__ == "__main__":
    logger.info("Loading complete.")
    logger.warning("""Running this does nothing except loading. Choose one of the other commands.""")
    exit_gracefully(0)