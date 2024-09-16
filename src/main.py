
import logging
import log_setup
log_setup.setup_logging()
logger = logging.getLogger()

import signal
import sys

logger.info("Importing torch")
import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info("Using torch device: %s", device)

def signal_handler(sig, frame):
    logger.warning("Received signal %s, attempting graceful exit", sig)
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)
logger.info("Signal handlers set")

