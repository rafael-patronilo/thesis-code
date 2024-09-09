
import logging
import log_setup
log_setup.setup_logging()
logger = logging.getLogger()

logger.info("Importing torch")
import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info("Using torch device: %s", device)