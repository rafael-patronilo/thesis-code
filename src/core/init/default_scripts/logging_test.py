import logging
from core.logging import NOTIFY
logger = logging.getLogger()

def main():
    """
    Tests each log level.
    """
    logger.debug("DEBUG Message")
    logger.info("INFO Message")
    logger.log(NOTIFY, "NOTIFY Message")
    logger.warning("WARNING Message")
    logger.error("ERROR Message")
    logger.critical("CRITICAL Message")
    logger.info(f"Long message:\n{'a'*4000}")
