
from core.init.options_parsing import option
from dataclasses import dataclass, field

from core.init import DO_SCRIPT_IMPORTS
from typing import TYPE_CHECKING
if TYPE_CHECKING or DO_SCRIPT_IMPORTS:
    import logging
    from core.logging import NOTIFY
    logger = logging.getLogger()

@dataclass
class Options:
    spam : bool = field(
        default=False,
        metadata=option(bool, help_="Activates a much more verbose logging test")
    )
    spam_count : int = field(
        default=100,
        metadata=option(int, help_="Number of spam messages to log")
    )


def main(options : Options):
    """
    Tests each log level.
    """
    logger.debug("DEBUG Message")
    logger.info("INFO Message")
    logger.log(NOTIFY, "NOTIFY Message")
    logger.warning("WARNING Message")
    logger.error("ERROR Message")
    logger.critical("CRITICAL Message")
    if options.spam:
        spam = 'a'*4000
        for i in range(options.spam_count):
            logger.info(f"Spam message {i}:\n{spam}")

