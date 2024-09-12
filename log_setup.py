import logging
import datetime
from utils import discord_webhook
import os
from pathlib import Path

LOG_LEVEL = logging.INFO
LOG_DIR = "logs"
FORMAT = '%(asctime)s [%(levelname)s] (%(name)s|%(threadName)s) %(message)s'

NOTIFY = logging.INFO + 1

def setup_logging():
    logging.addLevelName(NOTIFY, "NOTIFY")
    formatter=logging.Formatter(FORMAT)
    logger = logging.getLogger()
    logger.setLevel(os.getenv("LOG_LEVEL") or LOG_LEVEL)
    
    # Setup console logging
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
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
        logger.addHandler(discord_webhook_handler)
    else:
        logger.warning("Discord webhook url not specified, discord logging disabled")
    
    logger.info(
f"""Start of logging
\tTime: {datetime.datetime.now().astimezone().isoformat()}
\tLevel: {logging.getLevelName(LOG_LEVEL)}
\tHandlers: {", ".join(type(handler).__name__ for handler in logger.handlers)}"""
)
    for handler in logger.handlers:
        handler.flush()