import logging
from utils import discord_webhook
import os

LOGGING_LEVEL = logging.INFO

NOTIFY = logging.INFO + 1

def setup_logging():
    logging.addLevelName(NOTIFY, "NOTIFY")
    
    discord_webhook_url = os.getenv("DISCORD_WEBHOOK_URL")
    
    logging.basicConfig(level=LOGGING_LEVEL, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger()

    # setup discord based logging
    if discord_webhook_url is not None:
        discord_webhook_handler = discord_webhook.DiscordWebhookHandler(
            webhook_url=discord_webhook_url, 
            mention_everyone_levels=[NOTIFY]
        )
        discord_webhook_handler.setLevel(logging.INFO)
        logger.addHandler(discord_webhook_handler)
    else:
        logger.warn("Discord webhook url not specified, discord logging disabled")