import logging
import os
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger()

#import torch
import discord_webhook

logger.addHandler(discord_webhook.DiscordWebhookHandler(webhook_url=os.getenv("DISCORD_WEBHOOK_URL")))
#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = 'cpu'
logger.info(f"Torch device: {device}")
