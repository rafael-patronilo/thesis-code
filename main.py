import logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("main")

import torch
import discord_webhook


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Torch device: {device}")
discord_webhook.notify("Torch device: " + str(device))
