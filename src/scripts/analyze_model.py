import sys
from core import Trainer, ModelFileManager
import torch
import logging
logger = logging.getLogger(__name__)

def analyze_weights(trainer: Trainer):
    logger.info("Analyzing model weights...")
    for name, param in trainer.model.named_parameters():
        logger.info(
            f"{name}:\n"
            f"\tMax: {param.max()}\n"
            f"\tMin: {param.min()}\n"
            f"\tMean: {param.mean()}\n"
            f"\tStd: {param.std()}"
        )


def main():
    path = sys.argv[1]
    with ModelFileManager(path) as file_manager:
        trainer = Trainer.load_checkpoint(file_manager)
        trainer.model.eval()
        with torch.no_grad():
            analyze_weights(trainer)