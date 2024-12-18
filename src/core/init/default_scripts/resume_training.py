#!/usr/bin/env python
from pathlib import Path
from core.training import Trainer
from core.storage_management import ModelFileManager
from core.init.options_parsing import option, positional
from dataclasses import dataclass, field
from core.util.strings import multiline_repr
from core.init import options as global_options
import logging
logger = logging.getLogger(__name__)

@dataclass
class Options:
    model: Path = field(
        metadata=positional(Path, help_="The model to load")
    )

    num_epochs : int | None = field(default=None,
                                metadata=option(int, help_="Number of epochs to train for"))

    end_epoch : int | None = field(default=None,
                                metadata=option(int, help_="End epoch for training"))

    def __repr__(self):
        return multiline_repr(self)


def main(options : Options):
    """
    Resume training from the last checkpoint
    """
    # TODO rework script?
    with ModelFileManager(path=options.model) as file_manager:
        # Load trainer
        trainer = Trainer.load_checkpoint(file_manager)
        if options.num_epochs is not None:
            if options.end_epoch is not None:
                logger.warning("Both num_epochs and end_epoch specified, ignoring end_epoch")
            trainer.train_epochs(options.num_epochs)
        elif options.end_epoch is not None:
            trainer.train_until_epoch(options.end_epoch)
        else:
            trainer.train_indefinitely()