#!/usr/bin/env python
from pathlib import Path
from typing import Literal
from core.training import Trainer
from core.storage_management import ModelFileManager
from core.init.options_parsing import option, positional
from dataclasses import dataclass, field
from core.util.strings import multiline_repr
import logging

from src.core.training.stop_criteria import early_stop
logger = logging.getLogger(__name__)

@dataclass
class Options:
    model: Path = field(
        metadata=positional(Path, help_="The model to load")
    )

    preferred_checkpoint : str = field(default='best',
                                       metadata=option(str, help_=
                                       "Either 'best' or 'last'. Defaults to 'best'. "
                                       "Specifies which checkpoint to prefer "
                                       "during checkpoint discovery. "
                                       "If the checkpoint option is specified, "
                                       "this option is ignored.")
                                       )

    checkpoint : Path | None = field(default=None,
                                metadata=option(Path, help_=
                                "The checkpoint file to load. If not specified, "
                                "the script will automatically decide which checkpoint to load, "
                                "in accordance to the preferred_checkpoint option."))

    num_epochs : int | None = field(default=None,
                                metadata=option(int, help_="Number of epochs to train for"))

    end_epoch : int | None = field(default=None,
                                metadata=option(int, help_="End epoch for training"))

    clear_stop_criteria : bool = field(default=False,
                                metadata=option(bool, help_="Clear stop criteria defined by the build script"))

    early_stop : int | None = field(default=None,
                                metadata=option(int, help_="If specified, will use EarlyStop with the given patience"))

    def __repr__(self):
        return multiline_repr(self)



def main(options : Options):
    """
    Resume training from the last checkpoint
    """
    with ModelFileManager(path=options.model) as file_manager:
        # Load trainer
        assert options.preferred_checkpoint in ['best', 'last']
        preferred_checkpoint: Literal['best', 'last']
        preferred_checkpoint = options.preferred_checkpoint # type: ignore
        trainer = Trainer.load_checkpoint(file_manager, options.checkpoint, preferred_checkpoint)
        if options.clear_stop_criteria:
            trainer.stop_criteria = []
        if sum(1 if x is not None else 0 for x in [options.num_epochs, options.end_epoch]) > 1:
            logger.warning("At most one of num_epochs and end_epoch should be specified")
        if options.early_stop is not None:
            objective = trainer.objective
            if objective is None:
                raise ValueError("Trainer has no objective, cannot use early stop")
            trainer.stop_criteria.append(early_stop.EarlyStop(objective, options.early_stop))

        if options.num_epochs is not None:
            trainer.train_epochs(options.num_epochs)
        elif options.end_epoch is not None:
            trainer.train_until_epoch(options.end_epoch)
        else:
            trainer.train_indefinitely()