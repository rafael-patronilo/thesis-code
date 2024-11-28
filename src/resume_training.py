#!/usr/bin/env python
import command_base
from core import Trainer
from core.storage_management import ModelFileManager
import sys
from core.stop_criteria import StopAtEpoch, EarlyStop
from core.checkpoint_triggers import BestMetric


@command_base.main_wrapper
def main():
    # Resume training from the last checkpoint
    model_path = sys.argv[1]

    with ModelFileManager(path=model_path) as file_manager:
        # Load trainer
        trainer = Trainer.load_checkpoint(file_manager)
        
        checkpoint_triggers = [
            BestMetric()
        ]
        trainer.checkpoint_triggers += checkpoint_triggers

        stop_criteria = [
            EarlyStop()
        ]
        trainer.train_until(stop_criteria)

if __name__ == '__main__':
    main()
