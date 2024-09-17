import main
from core import Trainer, ModelFileManager
import sys
from core.util.stop_criteria import StopAtEpoch, EarlyStop
from core.util.checkpoint_triggers import BestMetric


@main.main_thread_wrapper
def main_function():
    # Resume training from the last checkpoint
    model_path = sys.argv[1]
    parts = model_path.split("_")
    model_name = "_".join(parts[:-1])
    model_identifier = parts[-1]

    with ModelFileManager(
        model_name=model_name, 
        model_identifier=model_identifier, 
        conflict_strategy='load') as file_manager:
        # Load trainer
        trainer = Trainer.load_checkpoint(file_manager, device=main.device)
        
        checkpoint_triggers = [
            BestMetric()
        ]
        trainer.checkpoint_triggers += checkpoint_triggers

        stop_criteria = [
            EarlyStop()
        ]
        trainer.train_until(stop_criteria)

if __name__ == '__main__':
    main_function()
