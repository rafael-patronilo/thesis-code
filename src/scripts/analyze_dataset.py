import sys
from core.datasets import SplitDataset, get_dataset
from core import ModelFileManager, Trainer

TRAINER_PREFIX = 'from_trainer:'

def main():
    dataset_name : str = sys.argv[0]
    if dataset_name.startswith(TRAINER_PREFIX):
        dataset_name = dataset_name[len(TRAINER_PREFIX):]
        with ModelFileManager(path=dataset_name) as file_manager:
            config = file_manager.load_config()
            trainer = Trainer.from_config(config)
            # = trainer.
    else:
        dataset = get_dataset(dataset_name)
    raise NotImplementedError()