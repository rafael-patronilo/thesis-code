import sys
from core.datasets import SplitDataset, get_dataset
from core import ModelFileManager, Trainer

TRAINER_PREFIX = 'from_trainer:'

def main():
    dataset_name : str = sys.argv[0]
    dataset : SplitDataset
    if dataset_name.startswith(TRAINER_PREFIX):
        dataset_name = dataset_name[len(TRAINER_PREFIX):]
        with ModelFileManager(path=dataset_name) as file_manager:
            config = file_manager.load_config()
            trainer = Trainer.from_config(config)
            training_set = trainer.training_set

            if isinstance(training_set, SplitDataset):
                dataset = training_set
            elif hasattr(training_set, 'dataset'):
                dataset = training_set.dataset #type: ignore
            else:
                raise ValueError(f"Cannot get dataset from {sys.argv[0]}")
    else:
        dataset = get_dataset(dataset_name)

        
    raise NotImplementedError()