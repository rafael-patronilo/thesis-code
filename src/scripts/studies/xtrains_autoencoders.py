from core import StudyManager, ModelDetails, datasets
from core.storage_management import StudyFileManager
from typing import NamedTuple
from torch import nn
import sys

CONVOLUTIONS = (
    [32, 32, ('pool', 2)] + 
    [64, ('pool', 2)] * 2 + 
    [128, ('pool', 2)] * 2
)

DATASET = "xtrains"

EXPERIMENTS = [
    ('L16', [16]),
    ('L32', [32]),
    ('L64', [64]),
    ('L128', [128]),
    ('2L', [64, 32]),
    ('3L', [64, 32, 16]),
    ('4L', [128, 64, 32, 16])
]

def make_config(linears):
    return dict(
        dataset_name = DATASET,
        conv_layers = CONVOLUTIONS,
        linear_layers = linears,
        encoding_size = 9,
        encoding_activation = 'sigmoid',
        kernel_size = 3
    )


def main():
    # Load study manager
    file_manager = StudyFileManager(f"xtrains_autoencoders_1")
    study_manager = StudyManager(
        file_manager,
        compare_strategy="max",
        metric_key=("val", "loss"),
        num_epochs=35
    )
    configs = ((name, [], make_config(linears)) for name, linears in EXPERIMENTS)
    study_manager.run_with_script('conv_autoencoder', configs)

if __name__ == '__main__':
    main()