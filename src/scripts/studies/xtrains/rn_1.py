from typing import TYPE_CHECKING
from core.init import DO_SCRIPT_IMPORTS

if TYPE_CHECKING or DO_SCRIPT_IMPORTS:
    from core import datasets
    from core.studies import StudyManager
    from core.storage_management import StudyFileManager

DATASET = "xtrains_ontology"
#noinspection Duplicates
STUDY_NAME=__name__


CONFIGS=[
    ('L16', [16]),
    ('L32', [32]),
    ('L64', [64]),
    ('L16x2', [16, 16]),
    ('L16L32', [16, 32]),
    ('L32x2', [32, 32]),
    ('L32L64', [32, 64]),
    ('L64x2', [64, 64]),
    ('L16x3', [16, 16, 16]),
    ('L16L32x2', [16, 32, 32]),
    ('L32x3', [32, 32, 32]),
    ('L16L32L64', [16, 32, 64]),
    ('L64x3', [64, 64, 64]),
]

#noinspection Duplicates
def main():
    # Load study manager
    file_manager = StudyFileManager(STUDY_NAME)
    dataset = datasets.get_dataset(DATASET)
    num_ouputs = dataset.get_shape()[1][0]
    study_manager = StudyManager(
        file_manager,
        max_epochs=100
    )
    def config_of(layer_sizes):
        return dict(
            dataset_name=DATASET,
            layer_sizes=layer_sizes,
            num_outputs=num_ouputs
        )
    experiments = [(name, [], config_of(layer_sizes=arch)) for name, arch in CONFIGS]
    study_manager.run_with_script('linear_rn', experiments)
