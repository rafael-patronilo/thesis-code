from typing import TYPE_CHECKING
from core.init import DO_SCRIPT_IMPORTS

if TYPE_CHECKING or DO_SCRIPT_IMPORTS:
    from core import datasets
    from core.studies import StudyManager
    from core.storage_management import StudyFileManager

DATASET = "xtrains_ontology_lvl2"

#noinspection Duplicates
STUDY_NAME=f"xtrains_{__name__.split('.')[-1]}"

CONFIGS = [
    ('L64x3L32x2', [64, 64, 64, 32, 32]),
    ('L64x3L32x2L16', [64, 64, 64, 32, 32, 16]),
    ('L64x3L32x3L16', [64, 64, 64, 32, 32, 32, 16]),
    ('L128L64x2', [128, 64, 64, 32, 16]),
    ('L128L64x3', [128, 64, 64, 64, 32, 16]),
    ('L128x2L64x3', [128, 128, 64, 64, 64, 32, 16]),
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
