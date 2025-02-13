from typing import TYPE_CHECKING
from core.init import DO_SCRIPT_IMPORTS

if TYPE_CHECKING or DO_SCRIPT_IMPORTS:
    from core import datasets
    from core.studies import StudyManager
    from core.storage_management import StudyFileManager

DATASET = "xtrains_ontology"
#noinspection Duplicates
STUDY_NAME=f"xtrains_{__name__.split('.')[-1]}"


CONFIGS=[
    ('L16', [16]),
    ('L32', [32]),
    ('L64', [64]),
    ('L128', [128]),
    ('L256', [256]),
    ('L512', [256]),
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
            num_outputs=num_ouputs,
            patience=50
        )
    experiments = [(name, [], config_of(layer_sizes=arch)) for name, arch in CONFIGS]
    study_manager.run_with_script('linear_rn', experiments)
