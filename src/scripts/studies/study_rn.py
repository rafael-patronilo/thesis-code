from core import StudyManager, datasets
from core.storage_management import StudyFileManager
from torch import nn

DATASET = "xtrains_ontology_simplified_comp_some_all"

CONFIGS=[
    ('L16', [16]),
    ('L32', [32]),
    ('L16x2', [16, 16]),
    ('L32x2', [32, 32]),
    ('L16x3', [16, 16, 16]),
    ('L32x3', [32, 32, 32]),
]

def main():
    # Load study manager
    file_manager = StudyFileManager("rn_xtrains_1")
    dataset = datasets.get_dataset(DATASET)
    num_ouputs = dataset.get_shape()[1][0]
    study_manager = StudyManager(
        file_manager,
        compare_strategy="max",
        metric_key=("val", "accuracy"),
        num_epochs=50
    )
    def config_of(layer_sizes):
        return dict(
            dataset_name=DATASET,
            layer_sizes=layer_sizes,
            num_outputs=num_ouputs
        )
    experiments = [(name, [], config_of(layer_sizes=arch)) for name, arch in CONFIGS]
    study_manager.run_with_script('linear_rn', experiments)

if __name__ == '__main__':
    main()