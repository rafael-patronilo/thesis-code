from typing import TYPE_CHECKING
from collections import OrderedDict
from core.init import DO_SCRIPT_IMPORTS
if TYPE_CHECKING or DO_SCRIPT_IMPORTS:
    from core import datasets
    from core.studies import StudyManager
    from core.storage_management import StudyFileManager
    from typing import NamedTuple
    from torch import nn
    import sys
STUDY_NAME = "study_hn_1"
BUILD_SCRIPT = ''

BASE_KWARGS= {
    "dataset_name" : "xtrains",
    "concept_dataset_name" : "xtrains_concepts_only",
    "concepts" :[
        "PassengerCar",
        "FreightWagon",
        "EmptyWagon",
        "LongWagon",
        "ReinforcedCar",
        "LongPassengerCar",
        "AtLeast2PassengerCars",
        "AtLeast2FreightWagons",
        "AtLeast3Wagons",
        "AtLeast2LongWagons"
    ],
    "reasoning_network_path" : {
        "model_name" : "L32x2",
        "model_path" : "storage/studies/rn_xtrains_1"
    },
    "perception_network_config" : {
        "build_script" : "conv_pn",
        "build_args" : [],
        "build_kwargs" : {
            "num_concepts": 10,
        }
    }
}

# noinspection DuplicatedCode
CONVOLUTIONS = {
    "C32" : (
        [32, 32, ('pool', 2)]
    ),
    "C64" : (
        [32, 32, ('pool', 2)] +
        [64, ('pool', 2)]
    ),
    "C64x2" : (
        [32, 32, ('pool', 2)] +
        [64, ('pool', 2)] * 2
    ),
    "C64x3" : (
        [32, 32, ('pool', 2)] +
        [64, ('pool', 2)] * 3
    ),
    "C64x4" : (
        [32, 32, ('pool', 2)] +
        [64, ('pool', 2)] * 4
    ),
    "C128" : (
        [32, 32, ('pool', 2)] +
        [64, ('pool', 2)] * 2 +
        [128, ('pool', 2)]
    )
}


LINEARS = {
    '' : [],
    'L64': [64],
    'L128': [128],
    '2L': [64, 32],
    '3L': [64, 32, 16],
    '4L': [128, 64, 32, 16]
}

EXPERIMENTS = [
    ('C32', ''),
    ('C32', 'L64'),
    ('C32', 'L128'),
    ('C32', '2L'),
    ('C32', '3L'),
    ('C32', '4L'),
    ('C64', ''),
    ('C64', 'L64'),
    ('C64', 'L128'),
    ('C64', '2L'),
    ('C64', '3L'),
    ('C64', '4L'),
    ('C64x2', ''),
    ('C64x2', 'L64'),
    ('C64x2', 'L128'),
    ('C64x2', '2L'),
    ('C64x2', '3L'),
    ('C64x3', ''),
    ('C64x3', 'L64'),
    ('C64x3', 'L128'),
    ('C64x3', '3L'),
    ('C64x4', ''),
    ('C64x4', 'L64'),
    ('C64x4', '2L'),
    ('C128', ''),
    ('C128', 'L64'),
    ('C128', '3L')
]


def gen_configs() -> list[tuple[str, list, dict]]:
    configs = []
    for c_name, l_name in EXPERIMENTS:
        config = BASE_KWARGS.copy()
        config['perception_network_config']['build_kwargs'].update(
            conv_layers=CONVOLUTIONS[c_name],
            linear_layers=LINEARS[l_name]
        )
        configs.append((f"{c_name}_{l_name}", [], config))
    return configs


def main():
    # Load study manager
    file_manager = StudyFileManager(STUDY_NAME)
    study_manager = StudyManager(
        file_manager,
        max_epochs=100
    )
    configs = gen_configs()
    study_manager.run_with_script('build_hybrid_network', configs)

if __name__ == '__main__':
    main()