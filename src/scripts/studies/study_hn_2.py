from collections import OrderedDict
from typing import TYPE_CHECKING

from copy import deepcopy

from core.init import DO_SCRIPT_IMPORTS

if TYPE_CHECKING or DO_SCRIPT_IMPORTS:
    from core.storage_management import StudyFileManager
    from core.studies import StudyManager

STUDY_NAME = "study_hn_2"

DATASET_NAME = "xtrains"

BASE_KWARGS= {
    "dataset_name" : DATASET_NAME,
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
        "build_script" : "xtrains_perception_network",
        "build_args" : []
    }
}

COMMON_PN_CONFIG = {
    'dataset_name': DATASET_NAME,
    'first_kernel_size': 5,
    'first_pool_kernel_size': 9,
    'last_linear_output_size': 10,
    'hidden_activations': ('leaky_relu', 0.1)
}

CONV_K1_PN_CONFIGS=OrderedDict(
    C64_2C = {
        'first_conv_layers' : [32, 64],
        'k1_conv_layers': [32, 32, 6]
    },
    C64_3C = {
        'first_conv_layers' : [32, 64],
        'k1_conv_layers': [32, 32, 32, 6]
    },
    C64_4C = {
        'first_conv_layers' : [32, 64],
        'k1_conv_layers': [64, 32, 32, 32, 6]
    },
    C64_4C128 = {
        'first_conv_layers' : [32, 64],
        'k1_conv_layers': [128, 64, 64, 32, 6]
    },
    C64_5C = {
        'first_conv_layers' : [32, 64],
        'k1_conv_layers': [128, 64, 64, 32, 32, 6]
    },
)

def gen_configs():
    configs = []
    for name, kwargs in CONV_K1_PN_CONFIGS.items():
        config = deepcopy(BASE_KWARGS)
        subconfig : dict = config['perception_network_config']
        subconfig['build_kwargs'] = (COMMON_PN_CONFIG | kwargs)
        configs.append((name, [], config))
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