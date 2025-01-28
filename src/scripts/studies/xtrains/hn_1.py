from typing import TYPE_CHECKING

from core.init import DO_SCRIPT_IMPORTS
if TYPE_CHECKING or DO_SCRIPT_IMPORTS:
    from core.studies import StudyManager
    from core.storage_management import StudyFileManager

STUDY_NAME=f"xtrains_{__name__.split('.')[-1]}"

DATASET_NAME = "xtrains"
ENTRY_CONCEPTS = [
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
]

RN_WITH_WEIGHTS = {
    "model_name" : "L64x2",
    "model_path" : "storage/studies/rn_xtrains_1"
}

RN_WITHOUT_WEIGHTS = {
    "build_script" : "linear_rn",
    "build_args" : [],
    "build_kwargs" : {
        "dataset_name" : "xtrains_ontology",
        "layer_sizes" : [64, 64],
        "num_outputs" : 5
    }
}

def make_pn_config(kwargs):
    return {
        "build_script" : "conv_network",
        "build_args" : [],
        "build_kwargs" : {
            'dataset_name': DATASET_NAME,
            'num_outputs' : len(ENTRY_CONCEPTS),
            'hidden_activations' : ('leaky_relu', 0.1)
        } | kwargs
    }

def make_config(rn_config, pn_kwargs, extra_kwargs):
    kwargs = {
        "dataset_name": DATASET_NAME,
        "concept_dataset_name": "xtrains_concepts_only",
        "concepts": ENTRY_CONCEPTS,
        "pre_trained_learning_rate" : 0.001,
        "untrained_learning_rate" : 0.001,
        "reasoning_network_config" : rn_config,
        "perception_network_config" : make_pn_config(pn_kwargs),
        **extra_kwargs
    }
    return kwargs

# noinspection DuplicatedCode
CONVOLUTIONS = [
    ('C1', (
        [32, 32, ('pool', 2)] +
        [64, ('pool', 2)] * 2
    )),
    ('C2', (
        [32, 32, ('pool', 2)] +
        [64, ('pool', 2)] * 2 +
        [128, ('pool', 2)] * 2
    )),
]


LINEAR_CONFIGS = [
    ('', []),
    ('_L16', [16]),
    ('_L32', [32]),
    ('_L64', [64]),
    ('_L128', [128]),
    ('_2L', [64, 32]),
    ('_3L', [64, 32, 16]),
    ('_4L', [128, 64, 32, 16])
]

def make_configs():
    configs = []
    for name_linear, linear in LINEAR_CONFIGS:
        for name_conv, conv in CONVOLUTIONS:
            name = name_conv + name_linear
            pn_kwargs = {
                "conv_layers": conv,
                "linear_layers": linear
            }
            configs.append((name + '_untRN', [],
                            make_config(
                                RN_WITHOUT_WEIGHTS,
                                pn_kwargs,
                                {'skip_pn_eval': True,
                                 'rn_learning_rate': 0.001,
                                 'activation': 'relu'}
                            )))
            configs.append((name, [],
                            make_config(
                                RN_WITH_WEIGHTS,
                                pn_kwargs,
                                {}
                            )))
    return configs

def main():
    # Load study manager
    file_manager = StudyFileManager(STUDY_NAME)
    study_manager = StudyManager(
        file_manager,
        max_epochs=100
    )

    configs = make_configs()
    study_manager.run_with_script('build_hybrid_network', configs)
