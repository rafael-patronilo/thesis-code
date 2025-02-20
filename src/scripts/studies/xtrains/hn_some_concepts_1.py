

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
    "model_path" : "storage/studies/xtrains_rn_1"
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


SAMPLE_SELECTION_SEED = 1410951

def make_config(num_samples_with_concepts, rn_config, pn_kwargs, extra_kwargs):
    kwargs = {
        "dataset_name": DATASET_NAME,
        "concept_dataset_name": "xtrains_concepts_only",
        "concepts": ENTRY_CONCEPTS,
        "pre_trained_learning_rate" : 0.001,
        "untrained_learning_rate" : 0.001,
        "reasoning_network_config" : rn_config,
        "perception_network_config" : make_pn_config(pn_kwargs),
        "sample_selection_seed" : SAMPLE_SELECTION_SEED,
        "num_samples_with_concepts" : num_samples_with_concepts,
        **extra_kwargs
    }
    return kwargs

# noinspection DuplicatedCode
CONVOLUTIONS = [
    ('C2', (
        [32, 32, ('pool', 2)] +
        [64, ('pool', 2)] * 2 +
        [128, ('pool', 2)] * 2
    )),
]


LINEAR_CONFIGS = [
    ('_L128', [128])
]

NUM_SAMPLES_WITH_CONCEPTS = [
    5,
    10,
    25,
    50,
    75,
    100,
    150,
    200
]

def make_configs():
    configs = []
    for name_linear, linear in LINEAR_CONFIGS:
        for name_conv, conv in CONVOLUTIONS:
            for num_samples in NUM_SAMPLES_WITH_CONCEPTS:
                name = name_conv + name_linear + f"_S{num_samples}"
                pn_kwargs = {
                    "conv_layers": conv,
                    "linear_layers": linear
                }
                configs.append((name, [],
                                make_config(
                                    num_samples,
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
        max_epochs=20
    )

    configs = make_configs()
    study_manager.run_with_script('build_hn_some_concepts', configs)
