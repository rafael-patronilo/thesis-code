from pathlib import Path

from core.datasets import register_datasets, RandomDataset, SplitDataset
import logging

from core.datasets.csv_dataset import CSVDataset
from core.datasets import dataset_wrappers

logger = logging.getLogger(__name__)

BASIC_CONCEPTS = [
    'PassengerCar',
    'FreightWagon',
    'EmptyWagon',
    'LongWagon',
    'ReinforcedCar',
    'LongPassengerCar',
    'AtLeast2PassengerCars',
    'AtLeast2FreightWagons',
    'AtLeast3Wagons',
    'AtLeast2LongWagons'
]

INTERMEDIARY_CONCEPTS = [
    'WarTrain',
    'PassengerTrain',
    'FreightTrain',
    'RuralTrain',
    'MixedTrain',
    'LongTrain',
    'EmptyTrain',
    'LongFreightTrain'
]

CLASSES = [
    'TypeA',
    'TypeB',
    'TypeC',
    'Other',
    'valid'
]
PATH = Path('data/xtrains_ontology.csv')


COMPLETE_SPLIT = (1.0, 0.0)


basics_to_intermediary = CSVDataset(
    path = PATH,
    target = INTERMEDIARY_CONCEPTS,
    features = BASIC_CONCEPTS,
    splits=COMPLETE_SPLIT
)

intermediary_to_classes = CSVDataset(
    path = PATH,
    target = CLASSES,
    features = INTERMEDIARY_CONCEPTS,
    splits=COMPLETE_SPLIT
)

basics_to_classes = CSVDataset(
    path = PATH,
    target = CLASSES,
    features = BASIC_CONCEPTS,
    splits=COMPLETE_SPLIT
)

register_datasets(
    xtrains_ontology = basics_to_classes,
    xtrains_ontology_lvl1 = basics_to_intermediary,
    xtrains_ontology_lvl2 = intermediary_to_classes
)