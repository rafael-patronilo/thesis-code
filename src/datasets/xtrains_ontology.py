from pathlib import Path

from core.datasets import register_datasets, RandomDataset, SplitDataset
import logging

from core.datasets.csv_dataset import CSVDataset

logger = logging.getLogger(__name__)

CONCEPTS = [
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

CLASSES = ['TypeA', 'TypeB', 'TypeC', 'valid']
PATH = Path('data/xtrains_ontology.csv')


SEED = 170125
COMPLETE_SPLIT = (1.0, 0.0)


register_datasets(
    xtrains_ontology = CSVDataset(
        path = PATH,
        target = CLASSES,
        features = CONCEPTS,
        splits=COMPLETE_SPLIT,
        random_state=SEED
    )
)