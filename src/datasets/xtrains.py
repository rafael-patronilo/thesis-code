from core.datasets import CSVImageDataset, register_datasets
from pathlib import Path
from collections import defaultdict
import numpy as np

PATH=Path("data/xtrains_dataset")


SEED = 42

def some_class(x):
    return x['TypeA'] == 1 or x['TypeB'] == 1 or x['TypeC'] == 1

IMAGE_COLUMN = 'name'
def name_getter(x):
    return f"{int(x):07d}.png"

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

CLASSES = ['TypeA', 'TypeB', 'TypeC']

DTYPES = defaultdict(lambda : np.int32, {
    'name' : str,
    'Angle' : np.float32,
})
# def xtrains_dataset(
#     unfiltered=False,
#     with_targets=True,
#     with_concepts=False,
#     with_unsimplied_concepts=False,
#     seed=SEED
# ):
#     target = []
#     if with_targets:
#         target += ['TypeA', 'TypeB', 'TypeC']
#     if with_concepts:
#         target += []
#     if with_unsimplied_concepts:
#         target += []
#     filter = some_class if not unfiltered else None
#     return CSVImageDataset(
#         csv_path = PATH.joinpath('trains.csv'),
#         images_path = PATH.joinpath('images'),
#         image_columns = [
#             (IMAGE_COLUMN, name_getter),
#         ],
#         target = ['TypeA', 'TypeB', 'TypeC'],
#         features = [IMAGE_COLUMN],
#         filter = filter,
#         random_state=seed
#     )



register_datasets(
    xtrains_unfiltered = CSVImageDataset(
        csv_path = PATH.joinpath('trains.csv'),
        images_path = PATH.joinpath('images'),
        image_columns = [
            (IMAGE_COLUMN, name_getter),
        ],
        target = CLASSES,
        features = [IMAGE_COLUMN],
        random_state=SEED
    ),
    xtrains = CSVImageDataset(
        csv_path = PATH.joinpath('trains.csv'),
        images_path = PATH.joinpath('images'),
        image_columns = [
            (IMAGE_COLUMN, name_getter),
        ],
        target = CLASSES,
        features = [IMAGE_COLUMN],
        filter = some_class,
        random_state=SEED
    ),
    xtrains_with_concepts = CSVImageDataset(
        csv_path = PATH.joinpath('extended_trains.csv'),
        images_path = PATH.joinpath('images'),
        image_columns = [
            (IMAGE_COLUMN, name_getter),
        ],
        target = CLASSES + CONCEPTS,
        features = [IMAGE_COLUMN],
        filter = some_class,
        random_state=SEED
    ),
    xtrains_concepts_only = CSVImageDataset(
        csv_path = PATH.joinpath('extended_trains.csv'),
        images_path = PATH.joinpath('images'),
        image_columns = [
            (IMAGE_COLUMN, name_getter),
        ],
        target = CONCEPTS,
        features = [IMAGE_COLUMN],
        filter = some_class,
        random_state=SEED
    )
)