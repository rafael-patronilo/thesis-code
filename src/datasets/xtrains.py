from core.datasets import CSVImageDataset, register_datasets
from pathlib import Path

PATH=Path("data/xtrains_mine")
seed = 42

def some_class(x):
    return x['TypeA'] == 1 or x['TypeB'] == 1 or x['TypeC'] == 1

IMAGE_COLLUMN = 'name'
def name_getter(x):
    return f"{int(x):07d}.png"

register_datasets(
    xtrains_unfiltered = CSVImageDataset(
        csv_path = PATH.joinpath('trains.csv'),
        images_path = PATH.joinpath('images'),
        image_collumns = [
            (IMAGE_COLLUMN, name_getter),
        ],
        target = ['TypeA', 'TypeB', 'TypeC'],
        features = [IMAGE_COLLUMN],
        random_state=seed
    ),
    xtrains = CSVImageDataset(
        csv_path = PATH.joinpath('trains.csv'),
        images_path = PATH.joinpath('images'),
        image_collumns = [
            (IMAGE_COLLUMN, name_getter),
        ],
        target = ['TypeA', 'TypeB', 'TypeC'],
        features = [IMAGE_COLLUMN],
        filter = some_class,
        random_state=seed
    )
)
