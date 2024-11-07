from core.datasets import CSVImageDataset, register_datasets
from pathlib import Path

PATH=Path("data/xtrains_mine")

def some_class(x):
    return x['TypeA'] == 1 or x['TypeB'] == 1 or x['TypeC'] == 1

register_datasets(
    xtrains_unfiltered = CSVImageDataset(
        csv_path = PATH.joinpath('trains.csv'),
        images_path = PATH.joinpath('images'),
        image_collumns = [
            ('name', lambda x: f"{x}.png"),
        ],
        target = ['TypeA', 'TypeB', 'TypeC'],
        features = ['name']
    ),
    xtrains = CSVImageDataset(
        csv_path = PATH.joinpath('trains.csv'),
        images_path = PATH.joinpath('images'),
        image_collumns = [
            ('name', lambda x: f"{x}.png"),
        ],
        target = ['TypeA', 'TypeB', 'TypeC'],
        features = ['name'],
        filter = some_class
    )
)
