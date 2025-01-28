from typing import TYPE_CHECKING
from core.init import DO_SCRIPT_IMPORTS
from pathlib import Path
if TYPE_CHECKING or DO_SCRIPT_IMPORTS:
    import pickle
    import logging
    logger = logging.getLogger(__name__)

PATH = Path('data/xtrains_dataset')

def main():
    logger.info("Loading configs...")
    with open(PATH.joinpath('configs.pickle'), 'rb') as file:
        configs = pickle.load(file)
    logger.info("Opening original csv")
    original_csv = open(PATH.joinpath('trains.csv'), 'r')
    header = original_csv.readline().strip('\n').split(',')
    logger.info(f"Original header: {header}")
    augmented_header = header + [
        'AtLeast2PassengerCars',
        'AtLeast2FreightWagons',
        'LongPassengerCar',
        'AtLeast3Wagons',
        'AtLeast2LongWagons',
        'Other'
    ]
    logger.info(f"Augmenting to {augmented_header}")

    def parse_row(row: str):
        cells = row.split(',')
        return {k: v for k, v in zip(header, cells)}

    def at_least(row, value, collumn):
        return '1' if int(row[collumn]) >= value else '0'

    def other(row):
        return "1" if (
                row['TypeA'] == 0 and
                row['TypeB'] == 0 and
                row['TypeC'] == 0
        ) else "0"

    def long_passenger_car(row):
        index = int(row['name'])
        config = configs[index]
        value = any(
            car[1]['cargo_type'] == 'circle' and car[1]['width'] > 1
            for car in config['cars'])
        return '1' if value else '0'

    with open(PATH.joinpath('extended_trains.csv'), 'x') as dest_file:
        dest_file.write(','.join(augmented_header) + '\n')
        for line in original_csv.readlines():
            line = line.strip('\n')
            row = parse_row(line)
            logger.info(f"Processing {row['name']}")
            new_values = [
                at_least(row, 2, 'NumberOfPassengerCars'),
                at_least(row, 2, 'NumberOfFreightWagons'),
                long_passenger_car(row),
                at_least(row, 3, 'NumberOfWagons'),
                at_least(row, 2, 'NumberOfLongWagons'),
                other(row)
            ]
            new_line = line + ',' + ','.join(new_values)
            logger.info('\t' + new_line)
            dest_file.write(new_line + '\n')

    original_csv.close()