from datetime import timedelta
from typing import TYPE_CHECKING
from core.init import DO_SCRIPT_IMPORTS
from pathlib import Path

from core.util.progress_trackers import LogProgressContextManager

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
                int(row['TypeA']) == 0 and
                int(row['TypeB']) == 0 and
                int(row['TypeC']) == 0
        ) else "0"

    def long_passenger_car(row):
        index = int(row['name'])
        config = configs[index]
        value = any(
            car[1]['cargo_type'] == 'circle' and car[1]['width'] > 1
            for car in config['cars'])
        return '1' if value else '0'
    progress_cm = LogProgressContextManager(logger, cooldown=timedelta(seconds=30))
    with open(PATH.joinpath('extended_trains.csv'), 'x') as dest_file:
        dest_file.write(','.join(augmented_header) + '\n')
        samples = original_csv.readlines()
        with progress_cm.track('Augmenting trains dataset', 'samples', len(samples)) as progress_tracker:
            for line in samples:
                line = line.strip('\n')
                row = parse_row(line)
                new_values = [
                    at_least(row, 2, 'NumberOfPassengerCars'),
                    at_least(row, 2, 'NumberOfFreightWagons'),
                    long_passenger_car(row),
                    at_least(row, 3, 'NumberOfWagons'),
                    at_least(row, 2, 'NumberOfLongWagons'),
                    other(row)
                ]
                new_line = line + ',' + ','.join(new_values)
                progress_tracker.tick()
                dest_file.write(new_line + '\n')

    original_csv.close()