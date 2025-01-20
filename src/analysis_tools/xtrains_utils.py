import logging


CLASSES = [
    'PassengerCar',
    'FreightWagon',
    'EmptyWagon',
    'LongWagon',
    'ReinforcedCar',
    'LongPassengerCar',
    'AtLeast2PassengerCars',
    'AtLeast2FreightWagons',
    'AtLeast3Wagons',
    'AtLeast2LongWagons',
    'TypeA',
    'TypeB',
    'TypeC',
]

SHORT_CONCEPTS = [
    'Passenger',
    'Freight',
    'Empty',
    'Long',
    'Reinforced',
    'LongPassenger',
    '2Passenger',
    '2Freight',
    '3Wagons',
    '2Long'
]

SHORT_TYPES = [
    'TypeA',
    'TypeB',
    'TypeC',
]

SHORT_CLASSES = SHORT_CONCEPTS + SHORT_TYPES

def log_short_class_correspondence(logger : logging.Logger):
    correspondence = [f'\t{name} -> {short}' for name, short in zip(CLASSES, SHORT_CLASSES)]
    logger.info('Concept names have been shortened for convenience:\n' +
                ('\n'.join(correspondence)))