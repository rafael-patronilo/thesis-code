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

def class_to_manchester_assertion(cls : str, negate : bool = False) -> str:
    prefix = "__input__ Type: "
    negation = "not " if negate else ""
    concept : str
    match cls:
        case 'TypeA' | 'TypeB' | 'TypeC':
            concept = f"({cls})"
        case 'PassengerCar' | 'FreightWagon' | 'EmptyWagon' | 'LongWagon' | 'ReinforcedCar':
            concept = f"(has some {cls})"
        case 'LongPassengerCar':
            concept = f"(has some (LongWagon and PassengerCar))"
        case 'AtLeast2PassengerCars':
            concept = "(has min 2 PassengerCar)"
        case 'AtLeast2FreightWagons':
            concept = "(has min 2 FreightWagon)"
        case 'AtLeast3Wagons':
            concept = "(has min 3 Wagon)"
        case 'AtLeast2LongWagons':
            concept = "(has min 2 LongWagon)"
        case _:
            raise ValueError(f"Unknown class: {cls}")
    return f"{prefix}{negation}{concept}"
