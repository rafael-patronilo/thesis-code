import logging
from typing import Optional
from torch import nn
from core.nn.layers import MakeBinary, NegateMask, Reorder

module_logger = logging.getLogger(__name__)

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

def class_to_latex_cmd(cls : str):
    negate = False
    if cls.startswith('!'):
        cls = cls[1:]
        negate = True
    if cls not in CLASSES:
        if cls in SHORT_CLASSES:
            cls = CLASSES[SHORT_CLASSES.index(cls)]
        else:
            raise ValueError(f"Unknown class: {cls}")
    if cls.startswith('AtLeast2'):
        cmd = f"\\AtLeastTwo{cls[8:]}"
    elif cls.startswith('AtLeast3'):
        cmd =  f"\\AtLeastThree{cls[8:]}"
    else:
        cmd = f"\\{cls}"
    if negate:
        cmd = f"\\neg {cmd}"
    return f"${cmd}$"

def make_order_from_attribution(attribution: list[str]):
    for c in attribution:
        if c not in SHORT_CONCEPTS:
            raise ValueError(f"Unknown concept {c} in attribution")
    module_logger.info(f"Searching indices for concepts {attribution}")
    indices = [SHORT_CONCEPTS.index(c) for c in attribution]
    str_builder = [f"Found indices {indices}, corresponding to the attribution:\n"]
    for pn_i, rn_i in enumerate(indices):
        str_builder.append(f"\tPN output {pn_i} -> RN input {rn_i} ({SHORT_CONCEPTS[rn_i]})\n")
    str_builder.append("Reordering")
    module_logger.info(''.join(str_builder))
    order = [indices.index(i) for i in range(len(indices))]
    str_builder = [f"Order: {order}\n"]
    for rn_i, pn_i in enumerate(order):
        str_builder.append(f"\tPN output {pn_i} -> RN input {rn_i} "
                           f"({SHORT_CONCEPTS[rn_i]})\n")
    module_logger.info(''.join(str_builder))
    return order

def prepare_pn_with_attribution(
        pn : nn.Module,
        attribution : list[str],
        binary_threshold : Optional[float]
) -> nn.Sequential:
    negate = [c.startswith('!') for c in attribution]
    concepts = [c[1:] if n else c for c, n in zip(attribution, negate)]

    layers = [pn]
    if binary_threshold is not None:
        module_logger.info(f"Adding binary threshold layer with threshold {binary_threshold}")
        module_logger.warning("PN output values will be discrete: precisely 0 or 1")
        layers.append(MakeBinary(binary_threshold))
    if any(negate):
        module_logger.info(f"Applying negation mask to PN: {negate}")
        layers.append(NegateMask(negate))
    order = make_order_from_attribution(concepts)
    layers.append(Reorder(order))
    return nn.Sequential(*layers)