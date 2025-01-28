
from core.init import DO_SCRIPT_IMPORTS
from typing import TYPE_CHECKING
from collections import OrderedDict
from pathlib import Path


if TYPE_CHECKING or DO_SCRIPT_IMPORTS:
    from core.datasets.binary_generator import BinaryGeneratorBuilder
    import torch
    from datetime import timedelta
    from core.util.progress_trackers import LogProgressContextManager
    import logging
    logger = logging.getLogger(__name__)
    progress_cm = LogProgressContextManager(logger, cooldown=timedelta(minutes=5))

def _build_ontology() -> 'BinaryGeneratorBuilder':
    gen = BinaryGeneratorBuilder()

    # Features
    two_passenger = gen.free_variable()
    two_freight = gen.free_variable()
    long_passenger = gen.free_variable()
    two_long_wagon = gen.free_variable() # not a feature
    three_wagon = gen.free_variable()

    passenger_car = gen.implied_by(two_passenger | long_passenger)
    freight_wagon = gen.implied_by(two_freight)
    empty_wagon = gen.implied_by(
        (three_wagon | two_long_wagon) & # then it must have at least 1 (unspecified) wagon
        # default to empty if no other type is specified
        ~ passenger_car &
        ~ freight_wagon
    )

    long_wagon = gen.implied_by(two_long_wagon | long_passenger)

    reinforced_car = gen.free_variable()

    # store all nodes so far as features
    gen.features = OrderedDict(
        PassengerCar=passenger_car,
        FreightWagon=freight_wagon,
        EmptyWagon=empty_wagon,
        LongWagon=long_wagon,
        ReinforcedCar=reinforced_car,
        LongPassengerCar=long_passenger,
        AtLeast2PassengerCars=two_passenger,
        AtLeast2FreightWagons=two_freight,
        AtLeast3Wagons=three_wagon,
        AtLeast2LongWagons=two_long_wagon
    )

    # Intermediary concepts (added for readability)
    all_empty = ~ passenger_car & ~ freight_wagon
    passenger_car_or_freight_wagon = passenger_car | freight_wagon

    # Intermediary concepts (present in the original ontology)
    empty_train = all_empty & empty_wagon
    long_train = two_long_wagon | three_wagon

    # Simplified intermediary concepts:
    #       the following equivalences were reverse implications in the original ontology
    war_train = reinforced_car & passenger_car
    passenger_train = long_passenger | two_passenger
    freigh_train = two_freight
    rural_train = empty_wagon & passenger_car_or_freight_wagon & ~ long_wagon
    mixed_train = passenger_car & freight_wagon & empty_wagon

    # More intermediary concepts (present in the original ontology)
    # These were always equivalences.
    long_freight_train = long_train & freigh_train

    # Labels
    type_a = war_train | empty_train
    type_b = passenger_train | long_freight_train
    type_c = rural_train | mixed_train
    other = ~ (type_a | type_b | type_c)
    gen.labels.update(
        TypeA = type_a,
        TypeB = type_b,
        TypeC = type_c,
        Other = other,
        WarTrain = war_train,
        PassengerTrain = passenger_train,
        FreightTrain = freigh_train,
        RuralTrain = rural_train,
        MixedTrain = mixed_train,
        LongTrain = long_train,
        EmptyTrain = empty_train,
        LongFreightTrain = long_freight_train
    )

    return gen

PATH = Path("data/xtrains_ontology.csv")

def main():
    generator = _build_ontology().build()
    feature_names = generator.feature_names
    label_names = generator.label_names
    assert feature_names is not None and label_names is not None
    header = feature_names + label_names + [generator.valid_label]
    if PATH.exists():
        raise FileExistsError(f"{PATH} already exists")
    with open(PATH, "w") as f:
        f.write(",".join(header) + "\n")
        with progress_cm.track("Dataset Generation", "rows", len(generator)) as progress:
            for i in range(len(generator)):
                row = generator.generate_from_int(i, force_valid=False)
                row = torch.cat(row)
                row = row.tolist()
                f.write(",".join(map(str, row)) + "\n")
                progress.tick()
