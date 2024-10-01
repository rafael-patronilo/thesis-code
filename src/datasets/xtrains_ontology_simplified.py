from core.datasets.binary_generator import BinaryGeneratorBuilder, BinaryASTNode
from core.datasets import dataset_registry, RandomDataset, SplitDataset
import logging

logger = logging.getLogger(__name__)
PREFIX = 'xtrains_ontology_simplified'

def _build_ontology(simplify: bool = True, include_intermediary: bool = False) -> BinaryGeneratorBuilder:
    gen = BinaryGeneratorBuilder()
    
    # Features
    atLeast2PassengerCars = gen.gen_var()
    atLeast2FreightWagons = gen.gen_var()
    hasLongPassengerCar = gen.gen_var()
    atLeast2LongWagons = gen.gen_var()
    atLeast3Wagons = gen.gen_var()

    hasPassengerCar = gen.implied_by(atLeast2PassengerCars | hasLongPassengerCar)
    hasFreightWagon = gen.implied_by(atLeast2FreightWagons)
    hasEmptyWagon = gen.implied_by(
        (atLeast3Wagons | atLeast2LongWagons) & # must have at least 1 (unspecified) wagon
                                                # default to empty if no other type is specified
        ~ hasPassengerCar & 
        ~ hasFreightWagon
    )

    hasLongWagon = gen.implied_by(atLeast2LongWagons | hasLongPassengerCar)

    hasReinforcedCar = gen.gen_var()
    
    # store all nodes so far as features
    gen.features = {k : v for k, v in locals().items() if isinstance(v, BinaryASTNode)}

    # Intermediary concepts (added for readability)
    allEmpty = ~ hasPassengerCar & ~ hasFreightWagon
    hasPassengerCarOrFreightWagon = hasPassengerCar | hasFreightWagon

    # Intermediary concepts (present in the original ontology)
    emptyTrain = allEmpty & hasEmptyWagon
    longTrain = atLeast2LongWagons | atLeast3Wagons

    # Simplified intermediary concepts: 
    #       the following equivalences were reverse implications in the original ontology
    warTrain = hasReinforcedCar & hasPassengerCar
    passengerTrain = hasLongPassengerCar | atLeast2PassengerCars
    freighTrain = atLeast2FreightWagons
    ruralTrain = hasEmptyWagon & hasPassengerCarOrFreightWagon & ~ hasLongWagon
    mixedTrain = hasPassengerCar & hasFreightWagon & hasEmptyWagon

    if not simplify:
        # convert to reverse implications
        warTrain = gen.implied_by(warTrain)
        passengerTrain = gen.implied_by(passengerTrain)
        freighTrain = gen.implied_by(freighTrain)
        ruralTrain = gen.implied_by(ruralTrain)
        mixedTrain = gen.implied_by(mixedTrain)
    if include_intermediary:
        # add features
        gen.features['warTrain'] = warTrain
        gen.features['passengerTrain'] = passengerTrain
        gen.features['freighTrain'] = freighTrain
        gen.features['ruralTrain'] = ruralTrain
        gen.features['mixedTrain'] = mixedTrain

    # More intermediary concepts (present in the original ontology)
    # These were always equivalences.
    longFreightTrain = longTrain & freighTrain
    
    # Labels
    typeA = warTrain | emptyTrain
    typeB = passengerTrain | longFreightTrain
    typeC = ruralTrain | mixedTrain
    gen.labels["typeA"] = typeA
    gen.labels["typeB"] = typeB
    gen.labels["typeC"] = typeC

    return gen

_ontology = _build_ontology() # temporary variable to avoid re-executing the function

logger.debug(f"{_ontology}\nNumber of samples: {len(_ontology.build()):_}")
SIZE = len(_ontology.build())
del _ontology # remove temporary variable, to avoid reusing the same object

RANDOM_SIZES = (50_000, 10_000, 10_000)

if SIZE <= sum(RANDOM_SIZES):
    RANDOM_SIZES = (int(0.8*SIZE), int(0.1*SIZE), int(0.1*SIZE))

def _random_dataset(func) -> RandomDataset:
    return RandomDataset(
        func, 
        RANDOM_SIZES,
        val_seed=42,
        test_seed=43
    )

generators = {
    "typeA" : _build_ontology().with_labels("typeA").build(),
    "typeB" : _build_ontology().with_labels("typeB").build(),
    "typeC" : _build_ontology().with_labels("typeC").build(),
    "all"   : _build_ontology().build()
}

random_datasets : dict[str, RandomDataset] = {f"{PREFIX}_rand_{k}" : _random_dataset(v.generate_random) for k, v in generators.items()}
complete_datasets : dict[str, SplitDataset] = {f"{PREFIX}_comp_{k}" : v.as_complete_dataset() for k, v in generators.items()}
dataset_registry.update(random_datasets)
dataset_registry.update(complete_datasets)