from core.datasets.binary_generator import BinaryGeneratorBuilder, BinaryASTNode
from core.datasets import dataset_registry, RandomDataset, SplitDataset
from . import xtrains_ontology_simplified
import logging

logger = logging.getLogger(__name__)
PREFIX = 'xtrains_ontology'

def _build_ontology():
    # reuse ontology definition from xtrains_ontology_simplified
    return xtrains_ontology_simplified._build_ontology(simplify=False)

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
        val_seed=48,
        test_seed=49
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