from . import (
    xtrains_concepts, 
    xtrains_ontology_simplified,
    xtrains_ontology,
    xtrains
)

import logging
from core.datasets import _dataset_registry
logging.getLogger(__name__).debug(f"Registered datasets:\n{"\n".join(_dataset_registry.keys())}")