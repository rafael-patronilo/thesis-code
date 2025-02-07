from typing import  Optional

from dataclasses import dataclass

@dataclass(frozen=True)
class Axiom:
    manchester_syntax: str

@dataclass(frozen=True)
class ObservationAxiom(Axiom):
    concept_name : str
    belief : float

@dataclass(frozen=True)
class EntailmentAxiom(Axiom):
    concept_name : str

@dataclass(frozen=True)
class Justification:
    entailment: EntailmentAxiom
    belief : float
    used_observations : list[ObservationAxiom]
    axioms : list[Axiom]