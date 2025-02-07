import json
from typing import TypedDict, Optional

from core.eval.justifier_wrapper.justifier_result import Justification, Axiom, ObservationAxiom, EntailmentAxiom

class JSONAxiom(TypedDict):
    axiom: str
    belief : Optional[float]

class JSONJustification(TypedDict):
    entailment: str
    belief : float
    used_observations : list[int]
    axioms : list[JSONAxiom]

class JustifierOutput(TypedDict):
    loaded_observations : list[JSONAxiom]
    justifications : list[JSONJustification]

def parse_json(
        output : str,
        original_entailment : str,
        original_concepts : list[tuple[str, float]]
) -> list[Justification]:
    parsed_json = json.loads(output)
    return parse_output(parsed_json, original_entailment, original_concepts)

def parse_output(
        output: JustifierOutput,
        original_entailment : str,
        original_concepts : list[tuple[str, float]]
) -> list[Justification]:
    json_justifications = output['justifications']
    loaded_observations = output['loaded_observations']
    justifications = []
    for justification in json_justifications:
        axioms = [Axiom(ax['axiom']) for ax in justification['axioms']]
        used_observations = [
            ObservationAxiom(
                manchester_syntax=loaded_observations[index]['axiom'],
                concept_name=original_concepts[index][0],
                belief=original_concepts[index][1]
            ) for index in justification['used_observations']
        ]
        justifications.append(
            Justification(
                entailment=EntailmentAxiom(justification['entailment'], original_entailment),
                belief=justification['belief'],
                axioms=axioms,
                used_observations=used_observations
            )
        )
    return justifications

