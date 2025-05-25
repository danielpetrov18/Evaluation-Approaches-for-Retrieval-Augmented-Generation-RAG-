# pylint: disable=C0114
# pylint: disable=C0115
# pylint: disable=C0301
# pylint: disable=W0622

import json
from typing import List, Final, Tuple, Union, Literal

from pydantic import BaseModel
from deepeval.metrics.contextual_relevancy.template import ContextualRelevancyTemplate

class Verdict(BaseModel):
    verdict: Literal["yes", "no"]
    statement: str
    reason: Union[str, None] = None

class Verdicts(BaseModel):
    verdicts: List[Verdict]

FEW_SHOT_EXAMPLES: Final[List[Tuple[str, str, Verdicts]]] = [
    (
        "Albert Einstein developed the theory of relativity.",
        "Einstein was born in Ulm, Germany in 1879. He loved playing the violin. He worked at the Swiss Patent Office before publishing the theory of relativity. He was also a pacifist.",
        Verdicts(
            verdicts=[
                Verdict(verdict="no", statement="Einstein was born in Ulm, Germany in 1879.", reason="The birth location is not relevant to his development of the theory of relativity."),
                Verdict(verdict="no", statement="He loved playing the violin.", reason="Musical interests are unrelated to the theory of relativity."),
                Verdict(verdict="yes", statement="He worked at the Swiss Patent Office before publishing the theory of relativity."),
                Verdict(verdict="no", statement="He was also a pacifist.", reason="His pacifism is unrelated to the scientific work mentioned in the input."),
            ]
        )
    ),
]

class MyContextualRelevancyTemplate(ContextualRelevancyTemplate):

    @staticmethod
    def generate_verdicts(input: str, context: str):
        examples_str: str = ""
        for i, (i, c, v) in enumerate(FEW_SHOT_EXAMPLES, 1):
            examples_str += f"EXAMPLE {i}:\n"
            examples_str += f"INPUT:\n\"{i}\"\n"
            examples_str += f"Context:\n\"{c}\"\n"
            examples_str += f"JSON:\n{MyContextualRelevancyTemplate._clean_verdict_json(v)}\n\n"

        return f"""Your task is to determine for each statement in the context, whether or not it would be useful for answering the input.
Relevant statement would be something that is useful in answering the input, while irrelevant statement would be something that is not useful in answering the input.
Please generate a JSON object to indicate for each statement found in the context whether it is relevant to the provided input.
The JSON will be a list of 'verdicts', with 2 mandatory fields: 'verdict' and 'statement', and 1 optional field: 'reason'.
A reason should only be provided if the verdict is 'no'.
The 'verdict' key should STRICTLY be either 'yes' or 'no', and states whether the statement is relevant to the input.
The 'statement' key should contain the statement from the context, to which the verdict applies.

Please provide the verdicts in the following JSON format:
{{
    "verdicts": [
        {{
            "verdict": "<yes | no>",
            "statement": "<statement>",
            "reason": "<reason>" (optional, only if verdict is 'no')
        }},
        {{
            "verdict": "<yes | no>",
            "statement": "<statement>",
            "reason": "<reason>" ()
        }},
        // More verdicts...
]
}}

====== FEW SHOT EXAMPLES ======

{examples_str.strip()}

====== END OF EXAMPLES ======

**
IMPORTANT:
- Please make sure to only return in JSON format.
- DO NOT provide any further explanations or clarifications.
**

INPUT:
"{input}"

CONTEXT:
"{context}"

JSON:
"""

    @staticmethod
    def _clean_verdict_json(verdicts_model: Verdicts) -> str:
        """Custom JSON serialization that excludes null reason fields"""
        verdicts_dict = {"verdicts": []}

        for v in verdicts_model.verdicts:
            verdict_dict = {"verdict": v.verdict}
            if v.reason is not None:
                verdict_dict["reason"] = v.reason
            verdicts_dict["verdicts"].append(verdict_dict)

        return json.dumps(verdicts_dict, indent=4)
