# pylint: disable=C0114
# pylint: disable=C0115
# pylint: disable=C0301
# pylint: disable=W0622

from typing import List, Final, Tuple, Literal

from pydantic import BaseModel
from deepeval.metrics.contextual_recall import ContextualRecallTemplate

class Verdict(BaseModel):
    verdict: str = Literal["yes", "no"]
    reason: str

class Verdicts(BaseModel):
    verdicts: List[Verdict]

FEW_SHOT_EXAMPLES: Final[List[Tuple[str, List[str], Verdicts]]] = [
    (
        "The device operates for up to 8 hours on battery. It supports wireless charging. The screen is 5.5 inches wide.",
        [
            "Battery life extends to 8 hours with moderate usage.",
            "The screen measures 5.5 inches diagonally."
        ],
        Verdicts(
            verdicts=[
                Verdict(verdict="yes", reason="Battery life of 8 hours is mentioned in the first context."),
                Verdict(verdict="no", reason="There's no mention of wireless charging in the contexts."),
                Verdict(verdict="yes", reason="The third context supports the statement about the 5.5 inch screen.")
            ]
        )
    )
]

class MyContextualRecallTemplate(ContextualRecallTemplate):

    @staticmethod
    def generate_verdicts(expected_output: str, retrieval_context: List[str]) -> str:
        examples_str: str = ""
        for i, (eo, rc, verdicts) in enumerate(FEW_SHOT_EXAMPLES, 1):
            examples_str += f"EXAMPLE {i}:\n"
            examples_str += f"EXPECTED OUTPUT:\n\"{eo}\"\n\n"
            examples_str += f"RETRIEVED CONTEXTS:\n{rc}\n\n"
            examples_str += f"JSON:\n{verdicts.model_dump_json(indent=4)}\n\n"

        return f"""Your task is to determine whether each sentence in the expected output can be attributed to any information in the retrieval contexts.
Return a JSON object containing a key called 'verdicts' that contains an array of JSON objects that each contain a 'verdict' and a 'reason' key.
The `verdict` key should be either 'yes' or 'no'.
    - Answer 'yes' if the sentence can be attributed to any parts of the retrieval context, else answer 'no'.
Provide a brief reason that explains your verdict.

====== FEW SHOT EXAMPLES ======

{examples_str.strip()}

====== END OF EXAMPLES ======

**IMPORTANT:
- Return ONLY valid JSON with a "verdicts" list
- Keep reasons concise but specific
- The number of verdicts MUST EQUAL the number of sentences in the expected output
- DO NOT provide any further explanations or clarifications in your response (just JSON with verdicts)
**

Now verify if each claim in the expected output can be attributed to any information in the retrieval context:

EXPECTED OUTPUT:
"{expected_output}"

RETRIEVED CONTEXTS:
{retrieval_context}

JSON:
"""
