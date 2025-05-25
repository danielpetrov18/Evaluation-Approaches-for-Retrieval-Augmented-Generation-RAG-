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
        "Marie Curie discovered polonium and radium. She won two Nobel Prizes. Marie Curie developed the theory of relativity.",
        [
            "Marie Curie was a pioneer in the field of radioactivity.",
            "She discovered two new elements: polonium and radium.",
            "Marie Curie was awarded the Nobel Prize in Physics in 1903.",
            "In 1911, she won a second Nobel Prize, this time in Chemistry."
        ],
        Verdicts(
            verdicts=[
                Verdict(verdict="yes", reason="The second context explicitly states she discovered polonium and radium."),
                Verdict(verdict="yes", reason="The third and fourth contexts together support that she won two Nobel Prizes."),
                Verdict(verdict="no", reason="None of the contexts mention Marie Curie developing the theory of relativity.")
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

Respond in the following JSON format:
{{
    "verdicts": [
        {{
            "verdict": "<yes|no>",
            "reason": "<reason>"
        }},
        ...
    ]
}}

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
