# pylint: disable=C0114
# pylint: disable=C0115
# pylint: disable=C0301
# pylint: disable=W0622

from typing import List, Final, Tuple, Literal

from pydantic import BaseModel
from deepeval.metrics.contextual_precision import ContextualPrecisionTemplate

class Verdict(BaseModel):
    verdict: str = Literal["yes", "no"]
    reason: str

class Verdicts(BaseModel):
    verdicts: List[Verdict]

FEW_SHOT_EXAMPLES: Final[List[Tuple[str, str, List[str], Verdicts]]] = [
    (
        # Input
        "What are the symptoms of diabetes?",
        # Expected Output
        "Common symptoms of diabetes include increased thirst, frequent urination, extreme fatigue, and blurred vision.",
        # Retrieved Context
        [
            "Diabetes is a chronic condition that affects the way the body processes blood sugar (glucose).",
            "Symptoms include increased thirst, frequent urination, extreme fatigue, and blurred vision.",
            "Treatment includes diet, exercise, medication, and insulin therapy."
        ],
        # Verdicts
        Verdicts(
            verdicts=[
                Verdict(verdict="yes", reason="Mentions that diabetes affects blood sugar, which is relevant to understanding the condition."),
                Verdict(verdict="yes", reason="Directly lists the symptoms that appear in the expected output."),
                Verdict(verdict="no", reason="Focuses on treatment methods, not symptoms.")
            ]
        )
    ),
]

class MyContextualPrecisionTemplate(ContextualPrecisionTemplate):

    @staticmethod
    def generate_verdicts(
        input: str, expected_output: str, retrieval_context: List[str]
    ):
        examples_str: str = ""
        for i, (user_input, output, context, verdicts) in enumerate(FEW_SHOT_EXAMPLES, 1):
            examples_str += f"EXAMPLE {i}:\n"
            examples_str += f"QUESTION:\n\"{user_input}\"\n\n"
            examples_str += f"EXPECTED OUTPUT:\n\"{output}\"\n\n"
            examples_str += f"RETRIEVED CONTEXT:\n{context}\n\n"
            examples_str += f"JSON:\n{verdicts.model_dump_json(indent=4)}\n\n"

        return f"""Your task is to determine for each node in the context, whether or not it was remotely useful in arriving at the expected output with respect to the input.
Please return a JSON object containing a key called 'verdicts' that contains an array of JSON objects that each contain a 'verdict' and a 'reason' key.
The `verdict` key should be either 'yes' or 'no'.
- Answer 'yes' if the context was even remotely useful in arriving at the expected output.
- Answer 'no' otherwise.
The `reason` key should justify the verdict. In your reason, you should aim to quote parts of the context.

====== FEW SHOT EXAMPLES ======

{examples_str.strip()}

====== END OF EXAMPLES ======

**
IMPORTANT: 
- Please make sure to only return in JSON format, with the 'verdicts' key as a list of JSON. 
- Do not provide any further explanations or clarifications, just output the JSON.
- Since you are going to generate a verdict for each context, the number of 'verdicts' SHOULD BE STRICTLY EQUAL to that of the contexts.
**

Now generate a list of JSONs to indicate whether EACH context is remotely useful in arriving at the expected output with respect to the input:

QUESTION:
"{input}"

EXPECTED OUTPUT:
"{expected_output}"

RETRIEVED CONTEXT:
{retrieval_context}

JSON:
"""
