# pylint: disable=C0114
# pylint: disable=C0115
# pylint: disable=C0301
# pylint: disable=W0622

import json

from typing import List, Final, Tuple, Union, Literal

from pydantic import BaseModel
from deepeval.metrics.answer_relevancy import AnswerRelevancyTemplate

class Statements(BaseModel):
    statements: List[str]

FEW_SHOT_EXAMPLES_STATEMENTS: Final[List[Tuple[str, Statements]]] = [
    (
        "Our laptop has a Retina display and 12-hour battery life.",
        Statements(
            statements=[
                "The laptop has a Retina display.",
                "The laptop has a 12-hour battery life."
            ]
        )
    ),
    (
        "I've been using this blender for a month now — it crushes ice in seconds and the cleaning function is a lifesaver!",
        Statements(
            statements=[
                "The blender can crush ice in seconds.",
                "The blender has a cleaning function.",
                "The blender has been used for a month."
            ]
        )
    ),
    (
        "",
        Statements(
            statements=[]
        )
    )
]

class Verdict(BaseModel):
    verdict: Literal["yes", "idk", "no"]
    reason: Union[str, None] = None # Reason is only required if the verdict is 'no'

class Verdicts(BaseModel):
    verdicts: List[Verdict]

FEW_SHOT_EXAMPLES_VERDICTS: Final[List[Tuple[str, str, Verdicts]]] = [
    # INPUT, STATEMENTS, VERDICTS
    (
        "What should I consider when buying a used car?",
        """[
    "Check the vehicle history report for accidents or damage.",
    "Have a mechanic inspect the car before purchasing.",
    "Consider the car's mileage and age.",
    "The first car was invented in 1886 by Karl Benz.",
    "Negotiate the price based on market value and condition.",
    "Car loans typically have interest rates between 3-10% depending on credit score."
]""",
        Verdicts(
            verdicts=[
                Verdict(verdict="yes"),
                Verdict(verdict="yes"),
                Verdict(verdict="yes"),
                Verdict(verdict="no", reason="Historical information about the first car is not relevant to buying a used car today."),
                Verdict(verdict="yes"),
                Verdict(verdict="idk")
            ]
        )
    )
]

class MyAnswerRelevancyTemplate(AnswerRelevancyTemplate):

    @staticmethod
    def generate_statements(actual_output: str):
        examples_str: str = ""
        for i, (example_input, example_output) in enumerate(FEW_SHOT_EXAMPLES_STATEMENTS, 1):
            examples_str += f"EXAMPLE {i}:\n"
            examples_str += f"TEXT:\n\"{example_input}\"\n\n"
            examples_str += f"JSON:\n{example_output.model_dump_json(indent=4)}\n\n"

        return f"""You are a helpful assistant that extracts statements from a provided text.
Ambiguous statements and single words can also be considered as statements.
Try not to use pronouns in the statements, and instead use the noun or name that is being referred to.
All statements should be inferred from the text and should be presented in a JSON format, with a key "statements" mapping to a list of strings.

Provide your answer in the following JSON format:
{{
    "statements": [
        "<statement 1>",
        "<statement 2>",
        // More statements...
]
}}

====== FEW SHOT EXAMPLES ======

{examples_str.strip()}

====== END OF EXAMPLES ======

**IMPORTANT:
- Return ONLY a JSON output, with the "statements" key mapping to an array of strings
- Do not provide any further explanations or clarifications, just the JSON output
**

Extract the statements from the following text:

TEXT:
{actual_output}

JSON:
"""

    @staticmethod
    def generate_verdicts(input: str, statements: str) -> str:
        examples_str: str = ""
        for i, (example_input, example_statements, example_verdicts) in enumerate(FEW_SHOT_EXAMPLES_VERDICTS, 1):
            examples_str += f"EXAMPLE {i}:\n"
            examples_str += f"INPUT:\n\"{example_input}\"\n\n"
            examples_str += f"STATEMENTS:\n{example_statements}\n\n"
            examples_str += f"JSON:\n{MyAnswerRelevancyTemplate._clean_verdict_json(example_verdicts)}\n\n"

        return f"""Your task is to determine for each statement, whether or not it is relevant to address the input.
A relevant statement should be a direct response to the input, should stay on-topic and should fully answer the input question.
Please generate a list of JSON with a key `verdicts` that maps to an array of verdicts.
The `verdict` key should STRICTLY be either a 'yes', 'idk' or 'no'.
- 'yes' if the statement is relevant to addressing the original input
- 'no' if the statement is irrelevant
- 'idk' if it is ambiguous (eg., not directly relevant but could be used as a supporting point to address the input).

Please provide the verdicts in the following JSON format:
{{
    "verdicts": [
        {{
            "verdict": "<yes | idk | no>",
            "reason": "<reason>" (optional, only if verdict is 'no')
        }},
        {{
            "verdict": "<yes | idk | no>",
            "reason": "<reason>" ()
        }},
        // More verdicts...
    ]
}}

====== FEW SHOT EXAMPLES ======

{examples_str.strip()}

====== END OF EXAMPLES ======

**IMPORTANT:
- Return ONLY a JSON output, with the "verdicts" key mapping to an array of strings
- The number of verdict objects MUST EQUAL the number of statements.
- Do not provide any further explanations or clarifications, just the JSON output
**

Evaluate the relevancy of each statement for answering this user question:

QUESTION:
{input}

STATEMENTS:
{statements}

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
