# pylint: disable=C0114
# pylint: disable=C0115
# pylint: disable=C0301
# pylint: disable=W0622

import json
from typing import List, Optional, Final, Tuple, Literal

from pydantic import BaseModel
from deepeval.metrics.faithfulness import FaithfulnessTemplate

class Claims(BaseModel):
    claims: List[str]

FEW_SHOT_EXAMPLES_CLAIMS: Final[List[Tuple[str, Claims]]] = [
    (
        "The Eiffel Tower is located in Paris and was completed in 1889. It stands approximately 300 meters tall.",
        Claims(claims=[
            "The Eiffel Tower is located in Paris.",
            "The Eiffel Tower was completed in 1889.",
            "The Eiffel Tower stands approximately 300 meters tall."
        ])
    ),
    (
        "",
        Claims(claims=[])
    ),
]

class Truths(BaseModel):
    truths: List[str]

FEW_SHOT_EXAMPLES_TRUTHS: Final[List[Tuple[str, Truths]]] = [
    (
        """Apple Inc. was founded in 1976 by Steve Jobs, Steve Wozniak, and Ronald Wayne.\n
The company's first product was the Apple I computer, which was hand-built by Wozniak.\n
Apple went public in 1980 and has since become one of the most valuable companies in the world.""",
        Truths(
            truths=[
                "Apple Inc. was founded in 1976.",
                "Apple Inc. was founded by Steve Jobs, Steve Wozniak, and Ronald Wayne.",
                "Apple's first product was the Apple I computer.",
                "The Apple I computer was hand-built by Steve Wozniak.",
                "Apple went public in 1980.",
                "Apple is one of the most valuable companies in the world."
            ]
        )
    ),
    (
        "",
        Truths(
            truths=[]
        )
    )
]

class Verdict(BaseModel):
    verdict: Literal["yes", "no", "idk"]
    reason: Optional[str] = None

class Verdicts(BaseModel):
    verdicts: List[Verdict]

FEW_SHOT_EXAMPLES_VERDICTS: Final[List[Tuple[List[str], str, Verdicts]]] = [
    (
        # CLAIMS
        [
            "The Eiffel Tower is located in Berlin.",
            "The Eiffel Tower was completed in 1889.",
            "The Eiffel Tower may have been a gift from the United States.",
            "The Eiffel Tower stands approximately 250 meters tall."
        ],
        # RETRIEVAL CONTEXT
        "The Eiffel Tower is located in Paris and was completed in 1889.\n\nIt stands approximately 300 meters tall.",
        # EXPECTED VERDICTS
        Verdicts(
            verdicts=[
                Verdict(verdict="no", reason="The claim states the Eiffel Tower is located in Berlin, but the context states it is located in Paris."),
                Verdict(verdict="yes"),
                Verdict(verdict="idk"),
                Verdict(verdict="no", reason="The claim states the Eiffel Tower is 250 meters tall, but the context states it is approximately 300 meters tall.")
            ]
        )
    ),
]

class MyFaithfulnessTemplate(FaithfulnessTemplate):

    @staticmethod
    def generate_claims(actual_output: str) -> str:
        examples_str: str = ""
        for i, (example_input, example_output) in enumerate(FEW_SHOT_EXAMPLES_CLAIMS, 1):
            examples_str += f"EXAMPLE {i}:\n"
            examples_str += f"INPUT TEXT:\n\"{example_input}\"\n\n"
            examples_str += f"JSON:\n{example_output.model_dump_json(indent=4)}\n\n"

        return f"""Your task is to extract FACTUAL and UNDISPUTED truths from a provided text.
The truths that you are to extract musn't be taken out of context and should not contradict any of the data in the provided text.
You should return a JSON object with a "claims" key containing an array of strings.

====== FEW SHOT EXAMPLES ======

{examples_str.strip()}

====== END OF EXAMPLES ======

**
IMPORTANT:
- Please make sure to only return in JSON format, with the "claims" key as a list of strings.
- Accept the text at face value, DO NOT use any prior knowledge.
- Do not provide any further explanations or clarifications.
**

INPUT TEXT:
"{actual_output}"

JSON:
"""

    @staticmethod
    def generate_truths(retrieval_context: str, extraction_limit: Optional[int] = None) -> str:
        limit_text = "all factual truths" if extraction_limit is None else f"the {extraction_limit} most important factual truths"

        examples_str: str = ""
        for i, (example_input, example_output) in enumerate(FEW_SHOT_EXAMPLES_TRUTHS, 1):
            examples_str += f"EXAMPLE {i}:\n"
            examples_str += f"INPUT TEXT:\n\"{example_input}\"\n\n"
            examples_str += f"JSON:\n{example_output.model_dump_json(indent=4)}\n\n"

        return f"""Your task is to extract {limit_text} from the text. They all must be COHERENT and inferred from the provided text.

====== FEW SHOT EXAMPLES ======

{examples_str.strip()}

====== END OF EXAMPLES ======

**
IMPORTANT:
- Please make sure to only return in JSON format, with the "truths" key as a list of strings.
- DO NOT use any prior knowledge, when extracting truths.
- Do not provide any further explanations or clarifications.
- Only include truths that are factual, BUT IT DOESN'T MATTER IF THEY ARE FACTUALLY CORRECT.
- Remember to extract {limit_text} that can inferred from the provided text.
**

Now extract {limit_text} from the text.

INPUT TEXT:
"{retrieval_context}"

JSON:
"""

    @staticmethod
    def generate_verdicts(claims: List[str], retrieval_context: str) -> str:
        examples_str: str = ""
        for i, (example_claims, example_context, example_verdicts) in enumerate(FEW_SHOT_EXAMPLES_VERDICTS, 1):
            examples_str += f"EXAMPLE {i}:\n"
            examples_str += f"CLAIMS:\n{example_claims}\n\n"
            examples_str += f"CONTEXT:\n\"{example_context}\"\n\n"
            examples_str += f"JSON:\n{MyFaithfulnessTemplate._clean_verdict_json(example_verdicts)}\n\n"

        return f"""Your task is to determine if EACH claim is factually consistent with the context.
Factually consistent claims are those that do not explicitly contradict the context.
Return a JSON object with a `verdicts` key containing an array of objects.
Claims need to be classified as either 'yes', 'no', or 'idk', depending on whether the claim is consistent with facts from the retrieval context.
If a claim contradicts the context, it should be classified as 'no'.
If a claim doesn't contradict the context, it should be classified as 'yes'.
If the claim cannot be determined, it should be classified as 'idk'.
If the answer is no, a `reason` is to be provided, otherwise NO `reason` is to be provided.

====== FEW SHOT EXAMPLES ======

{examples_str.strip()}

====== END OF EXAMPLES ======

**IMPORTANT:
- Return a JSON object with a `verdicts` key containing an array of objects
- The number of verdict objects MUST EQUAL the number of claims.
- YOU SHOULD NEVER USE YOUR PRIOR KNOWLEDGE IN YOUR JUDGEMENT.
- Claims made using vague, suggestive, speculative language such as 'may have' does NOT count as a contradiction.
- Claims that is not backed up due to a lack of information/is not mentioned in the retrieval contexts MUST be answered 'idk'.
**

Please classify the claims as 'yes', 'no', or 'idk':

CONTEXT:
"{retrieval_context}"

CLAIMS:
{claims}

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
