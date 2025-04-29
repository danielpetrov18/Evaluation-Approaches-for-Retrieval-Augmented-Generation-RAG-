# pylint: disable=C0114
# pylint: disable=C0115
# pylint: disable=C0301
# pylint: disable=W0622

from typing import List, Optional
from deepeval.metrics.faithfulness import FaithfulnessTemplate

class MyFaithfulnessTemplate(FaithfulnessTemplate):

    @staticmethod
    def generate_claims(actual_output: str) -> str:
        return f"""Your task is to extract FACTUAL and UNDISPUTED truths from a provided text. The truths that you are to extract musn't be taken out of context.

EXAMPLE INPUT:
"Albert Einstein won the Nobel Prize in Physics in 1968 for his discovery of the photoelectric effect."

EXAMPLE OUTPUT:
{{
    "claims": [
        "Einstein won the Nobel Prize in Physics in 1968.",
        "Einstein discovered the photoelectric effect."
    ]
}}
===== END OF EXAMPLE ======

**
IMPORTANT:
- Please make sure to only return in JSON format, with the "claims" key as a list of strings.
- Accept the text at face value, DO NOT use any prior knowledge.
- Do not provide any further explanations or clarifications.
- Only include claims that are factual, BUT IT DOESN'T MATTER IF THEY ARE FACTUALLY CORRECT.
- The claims you extract should include the full context it was presented in, NOT cherry picked facts.
**

INPUT TEXT:
{actual_output}

JSON OUTPUT:
"""

    @staticmethod
    def generate_truths(retrieval_context: str, extraction_limit: Optional[int] = None) -> str:
        limit_text = "all factual truths" if extraction_limit is None else f"the {extraction_limit} most important factual truths"

        return f"""TASK: Extract {limit_text} from the text. They all must be COHERENT and inferred from the provided text.

EXAMPLE INPUT:
"Company X offers a 30-day refund policy for all customers."

EXAMPLE JSON:
{{
    "truths": [
        "Company X has a 30-day refund policy.",
        "The refund policy applies to all customers."
    ]
}}
===== END OF EXAMPLE ======

**
IMPORTANT:
- Please make sure to only return in JSON format, with the "truths" key as a list of strings.
- DO NOT use any prior knowledge, when extracting truths.
- Do not provide any further explanations or clarifications.
- Only include truths that are factual, BUT IT DOESN'T MATTER IF THEY ARE FACTUALLY CORRECT.
- Remember to extract {limit_text} that can inferred from the provided text.
**

TEXT:
{retrieval_context}

JSON OUTPUT:
"""

    @staticmethod
    def generate_verdicts(claims: List[str], retrieval_context: str) -> str:
        return f"""TASK: Determine if each claim is supported by the context.

EXAMPLE:
Example retrieval contexts: "Einstein won the Nobel Prize for his discovery of the photoelectric effect. Einstein won the Nobel Prize in 1968. Einstein is a German Scientist."
Example claims: ["Barack Obama is a caucasian male.", "Zurich is a city in London", "Einstein won the Nobel Prize for the discovery of the photoelectric effect which may have contributed to his fame.", "Einstein won the Nobel Prize in 1969 for his discovery of the photoelectric effect.", "Einstein was a Germen chef."]

Example OUTPUT:
{{
    "verdicts": [
        {{
            "verdict": "idk"
        }},
        {{
            "verdict": "idk"
        }},
        {{
            "verdict": "yes"
        }},
        {{
            "verdict": "no",
            "reason": "The actual output claims Einstein won the Nobel Prize in 1969, which is untrue as the retrieval context states it is 1968 instead."
        }},
        {{
            "verdict": "no",
            "reason": "The actual output claims Einstein is a Germen chef, which is not correct as the retrieval context states he was a German scientist instead."
        }},
    ]
}}
===== END OF EXAMPLE ======

**IMPORTANT:
- Return a JSON object with a "verdicts" key containing an array of objects
- Each verdict must have a "verdict" field
    - "yes" if the claim is supported by the context
    - "no" if the claim is directly contradicted by the context (include reason)
    - "idk" if it cannot be determined
- If the verdict is "no", include a "reason" field
- The number of verdict objects MUST EQUAL the number of claims.
- YOU SHOULD NEVER USE YOUR PRIOR KNOWLEDGE IN YOUR JUDGEMENT.
- Claims made using vague, suggestive, speculative language such as 'may have' does NOT count as a contradiction.
- Claims that is not backed up due to a lack of information/is not mentioned in the retrieval contexts MUST be answered 'idk'.
**

CONTEXT: {retrieval_context}

CLAIMS: {claims}

JSON OUTPUT:
"""
