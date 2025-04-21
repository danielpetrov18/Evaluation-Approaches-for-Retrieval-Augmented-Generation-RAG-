# pylint: disable=C0114
# pylint: disable=C0301
# pylint: disable=W0622

from typing import List, Optional
from deepeval.metrics.faithfulness import FaithfulnessTemplate

class Llama31FaithfulnessTemplate(FaithfulnessTemplate):
    """Optimized template for faithfulness metric with Llama 3.1 8B."""

    @staticmethod
    def generate_claims(actual_output: str) -> str:
        return f"""TASK: Extract factual claims from the text below. Do not provide any further explanations.

OUTPUT FORMAT:
Return a JSON object with a single key "claims" containing an array of strings.

EXAMPLE INPUT:
"Albert Einstein won the Nobel Prize in Physics in 1968 for his discovery of the photoelectric effect."

EXAMPLE OUTPUT:
{{
    "claims": [
        "Einstein won the Nobel Prize in Physics in 1968.",
        "Einstein discovered the photoelectric effect."
    ]
}}

INPUT TEXT:
{actual_output}

JSON OUTPUT:
"""

    @staticmethod
    def generate_truths(retrieval_context: str, extraction_limit: Optional[int] = None) -> str:
        limit_text = "all factual truths" if extraction_limit is None else f"the {extraction_limit} most important factual truths"

        return f"""TASK: Extract {limit_text} from the text. Do not provide any further explanations.

OUTPUT FORMAT:
Return a JSON object with a single key "truths" containing an array of strings.

EXAMPLE INPUT:
"Company X offers a 30-day refund policy for all customers."

EXAMPLE JSON:
{{
    "truths": [
        "Company X has a 30-day refund policy.",
        "The refund policy applies to all customers."
    ]
}}

TEXT:
{retrieval_context}

JSON OUTPUT:
"""

    @staticmethod
    def generate_verdicts(claims: List[str], retrieval_context: str) -> str:
        return f"""TASK: Determine if each claim is supported by the context.

OUTPUT FORMAT: Return a JSON object with a "verdicts" key containing an array of verdict objects. 
Each verdict must have a "verdict" field
    - "yes" if the claim is supported by the context
    - "no" if the claim is directly contradicted by the context (include reason)
    - "idk" if it cannot be determined

IMPORTANT: The number of verdict objects MUST EQUAL the number of claims.

CONTEXT: {retrieval_context}

CLAIMS: {claims}

EXAMPLE INPUT:
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

JSON OUTPUT:
"""

    @staticmethod
    def generate_reason(score: float, contradictions: List[str]):
        return f"""TASK: Explain the faithfulness score based on contradictions found.

SCORE: {score} (0-1 scale, higher is better)

CONTRADICTIONS FOUND:
{contradictions}

INSTRUCTIONS:
1. Briefly explain why the output received this score.
2. Focus only on the contradictions listed above.
3. If no contradictions, provide encouraging feedback.
4. Return ONLY a JSON with a "reason" field.

FORMAT:
{{
    "reason": "The score is {score} because [your explanation]."
}}

JSON OUTPUT:
"""
