"""Custom template for faithfulness respecting llama3.1 8B lesser capabilities."""

# pylint: disable=C0301
# pylint: disable=W0622

from typing import List, Optional
from deepeval.metrics.faithfulness import FaithfulnessTemplate

class Llama31FaithfulnessTemplate(FaithfulnessTemplate):
    """Optimized template for faithfulness metric with Llama 3.1 8B."""

    @staticmethod
    def generate_claims(actual_output: str):
        return f"""TASK: Extract factual claims from the text below.

INSTRUCTIONS:
1. Read the text carefully.
2. Identify statements presented as facts.
3. List each claim as a separate string.
4. Return ONLY a JSON with a "claims" list.

FORMAT:
Return a JSON object with a single key "claims" containing an array of strings.

EXAMPLE INPUT:
"Albert Einstein won the Nobel Prize in Physics in 1968 for his discovery of the photoelectric effect."

EXAMPLE OUTPUT:
{{
    "claims": [
        "Einstein won the Nobel Prize in Physics in 1968.",
        "Einstein won the Nobel Prize for discovering the photoelectric effect."
    ]
}}

INPUT TEXT:
{actual_output}

JSON OUTPUT:
"""

    @staticmethod
    def generate_truths(retrieval_context: str, extraction_limit: Optional[int] = None) -> str:
        limit_text = "all factual truths" if extraction_limit is None else f"the {extraction_limit} most important factual truths"

        return f"""TASK: Extract {limit_text} from the text.

TEXT:
{retrieval_context}

INSTRUCTIONS:
1. Read the text carefully.
2. Identify key factual statements.
3. List each fact as a separate string.
4. Return ONLY a JSON with "truths" list.

EXAMPLE 1:
"Company X offers a 30-day refund policy for all customers."

EXAMPLE OUTPUT 1:
{{
    "truths": [
        "Company X has a 30-day refund policy.",
        "The refund policy applies to all customers."
    ]
}}

EXAMPLE 2:
"The new application requires 4GB of RAM and works on Windows 10."

EXAMPLE OUTPUT 2:
{{
    "truths": [
        "The application needs 4GB of RAM.",
        "The application is compatible with Windows 10."
    ]
}}

JSON OUTPUT:
"""

    @staticmethod
    def generate_verdicts(claims: List[str], retrieval_context: str):
        return f"""TASK: Check if each claim matches information in the context.

CONTEXT:
{retrieval_context}

CLAIMS TO VERIFY:
{claims}

INSTRUCTIONS:
1. For each claim, determine if it's supported by the context.
2. Respond ONLY with "yes" (supported), "no" (contradicted), or "idk" (can't determine).
3. If "no", provide a short correction based ONLY on the context.
4. Return a JSON with "verdicts" containing your assessments.

FORMAT:
{{
    "verdicts": [
        {{ "verdict": "yes" }},
        {{ "verdict": "no", "reason": "Correction based on context" }},
        {{ "verdict": "idk" }}
    ]
}}

NOTE: Only say "no" if directly contradicted by the context. If uncertain, use "idk".

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

OUTPUT:
"""
