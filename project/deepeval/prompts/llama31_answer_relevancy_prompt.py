"""Custom template for answer relevancy respecting llama3.1 8B lesser capabilities."""

# pylint: disable=C0301
# pylint: disable=W0622

from typing import List
from deepeval.metrics.answer_relevancy import AnswerRelevancyTemplate

class Llama31AnswerRelevancyTemplate(AnswerRelevancyTemplate):
    """Optimized template for answer relevancy metric with Llama 3.1 8B."""

    @staticmethod
    def generate_statements(actual_output: str) -> str:
        """Uses very explicit instructions, clear examples, and repetitive formatting guidance."""

        return f"""Extract all individual statements from the text. Each sentence may contain multiple statements.

TEXT TO ANALYZE:
{actual_output}

INSTRUCTIONS:
1. Break the text into separate, distinct statements
2. Each statement should express a single idea or claim
3. Include both major and minor statements
4. Format your answer ONLY as JSON with a "statements" list

EXAMPLE:
Text: "Our laptop has a Retina display with crisp visuals. The battery lasts 12 hours and charges quickly in just 1 hour. It comes with advanced security features."

JSON output:
{{
  "statements": [
    "The laptop has a Retina display",
    "The Retina display has crisp visuals",
    "The battery lasts 12 hours",
    "The battery charges quickly",
    "Battery charging takes 1 hour",
    "It comes with advanced security features"
  ]
}}

YOUR TASK:
1. Create a similar list of statements for the text above
2. Return ONLY valid JSON
3. Do not include any explanations or notes
4. Make sure each statement is complete and makes sense on its own

OUTPUT JSON:
"""

    @staticmethod
    def generate_verdicts(input: str, statements: str) -> str:
        """Uses numbered instructions, repeated formatting guides, and simplified classification."""

        return f"""Evaluate if each statement is relevant to answering the question.

QUESTION: 
{input}

STATEMENTS TO EVALUATE:
{statements}

TASK:
For each statement, determine if it's:
- "yes" = directly relevant and helpful for answering the question
- "no" = irrelevant or off-topic
- "idk" = partially relevant or indirectly helpful

REQUIRED FORMAT:
Return a JSON object with a "verdicts" array containing one object per statement.
- For "yes" and "idk" verdicts: {{ "verdict": "yes" }} or {{ "verdict": "idk" }}
- For "no" verdicts only: {{ "verdict": "no", "reason": "brief explanation" }}

EXAMPLE:
Question: "What are the specs of the new phone?"
Statements: ["The phone has 8GB RAM", "The company was founded in 2010", "It has a 6-inch screen"]

Correct output:
{{
  "verdicts": [
    {{ "verdict": "yes" }},
    {{ "verdict": "no", "reason": "Company founding date is not a phone specification" }},
    {{ "verdict": "yes" }}
  ]
}}

IMPORTANT RULES:
1. Return exactly the same number of verdicts as statements
2. Always use the exact format shown in the example
3. Provide reasons ONLY for "no" verdicts
4. Keep your reasons very brief - just a few words
5. Output only the JSON - no other text

YOUR RESPONSE (JSON ONLY):
"""

    @staticmethod
    def generate_reason(irrelevant_statements: List[str], input: str, score: float) -> str:
        """Uses a fill-in-the-blank approach and specific structural guidance."""

        formatted_score = f"{score:.2f}"

        # Format the irrelevant statements list for better parsing
        formatted_irrelevant = "\n".join([f"- {stmt}" for stmt in irrelevant_statements]) if irrelevant_statements else "None"

        return f"""Create a brief explanation for the answer relevancy score.

QUESTION: {input}

SCORE: {formatted_score} (scale 0-1)

IRRELEVANT STATEMENTS:
{formatted_irrelevant}

TASK:
Explain why the score is {formatted_score} by mentioning:
1. What the response did well (if anything)
2. What irrelevant information was included (if any)
3. How directly the response addressed the question

ANSWER FORMAT:
{{
  "reason": "The score is {formatted_score} because [YOUR EXPLANATION]"
}}

TIPS FOR YOUR EXPLANATION:
- If score is high (>0.8): Focus on what made the answer relevant
- If score is medium (0.5-0.8): Explain both strengths and weaknesses
- If score is low (<0.5): Explain the main issues with relevance
- Keep explanation under 50 words
- Be specific and factual

YOUR ANSWER (JSON ONLY):
"""
