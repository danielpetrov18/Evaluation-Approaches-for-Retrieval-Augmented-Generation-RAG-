# pylint: disable=C0114
# pylint: disable=C0115
# pylint: disable=C0301
# pylint: disable=W0622

from typing import List
from deepeval.metrics.contextual_recall import ContextualRecallTemplate

class Llama31ContextualRecallTemplate(ContextualRecallTemplate):

    @staticmethod
    def generate_verdicts(expected_output: str, retrieval_context: List[str]) -> str:
        formatted_context = "\n".join([f"Node {i+1}: {ctx}" for i, ctx in enumerate(retrieval_context)])

        return f"""You are evaluating the quality of a retrieval system. Your task is to determine whether each sentence in the expected output can be attributed to any information in the retrieval context.

Instructions:
1. For each sentence, decide if it can be attributed to any part of the retrieval context.
2. Verdict must be EXACTLY "yes" or "no" - nothing else.
    - Answer 'yes' if the sentence can be attributed to any parts of the retrieval context, else answer 'no'.
3. Provide a brief reason that explains your verdict.
4. In your reason, mention which specific node number(s) in the retrieval context support your verdict.

Example expected output:
"The product costs $50. It includes a 30-day warranty. Shipping takes 3-5 business days."

Example retrieval context:
Node 1: "Our product is priced at $50 with free shipping that takes 3-5 business days."
Node 2: "All products include a 30-day limited warranty and a satisfaction guarantee."

Example response:
{{
    "verdicts": [
        {{
            "verdict": "yes",
            "reason": "The sentence 'The product costs $50' is directly supported by Node 1 which states 'Our product is priced at $50...'"
        }},
        {{
            "verdict": "yes",
            "reason": "The sentence 'It includes a 30-day warranty' is supported by Node 2 which mentions 'All products include a 30-day limited warranty...'"
        }},
        {{
            "verdict": "yes",
            "reason": "The sentence 'Shipping takes 3-5 business days' is supported by Node 1 which states '...shipping that takes 3-5 business days.'"
        }}
    ]
}}

**IMPORTANT:
- Return ONLY valid JSON with a "verdicts" list
- For each sentence in the expected output, include exactly one verdict
- Each verdict must be either "yes" or "no" (lowercase)
- Keep reasons concise but specific
- The number of verdicts MUST EQUAL the number of sentences in the expected output
- DO NOT provide any further explanations or clarifications in your response (just JSON with verdicts)
**

Expected Output:
{expected_output}

Retrieval Context:
{formatted_context}

OUTPUT JSON:
"""
