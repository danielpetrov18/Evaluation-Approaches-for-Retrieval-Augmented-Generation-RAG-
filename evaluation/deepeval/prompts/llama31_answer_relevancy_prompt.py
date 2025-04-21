# pylint: disable=C0114
# pylint: disable=C0301
# pylint: disable=W0622

from deepeval.metrics.answer_relevancy import AnswerRelevancyTemplate

class Llama31AnswerRelevancyTemplate(AnswerRelevancyTemplate):
    """Optimized template for answer relevancy metric with Llama 3.1 8B."""

    @staticmethod
    def generate_statements(actual_output: str):
        return f"""Extract all statements from this text. Return ONLY a JSON with "statements" as a list. Do not provide any further explanations.

Example text:
"Our laptop has a Retina display and 12-hour battery."

Example output:
{{
    "statements": [
        "The laptop has a Retina display.",
        "The laptop has a 12-hour battery."
    ]
}}

Text to analyze:
{actual_output}

JSON output:
"""

    @staticmethod
    def generate_verdicts(input: str, statements: str) -> str:
        return f"""For the provided list of statements, determine whether each statement is relevant to address the input.
Please generate a list of JSON with two keys: `verdict` and `reason`.

The 'verdict' key should STRICTLY be either a 'yes', 'idk' or 'no'. 
    - Answer 'yes' if the statement is relevant to addressing the original input 
    - Answer 'no' if the statement is irrelevant
    - Answer 'idk' if it is ambiguous (eg., not directly relevant but could be used as a supporting point to address the input).

The 'reason' is the reason for the verdict.
Provide a 'reason' ONLY if the answer is 'no'. 

IMPORTANT: The number of verdict objects MUST EQUAL the number of statements.

Example input: What features does the new laptop have?

Example statements:
[
    "The new laptop model has a high-resolution Retina display.",
    "It includes a fast-charging battery with up to 12 hours of usage.",
    "Security features include fingerprint authentication and an encrypted SSD.",
    "Every purchase comes with a one-year warranty.",
    "24/7 customer support is included.",
    "Pineapples taste great on pizza.",
    "The laptop might be useful for professionals."
]

Correct JSON response:
{{
    "verdicts": [
        {{
            "verdict": "yes"
        }},
        {{
            "verdict": "yes"
        }},
        {{
            "verdict": "yes"
        }},
        {{
            "verdict": "no",
            "reason": "A one-year warranty is a purchase benefit, not a feature of the laptop itself."
        }},
        {{
            "verdict": "no",
            "reason": "Customer support is a service, not a feature of the laptop."
        }},
        {{
            "verdict": "no",
            "reason": "The statement about pineapples on pizza is completely irrelevant to the input."
        }},
        {{
            "verdict": "idk"
        }}
    ]
}}

Input: {input}

Statements: {statements}

JSON:
"""
