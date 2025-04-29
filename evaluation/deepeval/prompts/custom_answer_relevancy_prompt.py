# pylint: disable=C0114
# pylint: disable=C0115
# pylint: disable=C0301
# pylint: disable=W0622

from deepeval.metrics.answer_relevancy import AnswerRelevancyTemplate

class MyAnswerRelevancyTemplate(AnswerRelevancyTemplate):

    @staticmethod
    def generate_statements(actual_output: str):
        return f"""Your task is to analyze a text and extract statements from it.

Example text:
"Our laptop has a Retina display and 12-hour battery."

Example output:
{{
    "statements": [
        "The laptop has a Retina display.",
        "The laptop has a 12-hour battery."
    ]
}}

===== END OF EXAMPLE ======

**IMPORTANT:
- Ambiguous statements and single words can also be considered as statements
- Return ONLY a JSON output, with the "statements" key mapping to an array of strings
- Do not provide any further explanations or clarifications.  
**

Analyze the following text and extract statements:
{actual_output}

JSON output:
"""

    @staticmethod
    def generate_verdicts(input: str, statements: str) -> str:
        return f"""Your task is to determine for each statement, whether or not it is relevant to address the input. Please generate a list of JSON with two keys: `verdict` and `reason`.

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

**IMPORTANT:
* The 'verdict' key should STRICTLY be either a 'yes', 'idk' or 'no'.
    - Answer 'yes' if the statement is relevant to addressing the original input
    - Answer 'no' if the statement is irrelevant
    - Answer 'idk' if it is ambiguous (eg., not directly relevant but could be used as a supporting point to address the input).
* The 'reason' is the justification for the verdict. Provide one ONLY if the answer is 'no'.
* The number of verdict objects MUST EQUAL the number of statements.
**

Now analyze the following input and statements and generate the correct JSON response:
Input: {input}

Statements: {statements}

JSON:
"""
