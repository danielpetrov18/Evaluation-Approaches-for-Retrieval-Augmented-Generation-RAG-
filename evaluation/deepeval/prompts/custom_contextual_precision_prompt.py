# pylint: disable=C0114
# pylint: disable=C0115
# pylint: disable=C0301
# pylint: disable=W0622

from typing import List
from deepeval.metrics.contextual_precision import ContextualPrecisionTemplate

class MyContextualPrecisionTemplate(ContextualPrecisionTemplate):

    @staticmethod
    def generate_verdicts(
        input: str, expected_output: str, retrieval_context: List[str]
    ):
        return f"""Your task is to determine for each node in the context, whether or not it was remotely useful in arriving at the expected output with respect to the input.

**
IMPORTANT: 
- Please make sure to only return in JSON format, with the 'verdicts' key as a list of JSON. 
- Each `verdict` key can have 2 values:
    - Answer 'yes' if the context was even remotely useful in arriving at the expected output.
    - Answer 'no' otherwise
- There should also be a `reason` key to justify the verdict. In your reason, you should aim to quote parts of the context.
- Do not provide any further explanations or clarifications, just output the JSON.

Example Retrieval Context: ["Einstein won the Nobel Prize for his discovery of the photoelectric effect", "He won the Nobel Prize in 1968.", "There was a cat."]
Example Input: "Who won the Nobel Prize in 1968 and for what?"
Example Expected Output: "Einstein won the Nobel Prize in 1968 for his discovery of the photoelectric effect."

Example:
{{
    "verdicts": [
        {{
            "verdict": "yes",
            "reason": "It clearly addresses the question by stating that 'Einstein won the Nobel Prize for his discovery of the photoelectric effect.'"
        }},
        {{
            "verdict": "yes",
            "reason": "The text verifies that the prize was indeed won in 1968."
        }},
        {{
            "verdict": "no",
            "reason": "'There was a cat' is not at all relevant to the topic of winning a Nobel Prize."
        }}
    ]
}}
Since you are going to generate a verdict for each context, the number of 'verdicts' SHOULD BE STRICTLY EQUAL to that of the contexts.
**

Now generate a list of JSONs to indicate whether EACH context is remotely useful in arriving at the expected output with respect to the input:

Input:
{input}

Expected output:
{expected_output}

Retrieval Context:
{retrieval_context}

JSON:
"""
