# pylint: disable=C0114
# pylint: disable=C0115
# pylint: disable=C0301
# pylint: disable=W0622

from typing import List
from deepeval.metrics.hallucination import HallucinationTemplate

class MyHallucinationTemplate(HallucinationTemplate):

    @staticmethod
    def generate_verdicts(actual_output: str, contexts: List[str]):
        return f"""Task: For each context in contexts, please generate a list of JSON objects to indicate whether the given 'actual output' agrees with EACH context.

For each context, determine if the actual output CONTRADICTS it.
- Answer "yes" if the output agrees with or doesn't contradict the context
- Answer "no" if the output explicitly contradicts information in the context
- Only mark as contradiction if there's a clear factual conflict
- Missing details are NOT contradictions

**IMPORTANT:
* Output format: Return a JSON with "verdicts" as a list containing exactly {len(contexts)} items.
* Each verdict must have:
    - "verdict": "yes" or "no" 
    - "reason": explanation of your verdict
* Base your analysis ONLY on the contexts provided. Ignore any outside knowledge.
* DO NOT provide any further clarifications or exlanations.

Example:
Contexts: ["Einstein won Nobel Prize for photoelectric effect.", "Einstein won Nobel Prize in 1921."]
Output: "Einstein received the Nobel Prize in 1922 for the photoelectric effect."

JSON response:
{{
  "verdicts": [
    {{
      "verdict": "yes",
      "reason": "The output correctly states Einstein won for photoelectric effect."
    }},
    {{
      "verdict": "no",
      "reason": "The output says 1922, but context states 1921."
    }}
  ]
}}
**

Contexts: {contexts}
Actual Output: {actual_output}

JSON:
"""
