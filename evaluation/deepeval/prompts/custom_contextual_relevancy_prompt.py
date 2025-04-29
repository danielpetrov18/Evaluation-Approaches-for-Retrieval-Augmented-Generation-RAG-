# pylint: disable=C0114
# pylint: disable=C0115
# pylint: disable=C0301
# pylint: disable=W0622

from deepeval.metrics.contextual_relevancy.template import ContextualRelevancyTemplate

class MyContextualRelevancyTemplate(ContextualRelevancyTemplate):

    @staticmethod
    def generate_verdicts(input: str, context: str):
        return f"""Your task is to determine for each statement in the context, whether or not it was remotely useful in arriving at the input.
Based on the input and context, please generate a JSON object to indicate whether each statement found in the context is relevant to the provided input.
The JSON will be a list of 'verdicts', with 2 mandatory fields: 'verdict' and 'statement', and 1 optional field: 'reason'.

Instructions:
1. First extract statements found in the context
2. For each extracted statement, determine whether it is relevant to the input
    - If so, the verdict key should be 'yes'
    - If not, the verdict key should be 'no' (provide a reason as to why it is not relevant and quote the irrelevant parts of the statement)
3. Do not use any of your prior knowledge, accept all the information at face value

**
IMPORTANT:
- Please make sure to only return in JSON format.
- DO NOT provide any further explanations or clarifications.
- Each statement should be a complete, standalone claim without pronouns.
- Break down complex sentences into multiple simple statements.

Example Context: "Einstein won the Nobel Prize for his discovery of the photoelectric effect. He won the Nobel Prize in 1968. There was a cat."
Example Input: "What were some of Einstein's achievements?"

Example:
{{
    "verdicts": [
        {{
            "verdict": "yes",
            "statement": "Einstein won the Nobel Prize for his discovery of the photoelectric effect in 1968",
        }},
        {{
            "verdict": "no",
            "statement": "There was a cat.",
            "reason": "The retrieval context contained the information 'There was a cat' when it has nothing to do with Einstein's achievements."
        }}
    ]
}}
**

Input:
{input}

Context:
{context}

OUTPUT JSON:
"""
