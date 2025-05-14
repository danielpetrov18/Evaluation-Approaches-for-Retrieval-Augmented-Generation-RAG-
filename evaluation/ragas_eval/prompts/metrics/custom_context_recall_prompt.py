# pylint: disable=C0114
# pylint: disable=C0115
# pylint: disable=C0301

from typing import Optional, TypeVar

from pydantic import BaseModel
from ragas.prompt import PydanticPrompt
from ragas.metrics._context_recall import (
    QCA,
    ContextRecallClassification,
    ContextRecallClassifications
)

InputModel = TypeVar("InputModel", bound=BaseModel)

class MyContextRecallPrompt(PydanticPrompt[QCA, ContextRecallClassifications]):
    name: str = "custom_context_recall_classification"
    instruction: str = """Given a question, context, and an answer, analyze each claim in the answer and classify each claim as either attributed to the context or not.
If the claim can be attributed to the context, return 1; otherwise, return 0.
Additionally, provide a reason for the classification and the statement being classified.
The output should be a JSON object containing the following keys:
- "classifications": A list of dictionaries, where each dictionary contains the following keys:
    - "statement": The claim being classified.
    - "reason": A reason for the classification.
    - "attributed": 1 if the claim can be attributed to the context, and 0 otherwise."""
    input_model = QCA
    output_model = ContextRecallClassifications
    examples = [
        (
            QCA(
                question="What are some achievements of Albert Einstein?",
                context="Albert Einstein was a renowned physicist who developed the theory of relativity. His mass-energy equivalence formula, E = mc^2, became one of the most famous equations in physics. In 1921, he won the Nobel Prize in Physics for his work on the photoelectric effect, a key contribution to quantum theory.",
                answer="Albert Einstein developed the theory of relativity and won the Nobel Prize in 1921 for his work on the photoelectric effect. He also invented the first electric motor.",
            ),
            ContextRecallClassifications(
                classifications=[
                    ContextRecallClassification(
                        statement="Albert Einstein developed the theory of relativity.",
                        reason="The context clearly mentions Einstein's work on the theory of relativity.",
                        attributed=1
                    ),
                    ContextRecallClassification(
                        statement="He won the Nobel Prize in 1921 for his work on the photoelectric effect.",
                        reason="The context explicitly states that he won the Nobel Prize for the photoelectric effect.",
                        attributed=1
                    ),
                    ContextRecallClassification(
                        statement="He also invented the first electric motor.",
                        reason="The context does not mention anything about Einstein inventing an electric motor.",
                        attributed=0
                    ),
                ]
            )
        ),
    ]

    def to_string(self, data: Optional[InputModel] = None) -> str:
        examples_str: str = ""
        for i, (ex_input, ex_output) in enumerate(self.examples, 1):
            examples_str += f"EXAMPLE {i}:\n"
            examples_str += f"INPUT:\n{ex_input.model_dump_json(indent=4, exclude_none=True)}\n\n"
            examples_str += f"OUTPUT:\n{ex_output.model_dump_json(indent=4, exclude_none=True)}\n\n"

        input_obj: str = (
            data.model_dump_json(indent=4, exclude_none=True)
            if data is not None
            else "Input: (None)"
        )

        return f"""{self.instruction}

======= FEW SHOT EXAMPLES: =======
{examples_str}
======= END OF EXAMPLES =======

**IMPORTANT:
1. Make sure the output is always in JSON format.
2. DO NOT provide any further explanations or clarifications, just output the JSON.
3. Do not use any other knowledge you may have been trained on, accept all the information from the context at face value.
**

Now classify each claim in the answer as either attributed to the context or not:

INPUT:
{input_obj}

JSON:
"""
