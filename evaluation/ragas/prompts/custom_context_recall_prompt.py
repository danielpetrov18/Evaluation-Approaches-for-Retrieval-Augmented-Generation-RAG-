# pylint: disable=C0114
# pylint: disable=C0115
# pylint: disable=C0301

from typing import override, Optional, TypeVar

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
    instruction: str = "Given a context, and an answer, analyze each sentence in the answer and classify if the sentence can be attributed to the given context or not."
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

    @override
    def to_string(self, data: Optional[InputModel] = None) -> str:
        return f"""Task:
{self.instruction}

Example:
Input:
{self.examples[0][0].question}

Context:
{self.examples[0][0].context}

Answer:
{self.examples[0][0].answer}

Output:
{self.examples[0][1].model_dump_json(indent=4, exclude_none=True)}

**IMPORTANT:
1. Make sure the output is always in JSON format.
2. Each output object should contain a key "attributed"
    - If the statement can be attributed to the context, the value of "attributed" should be 1.
    - If the statement cannot be attributed to the context, the value of "attributed" should be 0.
3. Each output object should contain an additional key "reason" that provides a reason for the classification.
4. Each output object should contain a further key "statement" that provides the statement being classified.
5. DO NOT provide any further explanations or clarifications, just output the JSON.
**

Now perform the same for the following:
Input:
{data.question}

Context:
{data.context}

Answer:
{data.answer}

JSON:
"""
