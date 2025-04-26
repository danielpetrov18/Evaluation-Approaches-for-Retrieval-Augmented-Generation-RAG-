# pylint: disable=C0114
# pylint: disable=C0115
# pylint: disable=C0301

from typing import Optional, TypeVar

from pydantic import BaseModel
from ragas.prompt import PydanticPrompt
from ragas.metrics._context_precision import QAC, Verification

InputModel = TypeVar("InputModel", bound=BaseModel)

class MyContextPrecisionPrompt(PydanticPrompt[QAC, Verification]):
    name: str = "custom_context_precision"
    instruction: str = 'Given question, answer and context verify if the context was useful in arriving at the given answer.'
    input_model = QAC
    output_model = Verification
    examples = [
        (
            QAC(
                question="Who developed the theory of evolution?",
                context="Charles Darwin was an English naturalist and biologist known for his theory of evolution by natural selection. He published 'On the Origin of Species' in 1859, laying the foundation for evolutionary biology.",
                answer="Charles Darwin developed the theory of evolution.",
            ),
            Verification(
                reason="The context clearly identifies Charles Darwin as the developer of the theory of evolution, which supports the answer fully.",
                verdict=1,
            ),
        )
    ]

    def to_string(self, data: Optional[InputModel] = None) -> str:
        return f"""## Task:
{self.instruction}

--- EXAMPLES: ---
Question:
{self.examples[0][0].question}

Context:
{self.examples[0][0].context}

Answer:
{self.examples[0][0].answer}

Output:
{self.examples[0][1].model_dump_json(indent=4, exclude_none=True)}
{'-'*40}

**IMPORTANT:
1. Make sure the output is always in JSON format.
2. Each output object should contain a key "verdict"
    - If the context was useful in arriving at the answer, the value of the "verdict" key should be 1.
    - If the context was not useful in arriving at the answer, the value of the "verdict" key should be 0.
3. Each output object should contain additional key "reason" that provides a reason for the verdict.
4. DO NOT provide any further explanations or clarifications, just output the JSON.
**

Now perform the same for the following:
Question:
{data.question}

Context:
{data.context}

Answer:
{data.answer}

JSON: 
"""
