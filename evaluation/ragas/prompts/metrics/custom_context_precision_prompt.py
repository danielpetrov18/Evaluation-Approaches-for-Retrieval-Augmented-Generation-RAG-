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
        ),
        (
            QAC(
                question="What is the tallest mountain in the world?",
                context="The Andes is the longest continental mountain range in the world, located in South America. It stretches across seven countries and features many of the highest peaks in the Western Hemisphere. The range is known for its diverse ecosystems, including the high-altitude Andean Plateau and the Amazon rainforest.",
                answer="Mount Everest.",
            ),
            Verification(
                reason="The provided context discusses the Andes mountain range, which does not include Mount Everest or directly relate to the question about the world's tallest mountain.",
                verdict=0,
            ),
        )
    ]

    def to_string(self, data: Optional[InputModel] = None) -> str:
        return f"""{self.instruction}

======= EXAMPLES: =======
Example 1:
Question:
{self.examples[0][0].question}

Context:
{self.examples[0][0].context}

Answer:
{self.examples[0][0].answer}

Output:
{self.examples[0][1].model_dump_json(indent=4, exclude_none=True)}

Example 2:
Question:
{self.examples[1][0].question}

Context:
{self.examples[1][0].context}

Answer:
{self.examples[1][0].answer}

Output:
{self.examples[1][1].model_dump_json(indent=4, exclude_none=True)}
======= END OF EXAMPLES =======

**IMPORTANT:
1. Make sure the output is always in JSON format.
2. Each output object should contain a key "verdict"
    - If the context was remotely useful in arriving at the answer, the value of the "verdict" key should be 1.
    - If the context was not useful at all in arriving at the answer, the value of the "verdict" key should be 0.
3. Each output object should contain additional key "reason" that provides a reason for the verdict.
4. DO NOT provide any further explanations or clarifications, just output the JSON.
5. DO NOT use prior knowledge, accept all the information at face value.
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
