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
    instruction: str = """Given question, answer and context verify if the context was remotely useful in arriving at the given answer.
You should return a JSON object, containing a `verdict` key and a `reason` key.
The `verdict` key should be 1 if the context was remotely useful in arriving at the answer, and 0 if it was not useful at all.
Furthermore, a `reason` key should be provided that explains the verdict."""
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
        examples_str: str = ""
        for i, (ex_input, ex_output) in enumerate(self.examples, 1):
            examples_str += f"EXAMPLE {i}:\n"
            examples_str += f"INPUT:\n{ex_input.model_dump_json(indent=4, exclude_none=True)}\n\n"
            examples_str += f"OUTPUT:\n{ex_output.model_dump_json(indent=4, exclude_none=True)}\n\n"

        input_obj: str = (
            data.model_dump_json(indent=4, exclude_none=True)
            if data is not None
            else "Input: (None)\n"
        )

        return f"""{self.instruction}

======= FEW SHOT EXAMPLES: =======
{examples_str}
======= END OF EXAMPLES =======

**IMPORTANT:
1. Make sure the output is always in JSON format.
2. DO NOT provide any further explanations or clarifications, just output the JSON.
3. DO NOT use any prior knowledge, accept all the information from the context at face value.
**

Now determine if the context was useful in arriving at the answer relative to the question:

INPUT:
{input_obj}

JSON: 
"""
