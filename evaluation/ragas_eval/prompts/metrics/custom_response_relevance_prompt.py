# pylint: disable=C0114
# pylint: disable=C0115
# pylint: disable=C0301

from typing import Optional, TypeVar

from pydantic import BaseModel
from ragas.prompt import PydanticPrompt
from ragas.metrics._answer_relevance import (
    ResponseRelevanceInput,
    ResponseRelevanceOutput
)

InputModel = TypeVar("InputModel", bound=BaseModel)

class MyResponseRelevancePrompt(
    PydanticPrompt[ResponseRelevanceInput, ResponseRelevanceOutput]
):
    instruction = """Generate a question for the given answer and identify if the answer is noncommittal.
A noncommittal answer is one that is evasive, vague, or ambiguous. For example, "I don't know" or "I'm not sure" are noncommittal answers.
If the response is noncommittal, return 1; otherwise, return 0.
Additionally, provide a hypothetical question for the response, which can be inferred from the response itself.
The output should be a JSON object containing the following keys:
- "question": A hypothetical question for the response.
- "noncommittal": 1 if the response is noncommittal, and 0 otherwise.
The output should be in JSON format.
"""
    input_model = ResponseRelevanceInput
    output_model = ResponseRelevanceOutput
    examples = [
        (
            ResponseRelevanceInput(
                response="Albert Einstein was born in Germany.",
            ),
            ResponseRelevanceOutput(
                question="Where was Albert Einstein born?",
                noncommittal=0,
            ),
        ),
        (
            ResponseRelevanceInput(
                response="I don't know about the groundbreaking feature of the smartphone invented in 2023 as am unaware of information beyond 2022.",
            ),
            ResponseRelevanceOutput(
                question="What was the groundbreaking feature of the smartphone invented in 2023?",
                noncommittal=1,
            ),
        ),
    ]

    def to_string(self, data: Optional[ResponseRelevanceInput] = None) -> str:
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
**

Now come up with a hypothetical question for the response and classify it as noncommittal or committal.

INPUT:
{input_obj}

JSON:
"""
