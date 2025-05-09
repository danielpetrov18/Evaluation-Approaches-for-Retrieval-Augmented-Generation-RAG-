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
    instruction = "Generate a question for the given answer and identify if answer is noncommittal."
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
                response="""I don't know about the  groundbreaking feature of the smartphone invented in 2023 as am unaware of information beyond 2022. """,
            ),
            ResponseRelevanceOutput(
                question="What was the groundbreaking feature of the smartphone invented in 2023?",
                noncommittal=1,
            ),
        ),
    ]

    def to_string(self, data: Optional[ResponseRelevanceInput] = None) -> str:
        return f"""{self.instruction}

======= EXAMPLES: =======
Answer:
{self.examples[0][0].response}

Output:
{self.examples[0][1].model_dump_json(indent=4, exclude_none=True)}

Answer:
{self.examples[1][0].response}

Output:
{self.examples[1][1].model_dump_json(indent=4, exclude_none=True)}
======= END OF EXAMPLES =======

**IMPORTANT:
1. Make sure the output is always in JSON format.
2. Clasify each response as noncommittal or committal.
    - A noncommittal answer is one that is evasive, vague, or ambiguous. For example, "I don't know" or "I'm not sure" are noncommittal answers.
    - If the response is noncommittal, the value of the "noncommittal" key should be 1.
    - If the response is committal, the value of the "noncommittal" key should be 0.
3. Each output object should contain an additional key "question" that provides a question for the response.
4. DO NOT provide any further explanations or clarifications, just output the JSON.
**

Now perform the same for the following:

Answer:
{data.response}

JSON:
"""
