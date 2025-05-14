# pylint: disable=C0114
# pylint: disable=C0115
# pylint: disable=C0301

from typing import Optional, TypeVar

from pydantic import BaseModel
from ragas.prompt import PydanticPrompt
from ragas.metrics._faithfulness import (
    StatementGeneratorInput,
    StatementGeneratorOutput
)

InputModel = TypeVar("InputModel", bound=BaseModel)

class MyStatementGeneratorPrompt(
    PydanticPrompt[StatementGeneratorInput, StatementGeneratorOutput]
):
    instruction = """Given a question and answer pair break down the answer into one or more fully understandable statements.
Each statement should be a complete, standalone claim without pronouns and factually consistent with the answer.
The statements should be clear and concise, and they should not contain any unnecessary information."""
    input_model = StatementGeneratorInput
    output_model = StatementGeneratorOutput
    examples = [
        (
            StatementGeneratorInput(
                question="Who was Albert Einstein and what is he best known for?",
                answer="He was a German-born theoretical physicist, widely acknowledged to be one of the greatest and most influential physicists of all time. He was best known for developing the theory of relativity, he also made important contributions to the development of the theory of quantum mechanics.",
            ),
            StatementGeneratorOutput(
                statements=[
                    "Albert Einstein was a German-born theoretical physicist.",
                    "Albert Einstein is recognized as one of the greatest and most influential physicists of all time.",
                    "Albert Einstein was best known for developing the theory of relativity.",
                    "Albert Einstein also made important contributions to the development of the theory of quantum mechanics.",
                ]
            ),
        ),
        (
            StatementGeneratorInput(
                question="What is water?",
                answer="Water is a transparent, tasteless, odorless liquid that is essential for all known forms of life."
            ),
            StatementGeneratorOutput(
                statements=[
                    "Water is a transparent, tasteless, odorless liquid essential for all known forms of life."
                ]
            )
        )
    ]

    def to_string(self, data: Optional[StatementGeneratorInput] = None) -> str:
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
2. Each statement should be a complete, standalone claim without pronouns.
3. Break down complex sentences into multiple simple statements.
4. DO NOT provide any further explanations or clarifications, just output the JSON.
**

Now decompose the answer into one or more fully understandable statements:

INPUT:
{input_obj}

JSON:
"""
