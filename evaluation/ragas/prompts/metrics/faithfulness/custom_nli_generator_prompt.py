# pylint: disable=C0114
# pylint: disable=C0115
# pylint: disable=C0301

from typing import Optional, TypeVar

from pydantic import BaseModel
from ragas.prompt import PydanticPrompt
from ragas.metrics._faithfulness import (
    NLIStatementInput,
    NLIStatementOutput,
    StatementFaithfulnessAnswer
)

InputModel = TypeVar("InputModel", bound=BaseModel)

class MyNLIStatementPrompt(
    PydanticPrompt[NLIStatementInput, NLIStatementOutput]
):
    instruction = """Your task is to judge the faithfulness of a series of statements based on a given context.
Verify for each statement, whether or not it can be inferred from a context.
If a statement cannot be directly inferred from the context, it is considered unfaithful, and you should assign it a verdict of 0.
If a statement can be remotely inferred from the context, it is considered faithful, and you should assign it a verdict of 1.
Return a JSON object containing the following fields:
- `statements`: A list of objects, each containing:
    - `statement`: The statement being classified.
    - `verdict`: 1 if the statement can be inferred from the context, 0 otherwise.
    - `reason`: A reason for the verdict.
The output should be in JSON format.
"""
    input_model = NLIStatementInput
    output_model = NLIStatementOutput
    examples = [
        (
            NLIStatementInput(
                context="""John is a student at XYZ University. He is pursuing a degree in Computer Science. He is enrolled in several courses this semester, including Data Structures, Algorithms, and Database Management. John is a diligent student and spends a significant amount of time studying and completing assignments.""",
                statements=[
                    "John is majoring in Biology.",
                    "John is taking a course on Artificial Intelligence.",
                    "John is a dedicated student.",
                ],
            ),
            NLIStatementOutput(
                statements=[
                    StatementFaithfulnessAnswer(
                        statement="John is majoring in Biology.",
                        reason="John's major is explicitly mentioned as Computer Science. There is no information suggesting he is majoring in Biology.",
                        verdict=0,
                    ),
                    StatementFaithfulnessAnswer(
                        statement="John is taking a course on Artificial Intelligence.",
                        reason="The context mentions the courses John is currently enrolled in, and Artificial Intelligence is not mentioned. Therefore, it cannot be deduced that John is taking a course on AI.",
                        verdict=0,
                    ),
                    StatementFaithfulnessAnswer(
                        statement="John is a dedicated student.",
                        reason="The context states that he spends a significant amount of time studying and completing assignments. Additionally, it mentions that he often stays late in the library to work on his projects, which implies dedication.",
                        verdict=1,
                    ),
                ]
            ),
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
3. Do not use any other knowledge you may have been trained on.
**

Now judge the faithfulness of the following statements based on the context:

INPUT:
{input_obj}

JSON:
"""
