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
    instruction = """Your task is to evaluate the faithfulness of each statement with respect to a given context.
A statement is unfaithful if it:
1. **Contradicts** any part of the context.
2. **Introduces information** that is not present in the context.

For each statement:
- If it is fully supported by the context, mark it as **faithful** with a `verdict` of 1.
- If it contradicts or adds new, unsupported information, mark it as **unfaithful** with a `verdict` of 0.

Return your answer as a JSON object with this format:
{
    "statements": [
        {
            "statement": "<original statement>",
            "verdict": 0 or 1,
            "reason": "<clear explanation for the verdict>"
        },
        // additional statements
        ...
    ]
}"""

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
                        reason="This statement contradicts the context, which explicitly states that John is majoring in Computer Science.",
                        verdict=0,
                    ),
                    StatementFaithfulnessAnswer(
                        statement="John is taking a course on Artificial Intelligence.",
                        reason="The statement introduces a course not present in the context (Artificial Intelligence), making it unsupported.",
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

{examples_str.strip()}

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
