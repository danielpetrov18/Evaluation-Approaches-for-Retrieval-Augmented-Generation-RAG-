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
    instruction = "Your task is to judge the faithfulness of a series of statements based on a given context."
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
        return f"""### Task:
{self.instruction}

--- EXAMPLES: ---
Context:
{self.examples[0][0].context}

Statements:
{self.examples[0][0].statements}

Output:
{self.examples[0][1].model_dump_json(indent=4, exclude_none=True)}
{'-'*40}

**IMPORTANT:
1. Make sure the output is always in JSON format.
2. Each output object should contain a key "verdict"
- If the statement can be directly inferred based on the context, the value of the "verdict" key should be 1.
- If the statement can not be directly inferred based on the context, the value of the "verdict" key should be 0.
3. Each output object should contain an additional key "reason" that provides a reason for the verdict.
4. Each output object should contain a further key "statement" that provides the statement being classified.
5. DO NOT provide any further explanations or clarifications, just output the JSON.
**

Now perform the same for the following:
Context:
{data.context}

Statements:
{data.statements}

JSON:
"""
