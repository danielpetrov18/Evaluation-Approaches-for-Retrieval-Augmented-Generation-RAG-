# pylint: disable=C0114
# pylint: disable=C0115
# pylint: disable=C0301

from typing import override, Optional, TypeVar

from pydantic import BaseModel
from ragas.prompt import PydanticPrompt
from ragas.metrics._answer_correctness import (
    QuestionAnswerGroundTruth,
    ClassificationWithReason,
    StatementsWithReason
)

InputModel = TypeVar("InputModel", bound=BaseModel)

class MyCorrectnessClassifier(
    PydanticPrompt[QuestionAnswerGroundTruth, ClassificationWithReason]
):
    instruction = (
        """Given a list of ground truths and a list of answer statements, analyze each statement and classify them in one of the following categories: 
TP (true positive): statements that are present in answer that are also directly supported by one or more statements in the ground truth 
FP (false positive): statements present in the answer but not directly supported by any statement in ground truth
FN (false negative): statements found in the ground truth but not present in the answer.
Each statement can only belong to one of the categories. 
Provide a reason for each classification."""
    )

    input_model = QuestionAnswerGroundTruth
    output_model = ClassificationWithReason
    examples = [
        (
            QuestionAnswerGroundTruth(
                question="What is the boiling point of water?",
                answer=[
                    "The boiling point of water is 100 degrees Celsius at sea level."
                ],
                ground_truth=[
                    "The boiling point of water is 100 degrees Celsius (212 degrees Fahrenheit) at sea level.",
                    "The boiling point of water can change with altitude.",
                ],
            ),
            ClassificationWithReason(
                TP=[
                    StatementsWithReason(
                        statement="The boiling point of water is 100 degrees Celsius at sea level",
                        reason="This statement is directly supported by the ground truth which specifies the boiling point of water as 100 degrees Celsius at sea level.",
                    )
                ],
                FP=[],
                FN=[
                    StatementsWithReason(
                        statement="The boiling point of water can change with altitude.",
                        reason="This additional information about how the boiling point of water can vary with altitude is not mentioned in the answer.",
                    )
                ],
            ),
        ),
    ]

    @override
    def to_string(self, data: Optional[QuestionAnswerGroundTruth] = None) -> str:
        return f"""## Task:
{self.instruction}

--- EXAMPLES: ---
Question:
{self.examples[0][0].question}

Answer:
{self.examples[0][0].answer}

Ground Truth:
{self.examples[0][0].ground_truth}

Output:
{self.examples[0][1].model_dump_json(indent=4, exclude_none=True)}
{'-'*40}

**IMPORTANT:
1. Make sure the output is always in JSON format.
2. Remember to classify statements properly as mentioned in the instruction.
3. Each output object should contain an additional key "reason" that provides a reason for the classification.
4. DO NOT provide any further explanations or clarifications, just output the JSON.
**

Now perform the same for the following:
Question:
{data.question}

Answer:
{data.answer}

Ground Truth:
{data.ground_truth}

JSON:
"""
