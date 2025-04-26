# pylint: disable=C0114
# pylint: disable=C0115
# pylint: disable=C0301

from typing import override, Optional, TypeVar

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
    instruction = (
        """Extract simple factual statements from the answer text. For each statement: 
1. express exactly one complete fact
2. avoid using pronouns
3. make it standalone without requiring context from other statements"""
    )
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
        )
    ]

    @override
    def to_string(self, data: Optional[StatementGeneratorInput] = None) -> str:
        return f"""## Task:
{self.instruction}
    
--- EXAMPLES: ---
QUESTION:
{self.examples[0][0].question}

ANSWER: 
{self.examples[0][0].answer}

OUTPUT:
{self.examples[0][1].model_dump_json(indent=4, exclude_none=True)}
{'-'*40}

**IMPORTANT:
1. Make sure the output is always in JSON format.
2. Each statement should be a complete, standalone claim without pronouns.
3. Break down complex sentences into multiple simple statements.
4. DO NOT provide any further explanations or clarifications, just output the JSON.
**

Now perform the same for the following:
QUESTION:
{data.question}

ANSWER:
{data.answer}

JSON:
"""
