# pylint: disable=C0114
# pylint: disable=C0115
# pylint: disable=C0116
# pylint: disable=C0301
# pylint: disable=W0622

from typing import List, Final, Type, Optional

from pydantic import BaseModel

class FewShotExampleAnswerRelevance(BaseModel):
    input: str
    output: str
    score: float
    reason: str

FEW_SHOT_EXAMPLES: Final[List[Type[FewShotExampleAnswerRelevance]]] = [
    FewShotExampleAnswerRelevance(
        input="What is the capital of France?",
        output="The capital of France is Paris.",
        score=1.0,
        reason="The answer directly addresses the question with accurate and complete information without any irrelevant content."
    ),

    FewShotExampleAnswerRelevance(
        input="What year was the light bulb invented?",
        output="The light bulb was invented in 1879 by Thomas Edison. Edison was born in 1847 in Ohio and filed over 1,000 patents in his lifetime.",
        score=0.7,
        reason="The answer contains the correct information (1879) but includes unnecessary biographical details about Edison that weren't asked for."
    ),

    FewShotExampleAnswerRelevance(
        input="What are the three primary colors?",
        output="Red and blue are primary colors.",
        score=0.5,
        reason="The answer is on-topic and accurate but incomplete, as it only mentions two of the three primary colors (missing yellow)."
    )
]

ANSWER_RELEVANCE_TEMPLATE: Final[Type[str]] = """You are an expert judge tasked with evaluating the relevance of an AI-generated answer relative to the input.
Your task is to analyze the given answer and to determine if it's relevant to the input.

Guidelines:

1. Answer relevance is defined the following way:
    - There should be only information in the answer that is relevant to the input (on-topic).
    - The answer should be a direct response to the input (concise).
    - The answer should fully address the input (complete).

2. Analyze the OUTPUT thoroughly relative to the input and assign a relevance score between 0 and 1, where:
    - 0.0: The OUTPUT is entirely non-relevant to the INPUT
    - 1.0: The OUTPUT is entirely relevant to the INPUT (complete, concise, and on-topic)

3. It is crucial that you provide your answer in the following JSON format:
{{
    "score": <your score between 0.0 and 1.0>,
    "reason": <your reason for the score>
}}
  
===== FEW SHOT EXAMPLES =====

{examples_str}

=== END OF EXAMPLES ===

Now evaluate the relevance of the following OUTPUT relative to the INPUT:

INPUT:
{input}

OUTPUT:
{output}

VERDICT:
"""

def generate_query(
    input: str,
    output: str,
    few_shot_examples: Optional[List[FewShotExampleAnswerRelevance]] = None,
) -> str:
    # If the user doesn't provide examples of his own, use the default ones
    examples: Type[List[FewShotExampleAnswerRelevance]] = FEW_SHOT_EXAMPLES if few_shot_examples is None else few_shot_examples

    examples_str: Type[str] = "\n\n".join(
        [
            f"""EXAMPLE {i}:
    
INPUT:
{example.input}

OUTPUT:
{example.output}

VERDICT:
{{
    "score": "{example.score}",
    "reason": "{example.reason}"
}}"""
            for i, example in enumerate(examples, 1)
        ]
    )

    return ANSWER_RELEVANCE_TEMPLATE.format(
        examples_str=examples_str.strip(),
        input=input,
        output=output,
    )
