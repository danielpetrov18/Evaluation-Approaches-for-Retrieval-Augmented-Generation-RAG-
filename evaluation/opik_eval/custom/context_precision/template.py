# pylint: disable=C0114
# pylint: disable=C0115
# pylint: disable=C0116
# pylint: disable=C0301
# pylint: disable=W0622

from typing import List, Final, Type, Optional, Literal

from pydantic import BaseModel

class FewShotExampleContextPrecision(BaseModel):
    input: str
    contexts: List[str]
    expected_output: str
    verdicts: List[Literal["yes", "no"]]
    reasons: List[str]

FEW_SHOT_EXAMPLES: Final[List[Type[FewShotExampleContextPrecision]]] = [
    FewShotExampleContextPrecision(
        input="Who developed the theory of evolution?",
        contexts=[
            "Charles Darwin was an English naturalist and biologist known for his theory of evolution by natural selection. He published 'On the Origin of Species' in 1859.",
            "The process of evolution involves changes in the heritable characteristics of biological populations over successive generations.",
            "Alfred Russel Wallace was a British naturalist who independently conceived the theory of evolution through natural selection."
        ],
        expected_output="Charles Darwin developed the theory of evolution.",
        verdicts=["yes", "no", "no"],
        reasons=[
            "This context explicitly states that Charles Darwin is known for his theory of evolution, which directly supports the expected output.",
            "This context describes what evolution is but doesn't mention who developed the theory, so it's not useful for arriving at the expected output.",
            "While this context mentions another scientist related to evolution theory, it doesn't support the specific expected output about Charles Darwin."
        ]
    ),
    FewShotExampleContextPrecision(
        input="What is the tallest mountain in the world?",
        contexts=[
            "Mount Everest, located in the Mahalangur Himal sub-range of the Himalayas, is Earth's highest mountain above sea level at 8,848.86 meters.",
            "The Andes is the longest continental mountain range in the world, located in South America.",
            "K2 is the second-highest mountain on Earth, after Mount Everest, at 8,611 meters above sea level."
        ],
        expected_output="Mount Everest is the tallest mountain in the world.",
        verdicts=["yes", "no", "yes"],
        reasons=[
            "This context directly states that Mount Everest is Earth's highest mountain, which supports the expected output.",
            "This context discusses the Andes mountain range but doesn't provide information about the world's tallest mountain, so it's not useful for the expected output.",
            "This context mentions that K2 is the second-highest mountain after Mount Everest, which indirectly confirms that Everest is the tallest mountain in the world."
        ]
    ),
]

CONTEXT_PRECISION_TEMPLATE: Final[Type[str]] = """You are an expert judge evaluating whether each context node was remotely useful for arriving at the expected output based on the input question.

EVALUATION CRITERIA:
1. A context node is relevant (verdict: "yes") if it contains information that helps arrive at the expected output, even if only partially or indirectly.
2. A context node is NOT relevant (verdict: "no") if:
   - It contains completely unrelated information
   - It contradicts the expected output
   - It provides no value in answering the question
3. Do not provide any further explanations or clarifications.

Your task is to evaluate each context node and determine if it was even remotely useful in arriving at the expected output with respect to the input.

Provide your answer in the following JSON format:
{{
    "verdicts": [
        {{
            "verdict": "<yes or no>",
            "reason": "<your reason for the verdict>"
        }},
        // More verdicts for each context...
    ]
}}

===== FEW SHOT EXAMPLES =====

{examples_str}

=== END OF EXAMPLES ===

Now evaluate for each CONTEXT its usefulness for arriving at the EXPECTED OUTPUT:

INPUT:
{input}

CONTEXT:
{context}

EXPECTED OUTPUT:
{expected_output}

VERDICT:
"""

def generate_query(
    input: Type[str],
    expected_output: Type[str],
    context: List[Type[str]],
    few_shot_examples: Optional[List[Type[FewShotExampleContextPrecision]]] = None,
) -> str:
    # If the user doesn't provide examples of his own, use the default ones
    examples: List[Type[FewShotExampleContextPrecision]] = FEW_SHOT_EXAMPLES if few_shot_examples is None else few_shot_examples

    examples_str: Type[str] = "\n\n".join(
        [
            f"""EXAMPLE {i}:
    
INPUT:
{example.input}

CONTEXT:
{example.contexts}

OUTPUT:
{example.expected_output}

VERDICT:
{{
    "verdicts": [
        {{"verdict": "{example.verdicts[0]}", "reason": "{example.reasons[0]}"}},
        {{"verdict": "{example.verdicts[1]}", "reason": "{example.reasons[1]}"}},
        {{"verdict": "{example.verdicts[2]}", "reason": "{example.reasons[2]}"}}
    ]
}}"""
            for i, example in enumerate(examples, 1)
        ]
    )

    return CONTEXT_PRECISION_TEMPLATE.format(
        examples_str=examples_str.strip(),
        input=input,
        context=context,
        expected_output=expected_output,
    )
