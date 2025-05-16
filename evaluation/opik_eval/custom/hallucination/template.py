# pylint: disable=C0114
# pylint: disable=C0115
# pylint: disable=C0116
# pylint: disable=C0301
# pylint: disable=W0622

from typing import List, Optional, Type, Final

from pydantic import BaseModel

class FewShotExampleHallucination(BaseModel):
    input: str
    output: str
    context: List[str]
    score: float
    reason: str

FEW_SHOT_EXAMPLES: Final[List[FewShotExampleHallucination]] = [
    FewShotExampleHallucination(
        input="What is the Statue of Liberty?",
        context=[
            "The Statue of Liberty is a copper monument in New York Harbor.",
            "It was a gift from France to the United States in 1886."
        ],
        output="The Statue of Liberty is a copper monument located in New York Harbor that was gifted to the United States from France in 1886.",
        score=0.0,
        reason="The output contains only information present in the context with no contradictions or additions."
    ),

    FewShotExampleHallucination(
        input="Tell me about tigers.",
        context=[
            "Tigers are the largest cat species with distinctive orange fur and black stripes.",
            "They are native to Asia and are apex predators."
        ],
        output="Tigers are the largest cat species, recognizable by their orange fur with black stripes. They live primarily in Africa and parts of Asia where they hunt as apex predators.",
        score=0.5,
        reason="The output correctly describes tigers' appearance and predator status but falsely states they live in Africa, which contradicts the context that states they are native to Asia only."
    ),
]

HALLUCINATION_TEMPLATE: Final[Type[str]] = """You are an expert judge tasked with evaluating the faithfulness of an AI-generated answer relative to the context.
Analyze the provided INPUT, CONTEXT, and OUTPUT to determine if the OUTPUT contains any hallucinations or unfaithful information.
Unfaithful information or hallucinations refer to information that contradicts information provided in the CONTEXT.

Guidelines:
1. The OUTPUT must not introduce new information beyond what's provided in the CONTEXT.
2. The OUTPUT must not contradict any information given in the CONTEXT.
3. Consider partial hallucinations where some information is correct but other parts are not.

Analyze the OUTPUT thoroughly and assign a hallucination score between 0 and 1, where:
- 0.0: The OUTPUT is entirely faithful to the CONTEXT
- 1.0: The OUTPUT is entirely unfaithful to the CONTEXT

It is crucial that you provide your answer in the following JSON format:
{{
    "score": <your score between 0.0 and 1.0>,
    "reason": <your reason for the score>
}}

===== FEW SHOT EXAMPLES =====

{examples_str}

=== END OF EXAMPLES ===

Now evaluate the faithfulness of the following OUTPUT relative to the CONTEXT:

INPUT (for context only, not to be used for faithfulness evaluation):
{input}

CONTEXT:
{context}

OUTPUT:
{output}

VERDICT:
"""

def generate_query(
    input: str,
    output: str,
    context: List[str],
    few_shot_examples: Optional[List[FewShotExampleHallucination]] = None,
) -> str:
    # If the user doesn't provide examples of his own, use the default ones
    examples: List[FewShotExampleHallucination] = FEW_SHOT_EXAMPLES if few_shot_examples is None else few_shot_examples

    examples_str: Type[str] = "\n\n".join(
        [
            f"""EXAMPLE {i}:
    
INPUT:
{example.input}

CONTEXT:
{example.context}

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

    return HALLUCINATION_TEMPLATE.format(
        examples_str=examples_str.strip(),
        input=input,
        context=context,
        output=output,
    )
