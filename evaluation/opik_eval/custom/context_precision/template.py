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
        input="What is the capital of France?",
        contexts=[
            "Paris is the capital of France and has been since 1792.",
            "France is located in Western Europe and has a population of about 67 million people.",
            "The Eiffel Tower is located in Paris and was completed in 1889."
        ],
        expected_output="The capital of France is Paris.",
        verdicts=["yes", "no", "no"],
        reasons=[
            "This context explicitly states that Paris is the capital of France, which directly answers the question.",
            "This context provides information about France but does not mention its capital, so it cannot be used to answer the question.",
            "While this context mentions Paris, it does not state that Paris is the capital of France, so it cannot be used to directly answer the question."
        ]
    ),
    FewShotExampleContextPrecision(
        input="What is the boiling point of water?",
        contexts=[
            "Water boils at 100 degrees Celsius at standard pressure.",
            "H2O is the chemical formula for water, which is essential for life on Earth.",
            "The freezing point of water is 0 degrees Celsius, while its boiling point is 100 degrees Celsius."
        ],
        expected_output="The boiling point of water is 100 degrees Celsius at standard pressure.",
        verdicts=["yes", "no", "yes"],
        reasons=[
            "This context explicitly states the boiling point of water at standard pressure, directly answering the question.",
            "While this context provides information about water and is factually accurate, it does not mention its boiling point, so it cannot be used to answer the question.",
            "This context explicitly mentions that the boiling point of water is 100 degrees Celsius, directly answering the question."
        ]
    ),
]

CONTEXT_PRECISION_TEMPLATE: Final[Type[str]] = """You are an expert judge with EXTREMELY STRICT criteria for evaluating whether each context node is directly useful for arriving at the expected output.

EXTREMELY STRICT EVALUATION CRITERIA:
1. A context node is ONLY relevant (verdict: "yes") if it EXPLICITLY and LITERALLY contains the specific information needed to directly answer the input question.
2. A context node is NOT relevant (verdict: "no") if:
   - It contains related information but doesn't explicitly answer the question
   - It mentions entities from the answer but doesn't state the relationship requested in the question
   - It requires ANY inference or knowledge to reach the expected output
   - It contains only partial information that alone cannot produce the expected output
   - It does not LITERALLY contain the answer in the exact form required
3. Being "about" the same topic is NEVER sufficient for relevance
4. Indirect support or related facts are NEVER sufficient for relevance
5. The information must be EXPLICITLY and LITERALLY stated in that exact context to be considered relevant
6. DO NOT hallucinate or infer information not literally present in the context
7. READ EACH CONTEXT WORD FOR WORD and verify that it contains the specific answer

Your task is to evaluate each context node INDEPENDENTLY and determine if it ALONE contains the EXPLICIT and LITERAL information needed to answer the question. DO NOT combine information from different contexts.

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
