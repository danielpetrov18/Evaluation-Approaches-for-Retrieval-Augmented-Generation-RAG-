# pylint: disable=C0114
# pylint: disable=C0115
# pylint: disable=C0116
# pylint: disable=C0301
# pylint: disable=W0622

from typing import List, Final, Optional

from pydantic import BaseModel

from .models import Statement, ContextRecallVerdict

# This will be used to decompose the `expected_output` into separate claims
class FewShotExampleStatements(BaseModel):
    expected_output: str        # Input
    statements: List[Statement] # Output

FEW_SHOT_EXAMPLES_STATEMENTS: Final[List[FewShotExampleStatements]] = [
    FewShotExampleStatements(
        expected_output="Python is a popular programming language. It is widely used in data science.",
        statements=[
            Statement(text="Python is a popular programming language."),
            Statement(text="Python is widely used in data science."),
        ]
    ),
    FewShotExampleStatements(
        expected_output="Despite facing significant resistance from within her own party, the senator managed to pass the climate reform bill, which allocates billions towards renewable energy and aims to cut national emissions by 40 percent over the next decade.",
        statements=[
            Statement(text="The senator faced significant resistance from within her own party."),
            Statement(text="The senator managed to pass the climate reform bill."),
            Statement(text="The climate reform bill allocates billions towards renewable energy."),
            Statement(text="The climate reform bill aims to cut national emissions by 40 percent over the next decade."),
        ]
    ),
    FewShotExampleStatements(
        expected_output="",
        statements=[]
    )
]

TEXT_DECOMPOSITION_TEMPLATE: Final[str] = """You are a helpful assistant that specializes in breaking down complex or multi-part texts into individual, complete, and standalone claims. You DO NOT need to over-split into fine details; aim for meaningful, medium-grained statements.

RULES:
1. Decompose each sentence into standalone, atomic statements.
2. Do not include overlapping or duplicate statements.
3. Avoid breaking down sentences into overly fine details (e.g., word-by-word).
4. Preserve meaning and clarity.

Provide your output in the following JSON format:
{{
    "statements": [
        {{"text": "<statement 1>"}},
        {{"text": "<statement 2>"}},
        ...
    ]
}}

===== FEW SHOT EXAMPLES =====

{examples_str}

=== END OF EXAMPLES ===

Now decompose the following text:

TEXT:
{expected_output}

DECOMPOSED STATEMENTS:
"""

def generate_decomposition_query(
    expected_output: str,
    few_shot_examples: Optional[
        List[FewShotExampleStatements]
    ] = None
) -> str:
    examples: List[FewShotExampleStatements] = (
        FEW_SHOT_EXAMPLES_STATEMENTS if few_shot_examples is None else few_shot_examples
    )

    examples_str = "\n\n".join(
        [
            f"""EXAMPLE {i}:

TEXT:
{example.expected_output}

DECOMPOSED STATEMENTS::
{{
    "statements": [
        {',\n        '.join([f'{{"text": "{stmt.text}"}}' for stmt in example.statements])}
    ]
}}"""
            for i, example in enumerate(examples, 1)
        ]
    )

    return TEXT_DECOMPOSITION_TEMPLATE.format(
        examples_str=examples_str.strip(),
        expected_output=expected_output.strip()
    )

# `input` + `contexts` + `statements` is what we submit to the model
# `verdicts` is the output classifying each statement
class FewShotExampleContextRecall(BaseModel):
    input: str
    contexts: List[str]
    statements: List[Statement]          # The statements will be inferred from the `expected output`
    verdicts: List[ContextRecallVerdict] # Output containing all the statements classified as `yes` or `no`

FEW_SHOT_EXAMPLE_CONTEXT_RECALL: Final[List[FewShotExampleContextRecall]] = [
    FewShotExampleContextRecall(
        input="Tell me about Marie Curie's scientific contributions.",
        contexts=[
            "Marie Curie was a pioneer in the field of radioactivity.",
            "She discovered two new elements: polonium and radium.",
            "Marie Curie was awarded the Nobel Prize in Physics in 1903.",
            "In 1911, she won a second Nobel Prize, this time in Chemistry.",
            "Her work laid the foundation for nuclear physics."
        ],
        statements=[
            Statement(text="Marie Curie discovered polonium and radium."),
            Statement(text="Marie Curie developed the theory of relativity."),
            Statement(text="Marie Curie won two Nobel Prizes.")
        ],
        verdicts=[
            ContextRecallVerdict(
                statement=Statement(text="Marie Curie discovered polonium and radium."),
                attributed=True,
                reason="The second context explicitly states: 'She discovered two new elements: polonium and radium.'"
            ),
            ContextRecallVerdict(
                statement=Statement(text="Marie Curie developed the theory of relativity."),
                attributed=False,
                reason="None of the contexts mention the theory of relativity or connect it to Marie Curie."
            ),
            ContextRecallVerdict(
                statement=Statement(text="Marie Curie won two Nobel Prizes."),
                attributed=True,
                reason="The third and fourth contexts explicitly state she won Nobel Prizes in 1903 and 1911, making it clear she won two."
            )
        ]
    )
]

CONTEXT_RECALL_TEMPLATE: Final[str] = """You are an expert judge evaluating whether each statement derived from an expected output can be attributed to at least one of the CONTEXTS.

EVALUATION CRITERIA:
1. A statement is deemed attributed if it can be reasonably supported by one of the CONTEXTS.
2. A statement is deemed unattributed if no context contains information to support it.
3. Do not use any prior knowledge and accept the information from the CONTEXTS at face value.
4. Read the CONTEXTS and STATEMENTS carefully.
5. THE NUMBER OF VERDICTS MUST BE THE SAME AS THE NUMBER OF STATEMENTS.

Your task is to evaluate each statement in relation to the list of CONTEXTS, and decide if any of the contexts contain the information needed to support that statement.

Respond in the following JSON format:
{{
    "verdicts": [
        {{
            "statement": {{"text": "<statement>"}},
            "attributed": <true|false>,
            "reason": "<brief justification based only on the literal context match>"
        }},
        ...
    ]
}}

===== FEW SHOT EXAMPLES =====

{examples_str}

=== END OF EXAMPLES ===

Now evaluate:

INPUT:
{input}

CONTEXTS:
{contexts}

STATEMENTS:
{statements}

VERDICTS:
"""

def generate_query(
    input: str,
    statements: List[Statement],
    context: List[str],
    few_shot_examples: Optional[
        List[FewShotExampleContextRecall]
    ] = None,
) -> str:
    examples: List[FewShotExampleContextRecall] = (
        FEW_SHOT_EXAMPLE_CONTEXT_RECALL if few_shot_examples is None else few_shot_examples
    )

    examples_str: str = "\n\n".join(
        [
            f"""EXAMPLE {i}:

INPUT:
{example.input}

CONTEXTS:
[
    {',\n    '.join([f'"{c}"' for c in example.contexts])}
]

STATEMENTS:
[
    {',\n    '.join([f'{{"text": "{s.text}"}}' for s in example.statements])}
]

VERDICTS:
{{
    "verdicts": [
        {',\n        '.join([
            f'''{{
            "statement": {{"text": "{v.statement.text}"}},
            "attributed": {str(v.attributed).lower()},
            "reason": "{v.reason}"
        }}''' for v in example.verdicts])}
    ]
}}"""
            for i, example in enumerate(examples, 1)
        ]
    )

    # Format input `context` and `statements` the same way
    formatted_contexts = "[\n    " + ",\n    ".join(f'"{c}"' for c in context) + "\n]"
    formatted_statements = "[\n    " + ",\n    ".join(f'{{"text": "{s.text}"}}' for s in statements) + "\n]"

    return CONTEXT_RECALL_TEMPLATE.format(
        examples_str=examples_str.strip(),
        input=input.strip(),
        contexts=formatted_contexts,
        statements=formatted_statements
    )
