# pylint: disable=C0114
# pylint: disable=C0115
# pylint: disable=C0301

from typing import (
    TypeVar,
    Optional,
    Type,
    List,
    Tuple,
)

from pydantic import BaseModel
from ragas.prompt import PydanticPrompt
from ragas.testset.persona import Persona
from ragas.testset.synthesizers.single_hop.prompts import QueryCondition, GeneratedQueryAnswer

InputModel = TypeVar("InputModel", bound=BaseModel)

class QueryAnswerGenerationPrompt(PydanticPrompt[QueryCondition, GeneratedQueryAnswer]):
    instruction: str = (
        "Generate a single-hop query and answer based on the specified conditions (persona, term, style, length) "
        "and the provided context. Ensure the answer is entirely faithful to the context, using only the information "
        "directly from the provided context."
        "### Instructions:\n"
        "1. **Generate a Query**: Based on the context, persona, term, style, and length, create a question "
        "that aligns with the persona's perspective and incorporates the term.\n"
        "2. **Generate an Answer**: Using only the content from the provided context, construct a detailed answer "
        "to the query. Do not add any information not included in or inferable from the context.\n"
    )
    input_model: Type[QueryCondition] = QueryCondition
    output_model: Type[GeneratedQueryAnswer] = GeneratedQueryAnswer
    examples: List[Tuple[QueryCondition, GeneratedQueryAnswer]] = [
        (
            QueryCondition(
                persona=Persona(
                    name="Software Engineer",
                    role_description="Focuses on coding best practices and system design.",
                ),
                term="microservices",
                query_style="Formal",
                query_length="Medium",
                context="Microservices are an architectural style where applications are structured as a collection of loosely coupled services. "
                "Each service is fine-grained and focuses on a single functionality.",
            ),
            GeneratedQueryAnswer(
                query="What is the purpose of microservices in software architecture?",
                answer="Microservices are designed to structure applications as a collection of loosely coupled services, each focusing on a single functionality.",
            ),
        ),
    ]

    def to_string(self, data: Optional[InputModel] = None) -> str:
        examples_str = ""
        for i, (ex_input, ex_output) in enumerate(self.examples, 1):
            examples_str += f"EXAMPLE {i}:\n"
            examples_str += f"INPUT:\n{ex_input.model_dump_json(indent=4, exclude_none=True)}\n\n"
            examples_str += f"OUTPUT: {ex_output.model_dump_json(indent=4, exclude_none=True)}\n\n"

        input_obj: str = (
            data.model_dump_json(indent=4, exclude_none=True)
            if data is not None
            else "Input: (None)\n"
        )

        return f"""{self.instruction}

======= FEW SHOT EXAMPLES: =======
{examples_str}
======= END OF EXAMPLES =======

**IMPORTANT:
1. Make sure the output is always in JSON format.
2. Do not include any other text in the output.
3. The output should have a key "headlines" containing a list of unique level 2 and level 3 headlines.
4. Never exceed the specified maximum number of headlines.
5. DO NOT provide any further explanations or clarifications, just output the JSON.
**

Now perform the same for the following:

INPUT:
{input_obj}

JSON:
"""
