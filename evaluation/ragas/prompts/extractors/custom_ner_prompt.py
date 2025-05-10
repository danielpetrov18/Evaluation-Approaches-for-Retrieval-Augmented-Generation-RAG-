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
from ragas.testset.transforms.extractors.llm_based import NEROutput, TextWithExtractionLimit

InputModel = TypeVar("InputModel", bound=BaseModel)

class MyNERPrompt(PydanticPrompt[TextWithExtractionLimit, NEROutput]):
    name: str = "custom_ner_extractor_prompt"
    instruction: str = ("""Extract the named entities from the given text, limiting the output to the top entities as specified by the `max_num` parameter.
Focus on proper nouns such as people, organizations, locations, products, and events.
For people, include their full names when available.
If entities appear multiple times, only include them once in the results."""
    )

    input_model: Type[TextWithExtractionLimit] = TextWithExtractionLimit
    output_model: Type[NEROutput] = NEROutput

    examples: List[Tuple[TextWithExtractionLimit, NEROutput]] = [
        (
            TextWithExtractionLimit(
                text="""The United Nations Climate Change Conference, also known as COP26, was held in Glasgow, Scotland.
                Prime Minister Boris Johnson and President Joe Biden attended, along with representatives from over 100 countries.
                Microsoft and Apple announced new environmental initiatives during the event.""",
                max_num=8,
            ),
            NEROutput(
                entities=[
                    "United Nations Climate Change Conference",
                    "COP26",
                    "Glasgow",
                    "Scotland",
                    "Boris Johnson",
                    "Joe Biden",
                    "Microsoft",
                    "Apple"
                ]
            ),
        ),
        (
            TextWithExtractionLimit(
                text="",
                max_num=5,
            ),
            NEROutput(
                entities=[]
            ),
        ),
        (
            TextWithExtractionLimit(
                text="The Louvre Museum in Paris houses the Mona Lisa, painted by Leonardo da Vinci.",
                max_num=3,
            ),
            NEROutput(
                entities=[
                    "Louvre Museum",
                    "Paris",
                    "Mona Lisa"
                ]
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
2. Consider different forms or mentions of the same entity as a single entity.
3. The output should have a key "entities" containing a list of unique entities, where each entity is a string.
4. Never exceed the specified maximum number of entities.
5. If the text contains less entities than the `max_num`, output as many entities as possible without exceeding the `max_num`.
6. DO NOT provide any further explanations or clarifications, just output the JSON.
**

Now extract the named entities from the following text:

INPUT:
{input_obj}

JSON:
"""
