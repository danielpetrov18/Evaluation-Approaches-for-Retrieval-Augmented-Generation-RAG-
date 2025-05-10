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
from ragas.testset.transforms.extractors.llm_based import Keyphrases, TextWithExtractionLimit

InputModel = TypeVar("InputModel", bound=BaseModel)


class MyKeyphrasesExtractorPrompt(PydanticPrompt[TextWithExtractionLimit, Keyphrases]):
    name: str = "custom_keyphrases_extractor_prompt"
    instruction: str = "Extract the top `max_num` keyphrases from the provided text. All the keyphrases should be derived from the text."
    input_model: Type[TextWithExtractionLimit] = TextWithExtractionLimit
    output_model: Type[Keyphrases] = Keyphrases
    examples: List[Tuple[TextWithExtractionLimit, Keyphrases]] = [
        (
            TextWithExtractionLimit(
                text="Artificial intelligence\n\nArtificial intelligence is transforming various industries by automating tasks that previously required human intelligence. From healthcare to finance, AI is being used to analyze vast amounts of data quickly and accurately. This technology is also driving innovations in areas like self-driving cars and personalized recommendations.",
                max_num=5,
            ),
            Keyphrases(
                keyphrases=[
                    "Artificial intelligence",
                    "automating tasks",
                    "healthcare",
                    "self-driving cars",
                    "personalized recommendations",
                ]
            ),
        ),
        (
            TextWithExtractionLimit(
                text="The sun is bright today. I decided to wear sunglasses when I went outside.",
                max_num=5,
            ),
            Keyphrases(
                keyphrases=[
                    "sun",
                    "bright",
                    "sunglasses",
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
2. Consider different forms or mentions of the same keyphrase as a single keyphrase.
3. DO NOT provide repeated keyphrases.
4. Never exceed the specified maximum number of keyphrases.
5. If the text doesn't contain as many keyphrases as the `max_num`, output as many keyphrases as possible without exceeding the `max_num`.
6. DO NOT provide any further explanations or clarifications, just output the JSON.
**

Now extract the keyphrases from the following text:

INPUT:
{input_obj}

JSON:
"""
