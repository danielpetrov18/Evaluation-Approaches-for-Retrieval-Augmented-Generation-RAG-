# pylint: disable=C0114
# pylint: disable=C0115
# pylint: disable=C0301

from typing import Optional, TypeVar

from pydantic import BaseModel
from ragas.prompt import PydanticPrompt, StringIO
from ragas.metrics._context_entities_recall import EntitiesList

InputModel = TypeVar("InputModel", bound=BaseModel)

class MyContextEntitiesRecallPrompt(PydanticPrompt[StringIO, EntitiesList]):
    name: str = "custom_text_entity_extraction"
    instruction: str = """Given a text, extract unique entities without repetition.
Ensure you consider different forms or mentions of the same entity as a single entity.
You should return a JSON object, containing a `entities` key, which is a list of unique entities derived from the text."""
    input_model = StringIO
    output_model = EntitiesList
    examples = [
        (
            StringIO(
                text="The Eiffel Tower, located in Paris, France, is one of the most iconic landmarks globally. Millions of visitors are attracted to it each year for its breathtaking views of the city. Completed in 1889, it was constructed in time for the 1889 World's Fair."
            ),
            EntitiesList(
                entities=["Eiffel Tower", "Paris", "France", "1889", "World's Fair"]
            ),
        ),
        (
            StringIO(
                text="The Colosseum in Rome, also known as the Flavian Amphitheatre, stands as a monument to Roman architectural and engineering achievement. Construction began under Emperor Vespasian in AD 70 and was completed by his son Titus in AD 80. It could hold between 50,000 and 80,000 spectators who watched gladiatorial contests and public spectacles."
            ),
            EntitiesList(
                entities=["Colosseum", "Rome", "Flavian Amphitheatre", "Vespasian", "AD 70", "Titus", "AD 80"]
            ),
        ),
    ]

    def to_string(self, data: Optional[InputModel] = None) -> str:
        examples_str: str = ""
        for i, (ex_input, ex_output) in enumerate(self.examples, 1):
            examples_str += f"EXAMPLE {i}:\n"
            examples_str += f"INPUT:\n{ex_input.model_dump_json(indent=4, exclude_none=True)}\n\n"
            examples_str += f"OUTPUT:\n{ex_output.model_dump_json(indent=4, exclude_none=True)}\n\n"

        input_obj: str = (
            data.model_dump_json(indent=4, exclude_none=True)
            if data is not None
            else "Input: (None)"
        )

        return f"""{self.instruction}

======= FEW SHOT EXAMPLES: =======
{examples_str}
======= END OF EXAMPLES =======

**IMPORTANT:
1. Make sure the output is always in JSON format.
2. Consider different forms or mentions of the same entity as a single entity.
3. DO NOT provide any further explanations or clarifications, just output the JSON.
**

Now extract the entities from the following text:

INPUT:
{input_obj}

JSON: 
"""
