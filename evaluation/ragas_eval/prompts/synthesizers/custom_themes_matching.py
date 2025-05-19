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
from ragas.testset.synthesizers.prompts import ThemesPersonasInput, PersonaThemesMapping

InputModel = TypeVar("InputModel", bound=BaseModel)

class MyThemesPersonasMatchingPrompt(
    PydanticPrompt[ThemesPersonasInput, PersonaThemesMapping]
):
    name: str = "custom_themes_personas_matching_prompt"
    instruction: str = (
        """Given a list of themes and personas with their roles, associate each persona with relevant themes based on their role description.
Each persona should be matched with themes that align with their expertise, interests, responsibilities, requirements or needs."""
    )

    input_model: Type[ThemesPersonasInput] = ThemesPersonasInput
    output_model: Type[PersonaThemesMapping] = PersonaThemesMapping
    examples: List[Tuple[ThemesPersonasInput, PersonaThemesMapping]] = [
        (
            ThemesPersonasInput(
                themes=["Data visualization", "Machine learning", "Python", "Data cleaning", "Statistics"],
                personas=[
                    Persona(
                        name="Data Scientist",
                        role_description="Analyzes complex datasets and builds predictive models.",
                    ),
                    Persona(
                        name="Business Analyst",
                        role_description="Interprets data to provide business insights and recommendations.",
                    ),
                ],
            ),
            PersonaThemesMapping(
                mapping={
                    "Data Scientist": ["Machine learning", "Python", "Data cleaning", "Statistics", "Data visualization"],
                    "Business Analyst": ["Data visualization", "Statistics", "Data cleaning"],
                }
            ),
        ),
        (
            ThemesPersonasInput(
                themes=["Cybersecurity", "Cloud infrastructure", "Automation", "DevOps"],
                personas=[
                    Persona(
                        name="Security Engineer",
                        role_description="Implements security measures and monitors for threats.",
                    ),
                    Persona(
                        name="Cloud Architect",
                        role_description="Designs and oversees cloud computing strategies.",
                    ),
                ],
            ),
            PersonaThemesMapping(
                mapping={
                    "Security Engineer": ["Cybersecurity", "Cloud infrastructure"],
                    "Cloud Architect": ["Cloud infrastructure", "Automation", "DevOps"],
                }
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

{examples_str.strip()}

======= END OF EXAMPLES =======

**IMPORTANT:
1. Make sure the output is always in JSON format.
2. Do not include any other text in the output.
3. The output should have a key "mapping" containing a dictionary where keys are persona names and values are lists of relevant themes.
4. Only include themes from the provided list that are relevant to each persona.
5. DO NOT provide any further explanations or clarifications, just output the JSON.
**

Now match each persona with relevant themes based on their role description:

INPUT:
{input_obj}

JSON:
"""
