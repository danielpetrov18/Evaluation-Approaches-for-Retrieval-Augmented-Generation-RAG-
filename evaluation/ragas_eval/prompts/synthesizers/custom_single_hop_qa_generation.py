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
from ragas.testset.synthesizers.base import QueryLength, QueryStyle
from ragas.testset.synthesizers.single_hop.prompts import QueryCondition, GeneratedQueryAnswer

InputModel = TypeVar("InputModel", bound=BaseModel)

class MySingleHopQAGenerationPrompt(PydanticPrompt[QueryCondition, GeneratedQueryAnswer]):
    name: str = "custom_single_hop_qa_generation_prompt"
    instruction: str = (
        """Generate a single-hop query and answer based on the specified conditions (persona, term, style, length) and the provided context.
The query should reflect how the persona would naturally ask about the given term, and the answer must be entirely derived from the context.
A `single-hop` query is a query that can be answered using only information present in or directly inferable from the context.

Guidelines:
1. Create a query that incorporates the specified term and matches both the persona's expertise level and interests
2. Apply the specified query style:
   - "Misspelled queries": Include intentional spelling errors while keeping query understandable
   - "Perfect grammar": Use flawless grammar and proper punctuation
   - "Poor grammar": Include grammatical errors (missing articles, wrong verb tenses, etc.)
   - "Web search like queries": Short, keyword-focused phrases without full sentences
3. Match the specified query length:
   - "short": Brief, concise questions (3-5 words)
   - "medium": Standard question length (6-12 words)
   - "long": Detailed, elaborate questions (13+ words)
4. Generate an answer using ONLY information present in or directly inferable from the context
5. The answer should be comprehensive enough to address the query but limited to what the context actually contains"""
    )

    input_model: Type[QueryCondition] = QueryCondition
    output_model: Type[GeneratedQueryAnswer] = GeneratedQueryAnswer
    examples: List[Tuple[QueryCondition, GeneratedQueryAnswer]] = [
        # PERFECT_GRAMMAR examples
        (
            QueryCondition(
                persona=Persona(
                    name="Software Engineer",
                    role_description="Focuses on coding best practices and system design.",
                ),
                term="microservices",
                query_style=QueryStyle.PERFECT_GRAMMAR,
                query_length=QueryLength.MEDIUM,
                context="Microservices are an architectural style where applications are structured as a collection of loosely coupled services. Each service is fine-grained and focuses on a single functionality."
            ),
            GeneratedQueryAnswer(
                query="What is the purpose of microservices in software architecture?",
                answer="Microservices are designed to structure applications as a collection of loosely coupled services, each focusing on a single functionality."
            ),
        ),
        # MISSPELLED examples
        (
            QueryCondition(
                persona=Persona(
                    name="College Student",
                    role_description="Studies history and political science with interest in international relations.",
                ),
                term="Berlin Wall",
                query_style=QueryStyle.MISSPELLED,
                query_length=QueryLength.MEDIUM,
                context="The Berlin Wall stood as a physical and ideological barrier between East and West Berlin from 1961 to 1989. Its fall on November 9, 1989, marked a pivotal moment in the end of the Cold War and led to German reunification in 1990."
            ),
            GeneratedQueryAnswer(
                query="When did the Berln Wal fal and what happend after?",
                answer="The Berlin Wall fell on November 9, 1989. This event marked a pivotal moment in the end of the Cold War and led to German reunification in 1990."
            ),
        ),
        # POOR_GRAMMAR examples
        (
            QueryCondition(
                persona=Persona(
                    name="Healthcare Worker",
                    role_description="Works in emergency medicine and patient care.",
                ),
                term="antibiotics",
                query_style=QueryStyle.POOR_GRAMMAR,
                query_length=QueryLength.LONG,
                context="Antibiotics are medications that destroy or slow down the growth of bacteria. They are not effective against viral infections like the common cold or flu. Overuse of antibiotics can lead to antibiotic resistance, where bacteria evolve to survive treatment with these drugs."
            ),
            GeneratedQueryAnswer(
                query="Why antibiotics not work for virus and what happen when people use too much of them for sickness?",
                answer="Antibiotics are medications that destroy or slow down the growth of bacteria. They are not effective against viral infections like the common cold or flu. Overuse of antibiotics can lead to antibiotic resistance, where bacteria evolve to survive treatment with these drugs."
            ),
        ),
        # WEB_SEARCH_LIKE examples
        (
            QueryCondition(
                persona=Persona(
                    name="Home Cook",
                    role_description="Enjoys experimenting with recipes and learning cooking techniques.",
                ),
                term="fermentation",
                query_style=QueryStyle.WEB_SEARCH_LIKE,
                query_length=QueryLength.SHORT,
                context="Fermentation is a metabolic process where microorganisms convert carbohydrates to acids, gases, or alcohol in the absence of oxygen. In food preparation, fermentation is used to produce bread, yogurt, cheese, wine, beer, and many other products. The process not only preserves food but also enhances flavors and increases nutritional value by creating beneficial enzymes, B vitamins, and probiotics."
            ),
            GeneratedQueryAnswer(
                query="fermentation benefits food",
                answer="Fermentation preserves food, enhances flavors, and increases nutritional value by creating beneficial enzymes, B vitamins, and probiotics."
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
1. Make sure the output is always in JSON format matching the schema:
   {{"query": "string", "answer": "string"}}
2. The query must incorporate the specified term and reflect the persona's perspective
3. The query style and length must match the requested parameters
4. The answer must be faithfully derived from the provided context only
5. DO NOT provide any explanations outside the JSON structure
**

Now generate for the following input:

INPUT:
{input_obj}

JSON:
"""
