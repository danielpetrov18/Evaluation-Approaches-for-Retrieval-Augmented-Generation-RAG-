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
from ragas.testset.synthesizers.multi_hop.prompts import QueryConditions, GeneratedQueryAnswer

InputModel = TypeVar("InputModel", bound=BaseModel)

class MyMultiHopQAGenerationPrompt(PydanticPrompt[QueryConditions, GeneratedQueryAnswer]):
    name: str = "custom_multi_hop_qa_generation_prompt"
    instruction: str = (
        """Generate a multi-hop query and answer based on the specified conditions (persona, themes, style, length) and the provided context.
The query should reflect how the persona would naturally ask about the given themes, and the answer must connect information from multiple context segments.
A `multi-hop` query requires combining information from at least two different context segments to formulate a complete answer.

Guidelines:
1. Create a query that incorporates one or more of the specified themes and matches both the persona's expertise level and interests
2. Ensure your query requires connecting information from at least two context segments
3. Apply the specified query style:
   - "Misspelled queries": Include intentional spelling errors while keeping query understandable
   - "Perfect grammar": Use flawless grammar and proper punctuation
   - "Poor grammar": Include grammatical errors (missing articles, wrong verb tenses, etc.)
   - "Web search like queries": Short, keyword-focused phrases without full sentences
4. Match the specified query length:
   - "short": Brief, concise questions (3-5 words)
   - "medium": Standard question length (6-12 words)
   - "long": Detailed, elaborate questions (13+ words)
5. Generate an answer using ONLY information present in or directly inferable from MULTIPLE context segments
6. The answer should clearly demonstrate the connection between information from different context segments"""
    )

    input_model: Type[QueryConditions] = QueryConditions
    output_model: Type[GeneratedQueryAnswer] = GeneratedQueryAnswer
    examples: List[Tuple[QueryConditions, GeneratedQueryAnswer]] = [
        # PERFECT_GRAMMAR examples
        (
            QueryConditions(
                persona=Persona(
                    name="Historian",
                    role_description="Focuses on major scientific milestones and their global impact.",
                ),
                themes=["Theory of Relativity", "Experimental Validation"],
                query_style=QueryStyle.PERFECT_GRAMMAR,
                query_length=QueryLength.MEDIUM,
                context=[
                    "Albert Einstein developed the theory of relativity, introducing the concept of spacetime.",
                    "The bending of light by gravity was confirmed during the 1919 solar eclipse, supporting Einstein's theory."
                ]
            ),
            GeneratedQueryAnswer(
                query="How was Einstein's theory of relativity experimentally validated?",
                answer="Albert Einstein developed the theory of relativity, introducing the concept of spacetime. The theory was experimentally validated when the bending of light by gravity was confirmed during the 1919 solar eclipse, which supported Einstein's theory."
            ),
        ),
        # MISSPELLED examples
        (
            QueryConditions(
                persona=Persona(
                    name="Science Enthusiast",
                    role_description="Amateur science follower with interest in space exploration.",
                ),
                themes=["Moon Landing", "Space Race"],
                query_style=QueryStyle.MISSPELLED,
                query_length=QueryLength.MEDIUM,
                context=[
                    "The Space Race was a competition between the United States and the Soviet Union to achieve superior spaceflight capability, starting with the Soviet launch of Sputnik in 1957.",
                    "NASA's Apollo 11 mission successfully landed astronauts on the Moon on July 20, 1969, with Neil Armstrong becoming the first human to walk on the lunar surface."
                ]
            ),
            GeneratedQueryAnswer(
                query="Wich contry won the spase race by landig on the mooon?",
                answer="The Space Race was a competition between the United States and the Soviet Union to achieve superior spaceflight capability, starting with the Soviet launch of Sputnik in 1957. The United States achieved a significant victory in this competition when NASA's Apollo 11 mission successfully landed astronauts on the Moon on July 20, 1969, with Neil Armstrong becoming the first human to walk on the lunar surface."
            ),
        ),
        # POOR_GRAMMAR examples
        (
            QueryConditions(
                persona=Persona(
                    name="Healthcare Worker",
                    role_description="Works in emergency medicine and patient care.",
                ),
                themes=["Vaccine Development", "Polio Eradication"],
                query_style=QueryStyle.POOR_GRAMMAR,
                query_length=QueryLength.LONG,
                context=[
                    "Jonas Salk developed the first effective polio vaccine in 1955, which used inactivated poliovirus.",
                    "Global polio cases have decreased by over 99% since 1988, from an estimated 350,000 cases to just a few hundred reported cases per year."
                ]
            ),
            GeneratedQueryAnswer(
                query="How the vaccine from Jonas Salk help with polio cases in world and how many cases now compared to before?",
                answer="Jonas Salk developed the first effective polio vaccine in 1955, which used inactivated poliovirus. Since 1988, global polio cases have decreased by over 99%, from an estimated 350,000 cases to just a few hundred reported cases per year."
            ),
        ),
        # WEB_SEARCH_LIKE examples
        (
            QueryConditions(
                persona=Persona(
                    name="Technology Blogger",
                    role_description="Writes about technology trends and digital innovation.",
                ),
                themes=["Internet Invention", "Web Development"],
                query_style=QueryStyle.WEB_SEARCH_LIKE,
                query_length=QueryLength.SHORT,
                context=[
                    "ARPANET, the precursor to the internet, was first deployed in 1969 connecting four university computers.",
                    "Tim Berners-Lee invented the World Wide Web in 1989 while working at CERN, developing the first web browser and server."
                ]
            ),
            GeneratedQueryAnswer(
                query="internet history berners-lee connection",
                answer="ARPANET, the precursor to the internet, was first deployed in 1969 connecting four university computers. Later, Tim Berners-Lee invented the World Wide Web in 1989 while working at CERN, developing the first web browser and server."
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
1. Make sure the output is always in JSON format matching the schema:
   {{"query": "string", "answer": "string"}}
2. The query must incorporate one or more of the specified themes and reflect the persona's perspective
3. The query must require connecting information from at least two different context segments
4. The query style and length must match the requested parameters
5. The answer must demonstrate connections between multiple context segments
6. The answer must be faithfully derived from the provided context segments only
7. DO NOT provide any explanations outside the JSON structure
**

Now generate a query-answer pair based on the specified conditions (persona, themes, query style, length) and the provided context.

INPUT:
{input_obj}

JSON:
"""
