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
from ragas.testset.transforms.extractors.llm_based import Headlines, TextWithExtractionLimit

InputModel = TypeVar("InputModel", bound=BaseModel)

class MyHeadlinesExtractorPrompt(PydanticPrompt[TextWithExtractionLimit, Headlines]):
    name: str = "custom_headlines_extractor_prompt"
    instruction: str = ("""Extract the most important `max_num` headlines from the given text that can be used to split the text into independent self-contained sections.
Focus mainly on Level 2 and Level 3 headings."""
    )

    input_model: Type[TextWithExtractionLimit] = TextWithExtractionLimit
    output_model: Type[Headlines] = Headlines
    examples: List[Tuple[TextWithExtractionLimit, Headlines]] = [
        (
            TextWithExtractionLimit(
                text="""Introduction
Overview of the topic...

Main Concepts
Explanation of core ideas...

Detailed Analysis
Techniques and methods for analysis...

Subsection: Specialized Techniques
Further details on specialized techniques...

Future Directions
Insights into upcoming trends...

Subsection: Next Steps in Research
Discussion of new areas of study...

Conclusion
Final remarks and summary.
""",
                max_num=6,
            ),
            Headlines(
                headlines=[
                    "Introduction",
                    "Main Concepts",
                    "Detailed Analysis",
                    "Subsection: Specialized Techniques",
                    "Future Directions",
                    "Conclusion",
                ],
            ),
        ),
        (
            TextWithExtractionLimit(
                text="",
                max_num=10,
            ),
            Headlines(
                headlines=[],
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
5. If the text contains less headlines than the `max_num`, output as many headlines as possible without exceeding the `max_num`.
6. DO NOT provide any further explanations or clarifications, just output the JSON.
**

Now extract the headlines from the following text:

INPUT:
{input_obj}

JSON:
"""
