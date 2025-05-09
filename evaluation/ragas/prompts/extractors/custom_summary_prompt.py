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
from ragas.prompt import PydanticPrompt, StringIO

InputModel = TypeVar("InputModel", bound=BaseModel)

class MySummaryExtractorPrompt(PydanticPrompt[StringIO, StringIO]):
    instruction: str = "Summarize the given text in less than 10 sentences. The summary should be clear and concise and not skip any important details."
    input_model: Type[StringIO] = StringIO
    output_model: Type[StringIO] = StringIO
    examples: List[Tuple[StringIO, StringIO]] = [
        (
            StringIO(
                text="Artificial intelligence\n\nArtificial intelligence is transforming various industries by automating tasks that previously required human intelligence. From healthcare to finance, AI is being used to analyze vast amounts of data quickly and accurately. This technology is also driving innovations in areas like self-driving cars and personalized recommendations."
            ),
            StringIO(
                text="AI is revolutionizing industries by automating tasks, analyzing data, and driving innovations like self-driving cars and personalized recommendations."
            ),
        )
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
2. Do not include any other information in the output.
3. The output should have a key "text" containing the summary.
4. DO NOT exceed 10 sentences.
5. DO NOT provide any further explanations or clarifications, just output the JSON.
**

Now perform the same for the following:

INPUT:
{input_obj}

JSON:
"""
