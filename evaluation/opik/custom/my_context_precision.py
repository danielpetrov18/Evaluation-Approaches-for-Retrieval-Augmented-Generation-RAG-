# pylint: disable=C0114
# pylint: disable=C0301
# pylint: disable=C0325
# pylint: disable=W0622
# pylint: disable=R0913
# pylint: disable=R0917

from typing import Any, List, Union, Optional, Dict

from opik.exceptions import MetricComputationError
from opik.evaluation.models import base_model
from opik.evaluation.metrics import ContextPrecision
from opik.evaluation.metrics.score_result import ScoreResult
from opik.evaluation.metrics.llm_judges import parsing_helpers
from opik.evaluation.metrics.llm_judges.context_precision.metric import ContextPrecisionResponseFormat
from opik.evaluation.metrics.llm_judges.context_precision.template import FewShotExampleContextPrecision

FEW_SHOT_EXAMPLES: List[FewShotExampleContextPrecision] = [
    {
        "title": "Low Context Precision Score",
        "input": "What are the main causes of climate change?",
        "expected_output": "The main causes of climate change are greenhouse gas emissions, deforestation, and industrial processes. Carbon dioxide from burning fossil fuels is the largest contributor.",
        "context": [
            "The Olympics in 2024 will be hosted by Paris, France.",
            "Soccer is the most popular sport in the world with over 4 billion fans.",
            "Political tensions between countries have increased in recent years.",
            "Climate change is partially caused by carbon dioxide emissions.",
            "The average global temperature has risen by 1.1°C since pre-industrial times."
        ],
        "output": "Climate change is partially caused by carbon dioxide emissions and global temperatures have risen by 1.1°C since pre-industrial times.",
        "context_precision_score": 0.3,
        "reason": "Only 2 out of 5 context chunks are relevant to climate change causes. The first three chunks are completely irrelevant, and the relevant chunks are positioned poorly (ranks 4 and 5) instead of being ranked higher. The answer only incorporates the limited relevant information available."
    },
    {
        "title": "High Context Precision Score",
        "input": "What are the health benefits of regular exercise?",
        "expected_output": "Regular exercise offers numerous health benefits including improved cardiovascular health, weight management, stronger muscles and bones, reduced risk of chronic diseases, better mental health, and improved sleep quality.",
        "context": [
            "Regular exercise improves cardiovascular health by strengthening the heart and improving circulation.",
            "Physical activity helps with weight management and reduces the risk of obesity.",
            "Exercise strengthens muscles and bones, reducing the risk of osteoporosis and injuries.",
            "Regular physical activity is linked to reduced risk of chronic diseases like diabetes and heart disease.",
            "Exercise releases endorphins that improve mood and reduce symptoms of depression and anxiety."
        ],
        "output": "Regular exercise provides multiple health benefits: it improves cardiovascular health by strengthening the heart, helps with weight management, strengthens muscles and bones, reduces risk of chronic diseases like diabetes, and improves mental health by releasing mood-enhancing endorphins.",
        "context_precision_score": 0.9,
        "reason": "All 5 chunks are highly relevant to the question about exercise benefits. The chunks are also well-ranked with the most important information about cardiovascular benefits appearing first. The score is 0.9 rather than 1.0 because the context lacks information about improved sleep quality mentioned in the expected output."
    }
]

class MyContextPrecision(ContextPrecision):
    """
    A custom metric which works exactly like ContextPrecision but with a different prompt template.
    Since there's no way to overwrite the template, other than to specify a `prompt` in the `evaluate` function,
    this class can be used as a replacement for ContextPrecision, but with the added benefit of being able to modify
    the underlying prompt template and to test out new prompts.
    
    The LLM still acts as a judge and determines if the response is relevant based on the context.
    The score still ranges between 0.0 and 1.0, where higher values indicate better context precision.
    """

    def __init__(
        self,
        model: Optional[Union[str, base_model.OpikBaseModel]] = None,
        name: str = "my_context_precision_metric",
        few_shot_examples: Optional[List[FewShotExampleContextPrecision]] = None,
        track: bool = True,
        project_name: Optional[str] = None,
    ):
        super().__init__(
            model=model,
            name=name,
            few_shot_examples=few_shot_examples,
            track=track,
            project_name=project_name,
        )

    def score(
        self,
        input: str,
        output: str,
        expected_output: str,
        context: List[str],
        **ignored_kwargs: Any,
    ) -> ScoreResult:
        """
        Calculate the context precision score for the given input-output pair.

        Args:
            input: The input text to be evaluated.
            output: The output text to be evaluated.
            expected_output: The expected output for the given input.
            context: A list of context strings relevant to the input.
            **ignored_kwargs: Additional keyword arguments that are ignored.

        Returns:
            score_result.ScoreResult: A ScoreResult object containing the context precision score
            (between 0.0 and 1.0) and a reason for the score.
        """
        llm_query = self._generate_query_no_context(
            input=input,
            output=output,
            expected_output=expected_output,
            context=context,
            few_shot_examples=self.few_shot_examples,
        )
        model_output = self._model.generate_string(
            input=llm_query, response_format=ContextPrecisionResponseFormat
        )

        return self._parse_model_output(content=model_output, name=self.name)

    async def ascore(
        self,
        input: str,
        output: str,
        expected_output: str,
        context: List[str],
        **ignored_kwargs: Any,
    ) -> ScoreResult:
        """
        Asynchronously calculate the context precision score for the given input-output pair.

        This method is the asynchronous version of :meth:`score`. For detailed documentation,
        please refer to the :meth:`score` method.

        Args:
            input: The input text to be evaluated.
            output: The output text to be evaluated.
            expected_output: The expected output for the given input.
            context: A list of context strings relevant to the input.
            **ignored_kwargs: Additional keyword arguments that are ignored.

        Returns:
            score_result.ScoreResult: A ScoreResult object with the context precision score and reason.
        """
        llm_query = self._generate_query_no_context(
            input=input,
            output=output,
            expected_output=expected_output,
            context=context,
            few_shot_examples=self.few_shot_examples,
        )
        model_output = await self._model.agenerate_string(
            input=llm_query, response_format=ContextPrecisionResponseFormat
        )

        return self._parse_model_output(content=model_output, name=self.name)

    def _generate_query_no_context(
        self,
        input: str,
        output: str,
        expected_output: str,
        context: List[str],
        few_shot_examples: Optional[List[FewShotExampleContextPrecision]] = None,
    ) -> str:
        examples_str = "\n\n".join(
            [
                f"#### Example {i + 1}: {example['title']}\n\n"
                f'- **Input:** "{example["input"]}"\n'
                f'- **Output:** "{example["output"]}"\n'
                f'- **Expected Output:** "{example["expected_output"]}"\n'
                f'- **Context:** "{example["context"]}"\n'
                f"- **Result:**\n"
                f"  ```json\n"
                f"  {{\n"
                f'    "context_precision_score": {example["context_precision_score"]},\n'
                f'    "reason": "{example["reason"]}"\n'
                f"  }}\n"
                f"  ```"
                for i, example in enumerate(few_shot_examples)
            ]
        )

        # Format the context properly
        context: str = "\n\n".join([f"{i + 1}: {context_str}" for i, context_str in enumerate(context)])

        return f"""You are an evaluation expert for RAG (Retrieval-Augmented Generation) systems. Your task is to measure context precision - what fraction of the retrieved chunks are actually relevant to answering the user's question and are they ranked high.
 
###INSTRUCTIONS###

- Review the user's input question and the expected output answer
- For each piece of context, determine if it's relevant for answering the input, considering both relevance and ranking position
- A relevant chunk contains information needed to formulate the expected answer
- Calculate context precision by focusing on both relevance and ranking position
- Higher ranked (earlier appearing) relevant chunks should contribute more to the score
- DO NOT RETURN ANYTHING ELSE BESIDES THE JSON OBJECT
- DO NOT PROVIDE ANY FURTHER EXPLANATIONS OR CLARIFICATIONS

###SCORING GUIDE###

- 0.0: No retrieved chunks are relevant to answering the question
- 0.2: Only a small portion of chunks are relevant, or relevant chunks are ranked poorly
- 0.4: Some chunks are relevant but many irrelevant chunks are included or ranking is suboptimal
- 0.6: Most chunks are relevant with acceptable ranking, but some irrelevant chunks included
- 0.8: Nearly all chunks are relevant with good ranking priority
- 1.0: All retrieved chunks are relevant and optimally ranked (most relevant chunks first)

Remember that context precision measures:
1. How many of the retrieved chunks are actually relevant
2. Whether relevant chunks are ranked higher than irrelevant ones

###EXAMPLE OUTPUT FORMAT###
{{
    "context_precision_score": 0.6,
    "reason": "Most chunks are relevant to the question about climate change effects, but the third chunk about political elections is irrelevant, and one relevant chunk is ranked too low."
}}

###FEW-SHOT EXAMPLES###

{examples_str}

###EVALUATE:###

Input: {input}
Output: {output}
Expected Output: {expected_output}
Context: {context}

JSON:
"""

    def _parse_model_output(self, content: str, name: str) -> ScoreResult:
        try:
            dict_content: Dict = parsing_helpers.extract_json_content_or_raise(content)
            score: float = dict_content["context_precision_score"]

            if not (0.0 <= score <= 1.0):
                raise MetricComputationError(
                    f"Context precision score must be between 0.0 and 1.0, got {score}"
                )

            return ScoreResult(
                name=name, value=score, reason=dict_content["reason"]
            )
        except Exception as e:
            raise MetricComputationError from e
