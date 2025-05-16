# pylint: disable=C0114
# pylint: disable=C0301
# pylint: disable=C0325
# pylint: disable=W0622
# pylint: disable=R0913
# pylint: disable=R0917

from typing import Any, List, Union, Optional, Dict

from opik.evaluation.models import base_model
from opik.exceptions import MetricComputationError
from opik.evaluation.metrics.score_result import ScoreResult
from opik.evaluation.metrics.llm_judges import parsing_helpers
from opik.evaluation.metrics.llm_judges.answer_relevance import templates
from opik.evaluation.metrics.llm_judges.answer_relevance.metric import AnswerRelevance, AnswerRelevanceResponseFormat

class MyAnswerRelevance(AnswerRelevance):
    """
    A custom metric which works exactly like AnswerRelevance but with a different prompt template.
    Since there's no way to overwrite the template, other than to specify a `prompt` in the `evaluate` function,
    this class can be used as a replacement for AnswerRelevance, but with the added benefit of being able to modify
    the underlying prompt template and to test out new prompts.
    
    One more thing is that it makes no sense to incorporate the `context` into this metric evaluation,
    so the `context` parameter is removed. In both `RAGAs` and `DeepEval` only the `input` and `output` are used.

    The LLM still acts as a judge and determines how well the `output` answers the `input`.
    The score still ranges between 0.0 and 1.0, where higher values indicate better answer relevance.
    """

    def __init__(
        self,
        model: Optional[Union[str, base_model.OpikBaseModel]] = None,
        name: str = "my_answer_relevance_metric",
        few_shot_examples: Optional[
            List[templates.FewShotExampleWithContextAnswerRelevance]
        ] = None,
        few_shot_examples_no_context: Optional[
            List[templates.FewShotExampleNoContextAnswerRelevance]
        ] = None,
        require_context: bool = False,
        track: bool = True,
        project_name: Optional[str] = None,
    ):
        if require_context:
            raise NotImplementedError(
                "MyAnswerRelevance does not support context! Use AnswerRelevance instead or ",
                "set `require_context=False` in MyAnswerRelevance.__init__()."
            )

        super().__init__(
            name=name,
            track=track,
            project_name=project_name,
        )
        self._init_model(model)
        self._init_few_shot_examples(
            few_shot_examples_with_context=few_shot_examples,
            few_shot_examples_no_context=few_shot_examples_no_context,
        )

    def _init_few_shot_examples(
        self,
        few_shot_examples_with_context: Optional[
            List[templates.FewShotExampleWithContextAnswerRelevance]
        ],
        few_shot_examples_no_context: Optional[
            List[templates.FewShotExampleNoContextAnswerRelevance]
        ],
    ) -> None:
        self._few_shot_examples_no_context = (
            few_shot_examples_no_context
            if few_shot_examples_no_context
            else templates.FEW_SHOT_EXAMPLES_NO_CONTEXT
        )

    def score(
        self,
        input: str,
        output: str,
        context: Optional[List[str]] = None,
        **ignored_kwargs: Any,
    ) -> ScoreResult:
        """
        Calculate the answer relevance score for the given input-output pair.

        Args:
            input: The input text (question) to be evaluated.
            output: The output text (answer) to be evaluated.
            context: A list of context strings relevant to the input. If no context is given, the
                metric is calculated in no-context mode (the prompt template will not refer to context at all)
            **ignored_kwargs: Additional keyword arguments that are ignored.

        Returns:
            score_result.ScoreResult: A ScoreResult object containing the answer relevance score
            (between 0.0 and 1.0) and a reason for the score.
        """
        llm_query = self._generate_query_no_context(
            input=input,
            output=output,
            few_shot_examples=self._few_shot_examples_no_context,
        )
        model_output = self._model.generate_string(
            input=llm_query, response_format=AnswerRelevanceResponseFormat
        )
        return self._parse_model_output(content=model_output, name=self.name)

    async def ascore(
        self,
        input: str,
        output: str,
        context: Optional[List[str]] = None,
        **ignored_kwargs: Any,
    ) -> ScoreResult:
        """
        Asynchronously calculate the answer relevance score for the given input-output pair.

        This method is the asynchronous version of :meth:`score`. For detailed documentation,
        please refer to the :meth:`score` method.

        Args:
            input: The input text (question) to be evaluated.
            output: The output text (answer) to be evaluated.
            context: A list of context strings relevant to the input.
            **ignored_kwargs: Additional keyword arguments that are ignored.

        Returns:
            score_result.ScoreResult: A ScoreResult object with the answer relevance score and reason.
        """
        llm_query = self._generate_query_no_context(
            input=input,
            output=output,
            few_shot_examples=self._few_shot_examples_no_context,
        )
        model_output = await self._model.agenerate_string(
            input=llm_query, response_format=AnswerRelevanceResponseFormat
        )
        return self._parse_model_output(content=model_output, name=self.name)

    def _generate_query_no_context(
        self,
        input: str,
        output: str,
        few_shot_examples: Optional[
            List[templates.FewShotExampleNoContextAnswerRelevance]
        ] = None,
    ) -> str:
        examples_str: str = "\n\n".join(
            [
                f"#### Example {i + 1}: {example['title']}\n\n"
                f'- **Input:** "{example["input"]}"\n'
                f'- **Output:** "{example["output"]}"\n'
                f"- **Result:**\n"
                f"  ```json\n"
                f"  {{\n"
                f'    "answer_relevance_score": {example["answer_relevance_score"]},\n'
                f'    "reason": "{example["reason"]}"\n'
                f"  }}\n"
                f"  ```"
                for i, example in enumerate(few_shot_examples)
            ]
        )

        return f"""You are an NLP evaluation expert. Your task is to assess how relevant the given answer is to the user's question.

###INSTRUCTIONS###

- Analyze the user input and the provided answer
- Score relevance from 0.0 (irrelevant) to 1.0 (highly relevant)
- The perfect answer would be complete, fully relevant to the input and on-topic
- Return a JSON object containing 2 fields:
  - answer_relevance_score: a value between 0.0 and 1.0
  - reason: a short explanation for your verdict
- DO NOT RETURN ANYTHING ELSE BESIDES THE JSON OBJECT
- DO NOT PROVIDE ANY FURTHER EXPLANATIONS OR CLARIFICATIONS

###STEPS###

1. Identify the main points in the user's question
2. Check if the answer addresses these points
3. Consider any off-topic information in the answer and reduce the score
4. Assign a score and explain your reasoning

###EXAMPLE OUTPUT FORMAT###
{{
    "answer_relevance_score": 0.85,
    "reason": "The answer addresses the user's query about the primary topic but includes redundant information which reduces the score."
}}

###FEW-SHOT EXAMPLES###

{examples_str}

###EVALUATE:###
Input: {input}
Output: {output}

JSON:
"""

    def _parse_model_output(self, content: str, name: str) -> ScoreResult:
        try:
            dict_content: Dict = parsing_helpers.extract_json_content_or_raise(content)
            score: float = dict_content["answer_relevance_score"]

            if not (0.0 <= score <= 1.0):
                raise MetricComputationError(
                    f"Answer relevance score must be between 0.0 and 1.0, got {score}"
                )

            return ScoreResult(
                name=name, value=score, reason=dict_content["reason"]
            )
        except Exception as e:
            raise MetricComputationError from e
