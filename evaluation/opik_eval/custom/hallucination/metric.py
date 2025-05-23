# pylint: disable = C0114
# pylint: disable = C0301
# pylint: disable = R0913
# pylint: disable = R0917
# pylint: disable = W0221
# pylint: disable = W0622

from typing import Union, Optional, List, Any

from opik.evaluation.metrics.base_metric import BaseMetric
from opik.evaluation.metrics.score_result import ScoreResult
from opik.evaluation.models import base_model, models_factory
from opik.evaluation.metrics.llm_judges.hallucination import parser
from opik.evaluation.metrics.llm_judges.hallucination.metric import HallucinationResponseFormat

from .template import FewShotExampleHallucination, generate_query

class Hallucination(BaseMetric):
    """
    Custom implementation of the `Hallucination` metric provided by Opik.
    Since there's a pretty inflexible way to provide/overwrite template.
    
    This metric uses:
        - `input`
        - `output`
        - `context`
    
    It verifies if the `output` doesn't contradict the context.
    Furthermore, it checks if the `output` doesn't provide information outside the `context`.

    All metrics use the `LLM-As-A-Judge` to evaluate the output.
    The closer the score is to 1.0, the more likely the output is hallucinated.
    The lower the score is, the more likely the output is factual (based on the `context`).

    The only difference between this version and the original is that I always consider the context.
    The second difference is the prompt template I submit to the LLM, besides that it's the same.

    Args:
        model: The LLM to use for evaluation. Can be a string (model name) or an `opik.evaluation.models.OpikBaseModel` subclass instance.
            `opik.evaluation.models.LiteLLMChatModel` is used by default.
        name: The name of the metric.
        few_shot_examples: A list of few-shot examples to use for hallucination detection.  If None, default examples will be used.
        track: Whether to track the metric. Defaults to True.
        project_name: Optional project name to track the metric in for the cases when
            there are no parent span/trace to inherit project name from.
    """

    def __init__(
        self,
        model: Optional[Union[str, base_model.OpikBaseModel]] = None,
        name: str = "hallucination_metric",
        few_shot_examples: Optional[List[FewShotExampleHallucination]] = None,
        track: bool = True,
        project_name: Optional[str] = None,
    ):
        super().__init__(name=name, track=track, project_name=project_name)
        self._init_model(model)
        self.few_shot_examples = few_shot_examples

    def _init_model(
        self, model: Optional[Union[str, base_model.OpikBaseModel]]
    ) -> None:
        if isinstance(model, base_model.OpikBaseModel):
            self._model = model
        else:
            self._model = models_factory.get(model_name=model)

    def score(
        self,
        input: str,
        output: str,
        context: List[str],
        **ignored_kwargs: Any,
    ) -> ScoreResult:
        """
        Calculate the hallucination score for the given input, output, and context field.

        Args:
            input: The original input/question.
            output: The LLM's output to evaluate.
            context: A list of context strings.
            **ignored_kwargs: Additional keyword arguments that are ignored.

        Returns:
            ScoreResult: A ScoreResult object with a value of 1.0 if hallucination
                is detected, 0.0 otherwise, along with the reason for the verdict.
        """
        llm_query: str = generate_query(
            input=input,
            output=output,
            context=context,
            few_shot_examples=self.few_shot_examples,
        )
        model_output: str = self._model.generate_string(
            input=llm_query, response_format=HallucinationResponseFormat
        )

        return parser.parse_model_output(content=model_output, name=self.name)

    async def ascore(
        self,
        input: str,
        output: str,
        context: List[str],
        **ignored_kwargs: Any,
    ) -> ScoreResult:
        """
        Asynchronously calculate the hallucination score for the given input, output, and context field.

        Args:
            input: The original input/question.
            output: The LLM's output to evaluate.
            context: A list of context strings.
            **ignored_kwargs: Additional keyword arguments that are ignored.

        Returns:
            ScoreResult: A ScoreResult object with a value of 1.0 if hallucination
                is detected, 0.0 otherwise, along with the reason for the verdict.
        """
        llm_query: str = generate_query(
            input=input,
            output=output,
            context=context,
            few_shot_examples=self.few_shot_examples,
        )
        model_output: str = await self._model.agenerate_string(
            input=llm_query, response_format=HallucinationResponseFormat
        )

        return parser.parse_model_output(content=model_output, name=self.name)
