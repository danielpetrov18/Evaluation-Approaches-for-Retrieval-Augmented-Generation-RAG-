# pylint: disable=C0114
# pylint: disable=C0115
# pylint: disable=C0301
# pylint: disable=C0325
# pylint: disable=W0221
# pylint: disable=W0622
# pylint: disable=R0913
# pylint: disable=R0917

from typing import Any, List, Union, Optional

from opik.evaluation.metrics.base_metric import BaseMetric
from opik.evaluation.metrics.score_result import ScoreResult
from opik.evaluation.models import base_model, models_factory

from .parser import parse_model_output
from .models import ContextPrecisionVerdicts
from .template import generate_query, FewShotExampleContextPrecision

class ContextPrecision(BaseMetric):
    """
    A metric that evaluates the context precision of an input-output pair relative to the context using a LLM.
    Is the context fetched relevant and is it ranked high enough?

    This metric uses a language model to assess how well the given output aligns with
    the provided context for the given input. It returns a score between 0.0 and 1.0,
    where higher values indicate better context precision. It uses the `Weigthed Context Precision` formula.

    Args:
        model: The language model to use for evaluation. Can be a string (model name) or an `opik.evaluation.models.OpikBaseModel` subclass instance.
            `opik.evaluation.models.LiteLLMChatModel` is used by default.
        name: The name of the metric. Defaults to "context_precision_metric".
        few_shot_examples: A list of few-shot examples to provide to the model. If None, uses the default few-shot examples.
        track: Whether to track the metric. Defaults to True.
        project_name: Optional project name to track the metric in for the cases when
            there are no parent span/trace to inherit project name from.
    """

    def __init__(
        self,
        model: Optional[Union[str, base_model.OpikBaseModel]] = None,
        name: str = "context_precision_metric",
        few_shot_examples: Optional[
            List[FewShotExampleContextPrecision]
        ] = None,
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
        expected_output: str,
        context: List[str],
        **ignored_kwargs: Any,
    ) -> ScoreResult:
        """
        Calculate the context precision score for the given input-output pair.

        Args:
            input: The input text to be evaluated.
            expected_output: The expected output for the given input.
            context: A list of context strings relevant to the input.
            **ignored_kwargs: Additional keyword arguments that are ignored.

        Returns:
            ScoreResult: A ScoreResult object containing the context precision score
            (between 0.0 and 1.0) and a reason for the score.
        """
        llm_query: str = generate_query(
            input=input,
            expected_output=expected_output,
            context=context,
            few_shot_examples=self.few_shot_examples,
        )

        model_output: str = self._model.generate_string(
            input=llm_query, response_format=ContextPrecisionVerdicts
        )

        return parse_model_output(content=model_output, name=self.name)

    async def ascore(
        self,
        input: str,
        expected_output: str,
        context: List[str],
        **ignored_kwargs: Any,
    ) -> ScoreResult:
        """
        Asynchronously calculate the context precision score for the given input-output pair.

        Args:
            input: The input text to be evaluated.
            expected_output: The expected output for the given input.
            context: A list of context strings relevant to the input.
            **ignored_kwargs: Additional keyword arguments that are ignored.

        Returns:
            ScoreResult: A ScoreResult object with the context precision score and reason.
        """
        llm_query: str = generate_query(
            input=input,
            expected_output=expected_output,
            context=context,
            few_shot_examples=self.few_shot_examples,
        )

        model_output: str = await self._model.agenerate_string(
            input=llm_query, response_format=ContextPrecisionVerdicts
        )

        return parse_model_output(content=model_output, name=self.name)
