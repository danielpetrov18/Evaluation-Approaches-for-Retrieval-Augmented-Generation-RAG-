# pylint: disable = C0114
# pylint: disable = C0301
# pylint: disable = R0913
# pylint: disable = R0917
# pylint: disable = W0221
# pylint: disable = W0622

from typing import Any, List, Optional, Union

from opik.evaluation.metrics.base_metric import BaseMetric
from opik.evaluation.metrics.score_result import ScoreResult
from opik.evaluation.models import base_model, models_factory
from opik.evaluation.metrics.llm_judges.answer_relevance import parser
from opik.evaluation.metrics.llm_judges.answer_relevance.metric import AnswerRelevanceResponseFormat

from .template import FewShotExampleAnswerRelevance, generate_query

class AnswerRelevance(BaseMetric):
    """
    Custom metric for evaluating the response relevance of a LLM.
    Unlike `RAGAs` we are not going to use semantic similarity.
    Unlike `DeepEval` we are not going to use the formula `relevant statements/total statements`.
    
    The score will be fully judged by a LLM, with scores ranging from 0.0 to 1.0.
    The closer to 1.0, the more relevant the answer.

    Args:
        model: The language model to use for evaluation. Can be a string (model name) or an `opik.evaluation.models.OpikBaseModel` subclass instance.
            `opik.evaluation.models.LiteLLMChatModel` is used by default.
        name: The name of the metric. Defaults to "AnswerRelevanceMetric".
        few_shot_examples: A list of dict to include as examples to the prompt query. Context key is required.
            If not provided, Opik's generic examples will be used.
        few_shot_examples_no_context: A list of dict to include as examples to the prompt query in no-context mode (so, 'context' key is not needed).
            If not provided, Opik's generic examples will be used.
        require_context: if set to False, execution in no-context mode is allowed. Default is True.
        track: Whether to track the metric. Defaults to True.
        project_name: Optional project name to track the metric in for the cases when there are no parent span/trace to inherit project name from.
    """

    def __init__(
        self,
        model: Optional[Union[str, base_model.OpikBaseModel]] = None,
        name: str = "answer_relevance_metric",
        few_shot_examples: Optional[
            List[FewShotExampleAnswerRelevance]
        ] = None,
        track: bool = True,
        project_name: Optional[str] = None,
    ):
        super().__init__(
            name=name,
            track=track,
            project_name=project_name,
        )
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
        **ignored_kwargs: Any,
    ) -> ScoreResult:
        """
        Calculate the answer relevance score for the given input-output pair.

        Args:
            input: The input text (question) to be evaluated.
            output: The output text (answer) to be evaluated.
            **ignored_kwargs: Additional keyword arguments that are ignored.

        Returns:
            score_result.ScoreResult: A ScoreResult object containing the answer relevance score
            (between 0.0 and 1.0) and a reason for the score.
        """
        llm_query = generate_query(
            input=input,
            output=output,
            few_shot_examples=self.few_shot_examples
        )

        model_output = self._model.generate_string(
            input=llm_query, response_format=AnswerRelevanceResponseFormat
        )
        return parser.parse_model_output(content=model_output, name=self.name)

    async def ascore(
        self,
        input: str,
        output: str,
        **ignored_kwargs: Any,
    ) -> ScoreResult:
        """
        Asynchronously calculate the answer relevance score for the given input-output pair.

        This method is the asynchronous version of :meth:`score`. For detailed documentation,
        please refer to the :meth:`score` method.

        Args:
            input: The input text (question) to be evaluated.
            output: The output text (answer) to be evaluated.
            **ignored_kwargs: Additional keyword arguments that are ignored.

        Returns:
            ScoreResult: A ScoreResult object with the answer relevance score and reason.
        """
        llm_query = generate_query(
            input=input,
            output=output,
            few_shot_examples=self.few_shot_examples
        )
        model_output = await self._model.agenerate_string(
            input=llm_query, response_format=AnswerRelevanceResponseFormat
        )

        return parser.parse_model_output(content=model_output, name=self.name)
