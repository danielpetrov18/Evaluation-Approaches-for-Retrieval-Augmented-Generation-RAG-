# pylint: disable=C0114
# pylint: disable=C0301
# pylint: disable=R0902
# pylint: disable=R0913
# pylint: disable=R0917
# pylint: disable=W0221

from typing import List, Type, Optional

from ragas import (
    evaluate,
    RunConfig
)
from ragas.dataset_schema import (
    SingleTurnSample,
    EvaluationDataset
)
from ragas.prompt import PydanticPrompt
from ragas.metrics._answer_relevance import (
    ResponseRelevancy,
    ResponseRelevancePrompt,
    ResponseRelevanceInput,
    ResponseRelevanceOutput
)
from ragas.evaluation import EvaluationResult
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper

from deepeval.metrics import BaseMetric
from deepeval.telemetry import capture_metric_type
from deepeval.metrics.utils import check_llm_test_case_params
from deepeval.test_case import LLMTestCase, LLMTestCaseParams
from deepeval.metrics.ragas import format_ragas_metric_name, import_ragas

class CustomRAGASAnswerRelevancyMetric(BaseMetric):
    """
    Custom wrapper around the DeepEval RAGAs implementation for answer relevancy.
    I require this wrapper, since the default `DeepEval` implementation does not support passing a `RunConfig` or a DiskCacheBackend.
    Since we are using weaker models and the GPU is not that powerful, we need to extend the timeout.
    Furthermore, the original one doesn't support passing a custom template.
    Besides that, the implementation is very similar to the original one.
    """

    # Required parameters for the metric
    # Under: `/ragas/metrics/_answer_relevance.py` check out the metric definition
    _required_params: List[LLMTestCaseParams] = [
        LLMTestCaseParams.INPUT,
        LLMTestCaseParams.ACTUAL_OUTPUT
    ]

    def __init__(
        self,
        ragas_llm: LangchainLLMWrapper,
        ragas_embeddings: LangchainEmbeddingsWrapper,
        run_config: Optional[RunConfig] = None,
        threshold: float = 0.3,
        evaluation_template: Type[
            PydanticPrompt[ResponseRelevanceInput, ResponseRelevanceOutput]
        ] = ResponseRelevancePrompt(),
        strictness: float = 3, # Number of hypothetical questions to generate
        experiment_name: Optional[str] = None,
        _track: bool = True,
    ):
        # Verify if ragas is installed and is atleast version `0.2.1`
        import_ragas()

        self.model = ragas_llm
        self.embeddings = ragas_embeddings
        self.run_config = run_config
        self.threshold = threshold
        self.evaluation_template = evaluation_template
        self.strictness = strictness
        self.experiment_name = experiment_name
        self.score = 0.0
        self.success = False
        self._track = _track

    async def a_measure(
        self, test_case: LLMTestCase, _show_indicator: bool = False
    ) -> float:
        return self.measure(test_case)

    def measure(self, test_case: LLMTestCase) -> float:
        check_llm_test_case_params(
            test_case, self._required_params, self
        )

        sample = SingleTurnSample(
            user_input=test_case.input,
            response=test_case.actual_output
        )
        dataset = EvaluationDataset(samples=[sample])

        with capture_metric_type(
            self.__name__, _track=self._track, async_mode=False
        ):
            scores: EvaluationResult = evaluate(
                dataset,
                metrics=[
                    ResponseRelevancy(
                        question_generation=self.evaluation_template,
                        strictness=self.strictness,
                    )
                ],
                llm=self.model,
                embeddings=self.embeddings,
                experiment_name=self.experiment_name,
                run_config=self.run_config,
                show_progress=True
            )
            answer_relevancy_score: float = scores["answer_relevancy"][0]
            self.success = answer_relevancy_score >= self.threshold
            self.score = answer_relevancy_score
            return self.score

    def is_successful(self):
        return self.success

    @property
    def __name__(self):
        return format_ragas_metric_name("Answer Relevancy")
