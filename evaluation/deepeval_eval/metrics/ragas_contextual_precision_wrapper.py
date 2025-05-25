# pylint: disable=C0114
# pylint: disable=C0301
# pylint: disable=R0902
# pylint: disable=R0913
# pylint: disable=R0917
# pylint: disable=W0221

from typing import List, Type, Optional, Final

from ragas import (
    evaluate,
    RunConfig
)
from ragas.dataset_schema import (
    SingleTurnSample,
    EvaluationDataset
)
from ragas.prompt import PydanticPrompt
from ragas.metrics._context_precision import (
    QAC,
    Verification,
    ContextPrecisionPrompt,
    LLMContextPrecisionWithReference,
)
from ragas.llms import BaseRagasLLM
from ragas.evaluation import EvaluationResult

from deepeval.metrics import BaseMetric
from deepeval.telemetry import capture_metric_type
from deepeval.metrics.utils import check_llm_test_case_params
from deepeval.metrics.ragas import format_ragas_metric_name, import_ragas
from deepeval.test_case import LLMTestCase, LLMTestCaseParams, ConversationalTestCase

RAGAS_METRIC_NAME: Final[Type[str]] = "context_precision"

class CustomRAGASContextualPrecisionMetric(BaseMetric):
    """
    Custom implementation of the RAGAS metric for contextual precision.
    Ability to pass custom template and additional parameters.
    """

    _required_params: List[LLMTestCaseParams] = [
        LLMTestCaseParams.INPUT,
        LLMTestCaseParams.EXPECTED_OUTPUT,
        LLMTestCaseParams.RETRIEVAL_CONTEXT
    ]

    def __init__(
        self,
        model: BaseRagasLLM,
        run_config: Optional[RunConfig] = None,
        threshold: float = 0.3,
        context_precision_prompt: PydanticPrompt[
            QAC, Verification
        ] = ContextPrecisionPrompt(),
        max_retries: int = 1,
        experiment_name: Optional[str] = None,
        _track: bool = True,
    ):
        # Verify if ragas is installed and is atleast version `0.2.1`
        import_ragas()

        self.model = model
        self.run_config = run_config
        self.threshold = threshold
        self.context_precision_prompt = context_precision_prompt
        self.max_retries = max_retries
        self.experiment_name = experiment_name
        self.score = 0.0
        self.success = False
        self._track = _track

    async def a_measure(
        self, test_case: LLMTestCase, _show_indicator: bool = False
    ) -> float:
        return self.measure(test_case)

    def measure(self, test_case: LLMTestCase):
        if isinstance(test_case, ConversationalTestCase):
            raise ValueError("ConversationalTestCase is not supported!")

        check_llm_test_case_params(
            test_case, self._required_params, self
        )

        sample = SingleTurnSample(
            user_input=test_case.input,
            reference=test_case.expected_output,
            retrieved_contexts=test_case.retrieval_context
        )
        dataset = EvaluationDataset(samples=[sample])

        with capture_metric_type(
            self.__name__, _track=self._track, async_mode=False
        ):
            scores: EvaluationResult = evaluate(
                dataset,
                metrics=[
                    LLMContextPrecisionWithReference(
                        name=RAGAS_METRIC_NAME,
                        context_precision_prompt=self.context_precision_prompt,
                        max_retries=self.max_retries
                    )
                ],
                llm=self.model,
                experiment_name=self.experiment_name,
                run_config=self.run_config,
                show_progress=True
            )
            contextual_precision_score = scores[RAGAS_METRIC_NAME][0]
            self.success = contextual_precision_score >= self.threshold
            self.score = contextual_precision_score
            return self.score

    def is_successful(self):
        return self.success

    @property
    def __name__(self):
        return format_ragas_metric_name("Contextual Precision")
