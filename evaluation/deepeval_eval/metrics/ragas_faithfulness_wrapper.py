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
from ragas.metrics._faithfulness import (
    Faithfulness,
    NLIStatementInput,
    NLIStatementOutput,
    NLIStatementPrompt,
    StatementGeneratorInput,
    StatementGeneratorOutput,
    StatementGeneratorPrompt
)
from ragas.llms import LangchainLLMWrapper
from ragas.evaluation import EvaluationResult

from deepeval.metrics import BaseMetric
from deepeval.telemetry import capture_metric_type
from deepeval.metrics.utils import check_llm_test_case_params
from deepeval.test_case import LLMTestCase, LLMTestCaseParams
from deepeval.metrics.ragas import format_ragas_metric_name, import_ragas

class CustomRAGASFaithfulnessMetric(BaseMetric):
    """
    Custom implementation of the RAGAS metric for statement faithfulness.
    Ability to pass custom template and additional parameters.
    """

    # Required parameters for the metric
    # Under: `/ragas/metrics/_answer_relevance.py` check out the metric definition
    _required_params: List[LLMTestCaseParams] = [
        LLMTestCaseParams.INPUT,
        LLMTestCaseParams.ACTUAL_OUTPUT,
        LLMTestCaseParams.RETRIEVAL_CONTEXT
    ]

    def __init__(
        self,
        ragas_llm: LangchainLLMWrapper,
        run_config: Optional[RunConfig] = None,
        threshold: float = 0.3,
        nli_statements_prompt: Type[
            PydanticPrompt[NLIStatementInput, NLIStatementOutput]
        ] = NLIStatementPrompt(),
        statement_generator_prompt: Type[
            PydanticPrompt[StatementGeneratorInput, StatementGeneratorOutput]
        ] = StatementGeneratorPrompt(),
        max_retries: 'int' = 1,
        experiment_name: Optional[str] = None,
        _track: bool = True,
    ):
        # Verify if ragas is installed and is atleast version `0.2.1`
        import_ragas()

        self.model = ragas_llm
        self.run_config = run_config
        self.threshold = threshold
        self.nli_statements_prompt = nli_statements_prompt
        self.statement_generator_prompt = statement_generator_prompt
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
        check_llm_test_case_params(
            test_case, self._required_params, self
        )

        sample = SingleTurnSample(
            user_input=test_case.input,
            response=test_case.actual_output,
            retrieved_contexts=test_case.retrieval_context
        )
        dataset = EvaluationDataset(samples=[sample])

        with capture_metric_type(
            self.__name__, _track=self._track, async_mode=False
        ):
            scores: EvaluationResult = evaluate(
                dataset,
                metrics=[
                    Faithfulness(
                        nli_statements_prompt=self.nli_statements_prompt,
                        statement_generator_prompt=self.statement_generator_prompt,
                        max_retries=self.max_retries
                    )
                ],
                llm=self.model,
                experiment_name=self.experiment_name,
                run_config=self.run_config,
                show_progress=True
            )
            faithfulness_score = scores["faithfulness"][0]
            self.success = faithfulness_score >= self.threshold
            self.score = faithfulness_score
            return self.score

    def is_successful(self):
        return self.success

    @property
    def __name__(self):
        return format_ragas_metric_name("Faithfulness")
