# pylint: disable=C0114
# pylint: disable=C0115
# pylint: disable=C0301
# pylint: disable=C0325
# pylint: disable=W0221
# pylint: disable=W0622
# pylint: disable=R0913
# pylint: disable=R0917

from typing import Any, List, Optional, Union, Dict

from opik.evaluation.metrics.base_metric import BaseMetric
from opik.evaluation.metrics.score_result import ScoreResult
from opik.evaluation.models import base_model, models_factory

from .template import (
    generate_decomposition_query,
    generate_query,
    FewShotExampleStatements,
    FewShotExampleContextRecall,
)
from .models import Statements, Statement, ContextRecallVerdicts
from .parser import parse_statements, parse_classified_statements

class ContextRecall(BaseMetric):
    """
    A metric that evaluates the context recall of an input-output pair using an LLM.

    We first decompose the `expected output` into separate statements.
    Then we use the LLM to determine if each of those statements can be attributed to a node from the `retrieved context`.
    The final formula would be:
    * (number statements supported by context) / (total number statements in the expected output)

    Args:
        model: The language model to use for evaluation. Can be a string (model name) or an `opik.evaluation.models.OpikBaseModel` subclass instance.
            `opik.evaluation.models.LiteLLMChatModel` is used by default.
        name: The name of the metric. Defaults to "ContextRecallMetric".
        few_shot_examples_statements: A list of few-shot examples to provide to the model for decomposing the `expected output`.
        few_shot_examples_context_recall: A list of few-shot examples to provide to the model for classifying each statement.
        track: Whether to track the metric. Defaults to True.
        project_name: Optional project name to track the metric in for the cases when
            there are no parent span/trace to inherit project name from.
    """

    def __init__(
        self,
        model: Optional[Union[str, base_model.OpikBaseModel]] = None,
        name: str = "context_recall_metric",
        few_shot_examples_statements: Optional[
            List[FewShotExampleStatements]
        ] = None,
        few_shot_examples_context_recall: Optional[
            List[FewShotExampleContextRecall]
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
        self.few_shot_examples_statements = few_shot_examples_statements
        self.few_shot_examples_context_recall = few_shot_examples_context_recall

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
        Calculate the context recall score for the given input-output pair.

        Args:
            input: The input text to be evaluated.
            expected_output: The expected output for the given input.
            context: A list of context strings relevant to the input.
            **ignored_kwargs: Additional keyword arguments that are ignored.

        Returns:
            score_result.ScoreResult: A ScoreResult object containing the context recall score
            (between 0.0 and 1.0).
        """
        # Create the decomposition query
        llm_statement_decomposition_query: str = generate_decomposition_query(
            expected_output=expected_output,
            few_shot_examples=self.few_shot_examples_statements,
        )

        # Retrive the LLM response 
        statements_output: str = self._model.generate_string(
            input=llm_statement_decomposition_query, response_format=Statements
        )

        # Parse and extract the statements
        statements: List[Dict[str, str]] = parse_statements(
            content=statements_output,
        )

        # Convert statements to pydantic objects
        statements: List[Statement] = [
            Statement(**statement) for statement in statements
        ]

        # Generate context recall prompt 
        llm_context_recall_query: str = generate_query(
            input=input,
            statements=statements,
            context=context,
            few_shot_examples=self.few_shot_examples_context_recall,
        )

        # Submit the prompt and retrieve LLM answer
        model_output: str = self._model.generate_string(
            input=llm_context_recall_query, response_format=ContextRecallVerdicts
        )

        # Parse the classification output
        classified_statements: List[Dict[str, Any]] = parse_classified_statements(
            content=model_output
        )

        # Finally compute the score
        score: float = self._compute_score(classified_statements)
        return ScoreResult(
            name=self.name,
            value=score,
        )

    async def ascore(
        self,
        input: str,
        expected_output: str,
        context: List[str],
        **ignored_kwargs: Any,
    ) -> ScoreResult:
        """
        Asynchronously calculate the context recall score for the given input-output pair.

        This method is the asynchronous version of :meth:`score`. For detailed documentation,
        please refer to the :meth:`score` method.

        Args:
            input: The input text to be evaluated.
            expected_output: The expected output for the given input.
            context: A list of context strings relevant to the input.
            **ignored_kwargs: Additional keyword arguments that are ignored.

        Returns:
            score_result.ScoreResult: A ScoreResult object with the context recall score.
        """
        llm_statement_decomposition_query: str = generate_decomposition_query(
            expected_output=expected_output,
            few_shot_examples=self.few_shot_examples_statements,
        )

        statements_output: str = await self._model.agenerate_string(
            input=llm_statement_decomposition_query, response_format=Statements
        )

        statements: List[Dict[str, str]] = parse_statements(
            content=statements_output,
        )

        statements: List[Statement] = [
            Statement(**statement) for statement in statements
        ]

        llm_context_recall_query: str = generate_query(
            input=input,
            statements=statements,
            context=context,
            few_shot_examples=self.few_shot_examples_context_recall,
        )

        model_output: str = await self._model.agenerate_string(
            input=llm_context_recall_query, response_format=ContextRecallVerdicts
        )

        classified_statements: List[Dict[str, Any]] = parse_classified_statements(
            content=model_output
        )

        score: float = self._compute_score(classified_statements)
        return ScoreResult(
            name=self.name,
            value=score,
        )

    def _compute_score(self, classified_statements: List[Dict[str, Any]]) -> float:
        all_statements: int = len(classified_statements)
        if all_statements == 0:
            return 0

        attributed_statements: int = 0
        for statement in classified_statements:
            if statement["attributed"]:
                attributed_statements += 1

        score: float = attributed_statements / all_statements
        return score
