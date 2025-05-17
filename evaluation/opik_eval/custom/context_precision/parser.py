# pylint: disable=C0114
# pylint: disable=C0115
# pylint: disable=C0116
# pylint: disable=C0301
# pylint: disable=C0325

import logging
from typing import Any, Type, List, Dict, Literal

from opik import exceptions, logging_messages
from opik.evaluation.metrics.score_result import ScoreResult
from opik.evaluation.metrics.llm_judges import parsing_helpers

LOGGER = logging.getLogger(__name__)

def parse_model_output(content: Type[str], name: Type[str]) -> ScoreResult:
    try:
        dict_content: Type[Any] = parsing_helpers.extract_json_content_or_raise(content)
        verdicts: List[Dict[Literal["yes", "no"], str]] = dict_content["verdicts"]

        # Computes the weighted context precision
        score: Type[float] = _compute_wcp(verdicts=verdicts)

        if not (0.0 <= score <= 1.0):
            raise exceptions.MetricComputationError(
                f"Context precision score must be between 0.0 and 1.0, got {score}"
            )

        return ScoreResult(
            name=name, value=score,
        )
    except Exception as e:
        LOGGER.error("Failed to parse model output: %s", e, exc_info=True)
        raise exceptions.MetricComputationError(
            logging_messages.CONTEXT_PRECISION_SCORE_CALC_FAILED
        )

def _compute_wcp(verdicts: List[Dict[Literal["yes", "no"], str]]) -> Type[float]:
    node_verdicts: List[Type[int]] = [1 if ver['verdict'].strip().lower() == "yes" else 0 for ver in verdicts]

    sum_weighted_precision_at_k: Type[float] = 0.0
    relevant_nodes_count: Type[int] = 0
    for k, is_relevant in enumerate(node_verdicts, start=1):
        # If the item is relevant, update the counter and add the weighted precision at k to the sum
        if is_relevant:
            relevant_nodes_count += 1
            precision_at_k = relevant_nodes_count / k
            sum_weighted_precision_at_k += precision_at_k * is_relevant

    # Careful when trying to divide by 0
    if relevant_nodes_count == 0:
        return 0

    # Calculate weighted cumulative precision
    score: Type[float] = sum_weighted_precision_at_k / relevant_nodes_count
    return score
