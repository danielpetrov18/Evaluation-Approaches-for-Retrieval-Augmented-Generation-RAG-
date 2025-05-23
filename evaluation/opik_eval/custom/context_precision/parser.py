# pylint: disable=C0114
# pylint: disable=C0115
# pylint: disable=C0116
# pylint: disable=C0301
# pylint: disable=C0325

import logging
from typing import Any, List, Dict, Literal

from opik import exceptions, logging_messages
from opik.evaluation.metrics.score_result import ScoreResult
from opik.evaluation.metrics.llm_judges import parsing_helpers

LOGGER = logging.getLogger(__name__)

def parse_model_output(content: str, name: str) -> ScoreResult:
    try:
        # Try to extract verdicts from JSON output
        dict_content: Any = parsing_helpers.extract_json_content_or_raise(content)
        verdicts: List[Dict[Literal["yes", "no"], str]] = dict_content["verdicts"]

        # Computes the weighted context precision
        # https://docs.ragas.io/en/latest/concepts/metrics/available_metrics/context_precision/
        score: float = _compute_wcp(verdicts)

        if not (0.0 <= score <= 1.0):
            raise exceptions.MetricComputationError(
                f"Context precision score must be between 0.0 and 1.0, got {score}"
            )

        return ScoreResult(
            name=name, value=score
        )
    except Exception as e:
        LOGGER.error("Failed to parse model output: %s", e, exc_info=True)
        raise exceptions.MetricComputationError(
            logging_messages.CONTEXT_PRECISION_SCORE_CALC_FAILED
        )

def _compute_wcp(verdicts: List[Dict[Literal["yes", "no"], str]]) -> float:
    node_verdicts: List[int] = [
        1 if ver['verdict'].strip().lower() == "yes" else 0 # If verdict=="yes", return 1, 0 otherwise
        for ver in verdicts # Go over all verdicts
    ]

    sum_weighted_precision_at_k: float = 0.0
    relevant_nodes_count: int = 0
    for k, is_relevant in enumerate(node_verdicts, start=1):
        # If the item is relevant, update the counter and add the weighted precision at k to the sum
        if is_relevant: # The verdict is "yes" / 1
            relevant_nodes_count += 1
            precision_at_k: float = relevant_nodes_count / k # Precision@k, k is the rank
            sum_weighted_precision_at_k += precision_at_k * is_relevant # Consider value only if relevant

    # Careful when trying to divide by 0
    if relevant_nodes_count == 0:
        return 0

    # Calculate weighted cumulative precision
    score: float = sum_weighted_precision_at_k / relevant_nodes_count
    return score
