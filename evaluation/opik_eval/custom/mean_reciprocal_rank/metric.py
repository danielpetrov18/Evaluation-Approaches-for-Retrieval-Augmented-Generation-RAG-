# pylint: disable=C0114
# pylint: disable=C0301
# pylint: disable=R0913
# pylint: disable=R0917

from typing import Callable, List, Optional

import numpy as np
from opik.evaluation.metrics.score_result import ScoreResult
from opik.evaluation.metrics.aggregated_metric import AggregatedMetric

from .reciprocal_rank.metric import ReciprocalRank

def mean_aggregator(score_results: List[ScoreResult]) -> ScoreResult:
    """
    Aggregator function that calculates the mean of multiple ScoreResult values.
    
    Args:
        score_results: List of ScoreResult objects
        
    Returns:
        ScoreResult: A new ScoreResult with the mean value
    """
    if not score_results:
        return ScoreResult(
            name="mean_reciprocal_rank",
            value=0.0,
            reason="No score results to aggregate"
        )

    # Extract values from all score results
    values: List[float] = [result.value for result in score_results]
    mean_value: float = float(np.mean(values)) # Take the average to compute MRR

    return ScoreResult(
        name="mean_reciprocal_rank",
        value=mean_value
    )

class MeanReciprocalRank(AggregatedMetric):
    """
    Custom implementation of the MRR metric.
    
    For each data entry in our dataset, we will create a `Reciprocal Rank`, which has a `score` method.
    Then we will pass all those instances to this metric and using the `aggregator` we will get a final score.
    The `aggregator` is a function that takes a list of scores and returns a single score, by taking the average.
    """

    def __init__(
        self,
        name: str,
        metrics: Optional[List[ReciprocalRank]],
        aggregator:Optional[
            Callable[[List[ScoreResult]], ScoreResult]
        ] = mean_aggregator,
        track: bool = True,
        project_name: Optional[str] = None,
    ):
        super().__init__(
            name=name,
            metrics=metrics,
            aggregator=aggregator,
            track=track,
            project_name=project_name
        )
