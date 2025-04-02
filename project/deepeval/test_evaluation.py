"""
Defining test cases for all metrics and the whole dataset.

How to run the file: deepeval test run <file_name>
If you want to cache results for future re-runs add the `-c` flag.
Additionally, adding the `-id <Test run name>` can be used for easier identification.
"""

import pytest
from deepeval.test_case import (
    LLMTestCase,
    LLMTestCaseParams
)
from deepeval import assert_test
from deepeval.metrics import GEval
from deepeval.models.llms import OllamaModel
from deepeval.dataset import EvaluationDataset

evaluation_dataset = EvaluationDataset()
evaluation_dataset.pull("RAGAs Dataset")

@pytest.mark.parametrize(
    "test_case",
    evaluation_dataset
)
def test_correctness(test_case: LLMTestCase):
    """Runs a single LLMTestCase evaluating the correctness"""
    correctness_metric = GEval(
        name="Correctness",
        criteria="Determine if the 'actual output' is correct based on the 'expected output'.",
        evaluation_params=[LLMTestCaseParams.ACTUAL_OUTPUT, LLMTestCaseParams.EXPECTED_OUTPUT],
        threshold=0.70,
        model=OllamaModel()
    )
    assert_test(test_case, [correctness_metric])
