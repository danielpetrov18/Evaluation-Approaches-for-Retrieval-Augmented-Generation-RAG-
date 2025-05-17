# pylint: disable=C0114
# pylint: disable=C0115
# pylint: disable=C0116
# pylint: disable=C0301
# pylint: disable=C0325

import logging
from typing import Any, List, Dict

from opik import exceptions
from opik.evaluation.metrics.llm_judges import parsing_helpers

LOGGER = logging.getLogger(__name__)

def parse_statements(content: str) -> List[Dict[str, str]]:
    try:
        dict_content: Any = parsing_helpers.extract_json_content_or_raise(content)
        statements: List[Dict[str, str]] = dict_content["statements"]
        return statements
    except Exception as e:
        LOGGER.error("Failed to parse model output: %s", e, exc_info=True)
        raise exceptions.MetricComputationError(
            "Failed to extract statements from LLM output!"
        )

def parse_classified_statements(content: str) -> List[Dict[str, Any]]:
    try:
        dict_content: Any = parsing_helpers.extract_json_content_or_raise(content)
        classified_statements: List[Dict[str, Any]] = dict_content["verdicts"]
        return classified_statements
    except Exception as e:
        LOGGER.error("Failed to parse model output: %s", e, exc_info=True)
        raise exceptions.MetricComputationError(
            "Failed to extract classfied statements from LLM output!"
        )
