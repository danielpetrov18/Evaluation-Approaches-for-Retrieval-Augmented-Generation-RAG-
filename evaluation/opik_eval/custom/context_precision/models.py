# pylint: disable=C0114
# pylint: disable=C0115

from typing import List, Literal

from pydantic import BaseModel

class ContextPrecisionVerdict(BaseModel):
    verdict: Literal["yes", "no"]
    reason: str

class ContextPrecisionVerdicts(BaseModel):
    verdicts: List[ContextPrecisionVerdict]
