# pylint: disable=C0114
# pylint: disable=C0115

from typing import List

from pydantic import BaseModel

# Statements will be inferred from the `expected output`
class Statement(BaseModel):
    text: str

class Statements(BaseModel):
    statements: List[Statement]

class ContextRecallVerdict(BaseModel):
    statement: Statement
    attributed: bool
    reason: str

class ContextRecallVerdicts(BaseModel):
    verdicts: List[ContextRecallVerdict]
