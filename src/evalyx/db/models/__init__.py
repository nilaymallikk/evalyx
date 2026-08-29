"""Evalyx domain models.

Importing this package registers every model on ``Base.metadata`` so that
Alembic autogenerate sees the complete schema.
"""

from evalyx.db.models.application import Application, ApplicationVersion
from evalyx.db.models.base import Base
from evalyx.db.models.dataset import Dataset, DatasetVersion, TestCase
from evalyx.db.models.evaluation import (
    CaseStatus,
    EvaluationCaseResult,
    EvaluationRun,
    RunStatus,
)
from evalyx.db.models.guardrail import GuardrailResult, GuardrailStatus

__all__ = [
    "Application",
    "ApplicationVersion",
    "Base",
    "CaseStatus",
    "Dataset",
    "DatasetVersion",
    "EvaluationCaseResult",
    "EvaluationRun",
    "GuardrailResult",
    "GuardrailStatus",
    "RunStatus",
    "TestCase",
]
