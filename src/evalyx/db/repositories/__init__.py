"""Lightweight repository layer for Evalyx domain data access.

Repositories accept an ``AsyncSession`` explicitly (session-per-operation
managed by the caller via :class:`evalyx.db.session.DatabaseManager`) and
contain no hidden global state.
"""

from evalyx.db.repositories.application import ApplicationRepository
from evalyx.db.repositories.dataset import DatasetRepository
from evalyx.db.repositories.errors import (
    DuplicateVersionError,
    NotFoundError,
    RepositoryError,
)
from evalyx.db.repositories.evaluation import EvaluationRepository

__all__ = [
    "ApplicationRepository",
    "DatasetRepository",
    "DuplicateVersionError",
    "EvaluationRepository",
    "NotFoundError",
    "RepositoryError",
]
