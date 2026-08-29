"""Repository data-access errors."""

from uuid import UUID


class RepositoryError(Exception):
    """Base class for repository errors."""


class NotFoundError(RepositoryError):
    """A referenced entity does not exist."""


class DuplicateVersionError(RepositoryError):
    """A version identifier already exists for the parent entity."""

    def __init__(self, parent_id: UUID, version: str | int) -> None:
        super().__init__(
            f"Version {version!r} already exists for entity {parent_id}. "
            "Create a new version instead of overwriting."
        )
        self.parent_id = parent_id
        self.version = version
