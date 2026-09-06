"""Local identity: Evalyx serves a single local workspace.

There are no user accounts and no external identity provider. Every
request acts as the local operator; downstream code (services,
repositories, audit) consumes the immutable :class:`AuthContext` exactly
as before, so tenant scoping of all domain reads/writes is unchanged —
there is simply one tenant.
"""

from dataclasses import dataclass

#: The fixed user id recorded in audit events for local operations.
LOCAL_USER_ID = "local"


@dataclass(frozen=True)
class AuthContext:
    """The immutable identity context for one request (always local)."""

    user_id: str = LOCAL_USER_ID

    @property
    def is_authenticated(self) -> bool:
        return True


def local_context() -> AuthContext:
    """The single local-operator context."""
    return AuthContext()


__all__ = [
    "LOCAL_USER_ID",
    "AuthContext",
    "local_context",
]
