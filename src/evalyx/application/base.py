"""Application-under-test integration: invoking external AI applications.

Evalyx evaluates *applications*, not only raw model providers. MLGPT (the
reference RAG chatbot) is an **application under test** — a separate system
that Evalyx drives over HTTP. This module defines the minimal abstraction
for that boundary:

- :class:`ApplicationTarget` — the protocol an invoked application satisfies
- :class:`ApplicationResponse` — what an invocation returns (deliberately
  *not* :class:`evalyx.llm.base.LLMResponse`: an application response has no
  token usage or finish reason; it has HTTP-level metadata instead)
- :class:`ApplicationInvocationError` — typed, secret-safe invocation errors

Architectural rules enforced here:

- Evalyx never imports MLGPT internals; the boundary is HTTP only.
- MLGPT is never modelled as an LLM *provider* (it is an application whose
  own configuration decides which model it calls).
- No secrets live in this layer: the target needs only a base URL; the
  anonymous user id is a random, non-PII identifier by MLGPT's own design.
"""

import uuid
from typing import Protocol, runtime_checkable

from pydantic import BaseModel, Field


class ApplicationResponse(BaseModel):
    """One normalized application invocation result.

    ``metadata`` carries only bounded, safe fields (e.g. ``sources_count``,
    ``application``) — never raw retrieved documents, prompts, or outputs
    beyond ``content`` itself.
    """

    model_config = {"extra": "forbid"}

    content: str
    latency_ms: int
    status_code: int | None = None
    metadata: dict = Field(default_factory=dict)


class ApplicationInvocationError(Exception):
    """An application target could not be invoked successfully.

    Safe by construction: messages contain the HTTP status code and error
    *kind* only — never response bodies (which may echo prompts), headers,
    or credentials. The evaluation runner records these as per-case errors
    (like provider errors), so one failed invocation never kills a run.
    """


@runtime_checkable
class ApplicationTarget(Protocol):
    """An external AI application that can be invoked with one prompt.

    Implementations own their transport and configuration; the evaluation
    engine only knows this protocol. ``close`` releases transport resources.
    """

    async def invoke(self, prompt: str) -> ApplicationResponse: ...

    async def close(self) -> None: ...


#: Stable anonymous user id used when driving MLGPT. MLGPT requires a
#: ``user_id`` UUID per request to scope anonymous browser conversations;
#: a fixed generated identifier is not personal data.
ANONYMOUS_EVALUATION_USER_ID: str = "00000000-0000-4000-8000-000000000001"


def create_application_target(name: str, settings) -> ApplicationTarget:
    """Build a registered application target by name.

    The registry is deliberately tiny: adding a future target (Azure-hosted
    app, OpenAI app, custom agent) means adding one branch here — the
    evaluation engine never changes. Unknown names raise
    :class:`ApplicationInvocationError` (the run's target selector is
    invalid, a permanent configuration error).
    """
    # Imported lazily: keeps httpx-dependent transport out of import graphs
    # that only need the protocol.
    from evalyx.application.http import HttpApplicationTarget

    if name == "mlgpt":
        return HttpApplicationTarget(base_url=settings.mlgpt_base_url)
    raise ApplicationInvocationError(
        f"Unknown application target {name!r}. Registered targets: mlgpt."
    )


def application_name_from_model(agent_model: str) -> str | None:
    """Extract the target name from a run's ``application:<name>`` selector.

    Runs evaluated against an application (rather than a raw model) record
    ``agent_model = "application:mlgpt"`` — a bounded selector, not a model
    name. Returns ``None`` for ordinary model strings.
    """
    prefix = "application:"
    if agent_model.startswith(prefix):
        name = agent_model[len(prefix) :].strip()
        return name or None
    return None


__all__ = [
    "ANONYMOUS_EVALUATION_USER_ID",
    "ApplicationInvocationError",
    "ApplicationResponse",
    "ApplicationTarget",
    "application_name_from_model",
    "create_application_target",
    "uuid",
]
