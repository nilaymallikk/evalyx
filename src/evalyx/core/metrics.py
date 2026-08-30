"""Lightweight in-process metrics registry.

A small, dependency-free, thread-safe metrics foundation for Evalyx. It
supports **counters** and **timing observations** (histogram-style stats)
with bounded label sets. No external telemetry backend is required: values
live in process memory and are read via :meth:`MetricsRegistry.snapshot`
(for tests and local debugging). A future Prometheus/OpenTelemetry phase can
export the same registry without touching instrumented code.

Cardinality policy (production-engineering constraint, enforced):

- Labels must be **bounded**: method, route template, status code, task
  name, outcome, guardrail name, evaluation status. They are caller-supplied
  strings; call sites are reviewed and use only known-bounded values.
- Correlation identifiers are **forbidden as label keys** — ``request_id``,
  ``run_id``, ``task_id`` and friends would create unbounded label
  cardinality (one time series per id). The registry raises
  :class:`ValueError` if a call site ever tries.

Operational notes:

- Thread-safe via a single :class:`threading.Lock`; updates are pure
  in-memory dict operations (no I/O), so they are also safe to call from
  async code on the event loop.
- The module-level :data:`metrics` registry is the single shared instance
  for the API and worker processes. :meth:`MetricsRegistry.reset` exists
  for **tests only**; it is never exposed over HTTP.
"""

import threading
from dataclasses import dataclass
from typing import Final

# Label keys that would create unbounded cardinality. Correlation ids belong
# in logs, never in metric labels.
FORBIDDEN_LABELS: Final[frozenset[str]] = frozenset(
    {
        "request_id",
        "run_id",
        "task_id",
        "case_id",
        "test_case_id",
        "comparison_id",
        "application_id",
        "dataset_version_id",
        "user_id",
        "prompt",
        "output",
    }
)


@dataclass
class _Timing:
    """Aggregate statistics for one labelled timing series."""

    count: int = 0
    total_ms: float = 0.0
    max_ms: float = 0.0

    def observe(self, value_ms: float) -> None:
        self.count += 1
        self.total_ms += value_ms
        self.max_ms = max(self.max_ms, value_ms)


_LabelKey = tuple[str, tuple[tuple[str, str], ...]]
"""Metric identity: (name, sorted label items)."""


class MetricsRegistry:
    """Counters and timing observations with bounded labels.

    Keys are ``(name, sorted-label-items)`` pairs; snapshots are rendered
    deterministically (sorted by name, then label key/value) so tests can
    assert on exact structure.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._counters: dict[_LabelKey, float] = {}
        self._timings: dict[_LabelKey, _Timing] = {}

    # -- validation ----------------------------------------------------------

    @staticmethod
    def _check(name: str, labels: dict[str, str] | None) -> _LabelKey:
        if not name:
            raise ValueError("metric name must be non-empty")
        if labels is None:
            return (name, ())
        items: list[tuple[str, str]] = []
        for key, value in labels.items():
            if not isinstance(key, str) or not isinstance(value, str):
                raise TypeError("metric labels must be strings")
            if key in FORBIDDEN_LABELS:
                raise ValueError(
                    f"metric label {key!r} is forbidden: correlation identifiers "
                    "create unbounded cardinality; they belong in logs only"
                )
            items.append((key, value))
        return (name, tuple(sorted(items)))

    # -- recording API -------------------------------------------------------

    def increment(
        self,
        name: str,
        labels: dict[str, str] | None = None,
        value: float = 1,
    ) -> None:
        """Add ``value`` to the counter ``name`` (created at zero)."""
        key = self._check(name, labels)
        with self._lock:
            self._counters[key] = self._counters.get(key, 0) + value

    def observe(
        self,
        name: str,
        value_ms: float,
        labels: dict[str, str] | None = None,
    ) -> None:
        """Record one timing observation (milliseconds) for ``name``."""
        key = self._check(name, labels)
        with self._lock:
            timing = self._timings.get(key)
            if timing is None:
                timing = self._timings[key] = _Timing()
            timing.observe(value_ms)

    # -- reading API ---------------------------------------------------------

    def snapshot(self) -> dict[str, list[dict[str, object]]]:
        """Return a deterministic, JSON-serializable snapshot.

        ``{"<name>": [{"labels": {...}, "value": n} | {"labels": {...},
        "count": n, "total_ms": x, "max_ms": y, "avg_ms": z}]}`` — counters
        and timings are listed under their metric name, sorted by labels.
        """
        with self._lock:
            counters = dict(self._counters)
            timings = {k: _Timing(**vars(v)) for k, v in self._timings.items()}
        # (label-dict, payload-dict) pairs; sorted by labels for determinism.
        entries: dict[str, list[tuple[dict[str, str], dict[str, object]]]] = {}
        for (name, items), value in counters.items():
            labels = dict(items)
            entries.setdefault(name, []).append(
                (labels, {"labels": labels, "value": value})
            )
        for (name, items), timing in timings.items():
            labels = dict(items)
            avg = timing.total_ms / timing.count if timing.count else 0.0
            entries.setdefault(name, []).append(
                (
                    labels,
                    {
                        "labels": labels,
                        "count": timing.count,
                        "total_ms": timing.total_ms,
                        "max_ms": timing.max_ms,
                        "avg_ms": avg,
                    },
                )
            )
        return {
            name: [payload for _, payload in sorted(es, key=lambda e: sorted(e[0].items()))]
            for name, es in entries.items()
        }

    def reset(self) -> None:
        """Clear all recorded values. **Test isolation only** — never
        exposed over HTTP or called by production code paths."""
        with self._lock:
            self._counters.clear()
            self._timings.clear()


metrics = MetricsRegistry()
"""Shared in-process metrics registry (API + worker)."""
