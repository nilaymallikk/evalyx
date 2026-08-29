"""Regression service: orchestration around the pure comparison engine.

Responsibilities (and nothing else):

1. load the baseline/current runs
2. validate comparison compatibility (typed, clear errors)
3. load case/guardrail results and dataset-version case identities
4. invoke the pure comparison (:func:`evalyx.evaluation.regression.comparison.compare`)
5. persist the artifact idempotently
6. return the typed report

The service never modifies evaluation history: runs, case results, and
guardrail results are read-only here. Only ``regression_comparisons`` rows
are created.

Compatibility policy (MVP, deliberately strict):

- a run cannot be compared with itself
- both runs must be ``completed`` (pending/running/failed/cancelled are
  rejected — partial results would produce misleading metrics)
- both runs must belong to the same ``Application``
- both runs must sit on the same dataset *lineage* (same ``Dataset``);
  different versions of that dataset are allowed

Case identity across runs:

- same dataset version  → ``test_case_id`` (strongest identity)
- different versions of one dataset → test-case ``name`` (each version
  snapshots new ``TestCase`` rows, so names are the stable cross-version
  identity; this requires names to be stable when a dataset evolves —
  documented in the README)

Idempotency: the artifact is keyed by
``(baseline_run_id, current_run_id, policy_fingerprint)`` where the
fingerprint covers the threshold snapshot + comparison version. Comparing
the same pair with the same policy returns the *original* artifact (same
id and created_at). A different threshold policy creates a distinct,
coexisting artifact. Persisted thresholds make every decision
reproducible without relying on current configuration.
"""

import uuid
from collections import defaultdict

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from evalyx.db.models import (
    Dataset,
    DatasetVersion,
    EvaluationRun,
    RegressionComparison,
    RunStatus,
    TestCase,
)
from evalyx.db.repositories import (
    EvaluationRepository,
    NotFoundError,
    RegressionRepository,
)
from evalyx.evaluation.regression.comparison import (
    ComparisonInput,
    compare,
    diff_configurations,
    policy_fingerprint,
    sanitize_configuration,
)
from evalyx.evaluation.regression.models import (
    CaseResultSnapshot,
    ComparisonContext,
    ComparisonRunContext,
    GuardrailResultSnapshot,
    RegressionReport,
    RegressionThresholds,
)

__all__ = [
    "RegressionError",
    "RegressionService",
    "RegressionValidationError",
    "comparison_to_report",
]


class RegressionError(Exception):
    """Base class for regression-service errors."""


class RegressionValidationError(RegressionError):
    """The two runs cannot be meaningfully compared (clear, typed reason)."""


class RegressionService:
    """Orchestrates regression comparisons over persisted evaluation data."""

    def __init__(self, session_factory: async_sessionmaker[AsyncSession]) -> None:
        self._session_factory = session_factory
        self._evaluations = EvaluationRepository()
        self._regressions = RegressionRepository()

    # -- public API ----------------------------------------------------------

    async def compare_runs(
        self,
        baseline_run_id: uuid.UUID,
        current_run_id: uuid.UUID,
        thresholds: RegressionThresholds | None = None,
    ) -> RegressionReport:
        """Compare two completed runs and return the typed regression report.

        Repeated calls with the same pair and the same threshold policy are
        idempotent: the original artifact is returned unchanged.
        """
        thresholds = thresholds or RegressionThresholds()
        fingerprint = policy_fingerprint(thresholds)

        async with self._session_factory() as session:
            input_data = await self._build_comparison_input(
                session, baseline_run_id, current_run_id
            )
            report = compare(input_data, thresholds)

            existing = await self._regressions.get_by_pair_and_policy(
                session,
                baseline_run_id=baseline_run_id,
                current_run_id=current_run_id,
                policy_fingerprint=fingerprint,
            )
            if existing is not None:
                report.comparison_id = existing.id
                report.created_at = existing.created_at
                return report

            comparison = await self._regressions.create(
                session,
                baseline_run_id=baseline_run_id,
                current_run_id=current_run_id,
                result=report.result,
                regression_detected=report.regression_detected,
                comparison_version=report.comparison_version,
                policy_fingerprint=fingerprint,
                thresholds=thresholds.model_dump(),
                summary=_summary_payload(report),
            )
            report.comparison_id = comparison.id
            report.created_at = comparison.created_at
            return report

    async def get_report(self, comparison_id: uuid.UUID) -> RegressionReport:
        """Reload a persisted comparison as a typed report."""
        async with self._session_factory() as session:
            comparison = await self._regressions.get(session, comparison_id)
        if comparison is None:
            raise NotFoundError(f"Regression comparison {comparison_id} does not exist.")
        return comparison_to_report(comparison)

    async def list_for_run(self, run_id: uuid.UUID) -> list[RegressionReport]:
        """All comparisons involving ``run_id`` (either side), newest first."""
        async with self._session_factory() as session:
            comparisons = await self._regressions.list_for_run(session, run_id)
        return [comparison_to_report(comparison) for comparison in comparisons]

    # -- loading & validation -------------------------------------------------

    async def _build_comparison_input(
        self,
        session: AsyncSession,
        baseline_run_id: uuid.UUID,
        current_run_id: uuid.UUID,
    ) -> ComparisonInput:
        if baseline_run_id == current_run_id:
            raise RegressionValidationError(
                "A run cannot be compared with itself "
                f"({baseline_run_id} was given as both baseline and current)."
            )

        baseline_run = await self._evaluations.get_run(session, baseline_run_id)
        if baseline_run is None:
            raise NotFoundError(f"Evaluation run {baseline_run_id} does not exist.")
        current_run = await self._evaluations.get_run(session, current_run_id)
        if current_run is None:
            raise NotFoundError(f"Evaluation run {current_run_id} does not exist.")

        self._validate_runs(baseline_run, current_run)

        baseline_version = await session.get(DatasetVersion, baseline_run.dataset_version_id)
        current_version = await session.get(DatasetVersion, current_run.dataset_version_id)
        assert baseline_version is not None and current_version is not None  # FK-guaranteed
        if baseline_version.dataset_id != current_version.dataset_id:
            raise RegressionValidationError(
                "Runs reference different datasets "
                f"({baseline_version.dataset_id} vs {current_version.dataset_id}) and "
                "cannot be compared. Comparisons require the same dataset lineage."
            )
        baseline_dataset = await session.get(Dataset, baseline_version.dataset_id)
        current_dataset = await session.get(Dataset, current_version.dataset_id)
        assert baseline_dataset is not None and current_dataset is not None

        same_version = baseline_version.id == current_version.id
        strategy = "test_case_id" if same_version else "case_name"

        baseline_cases, baseline_dataset_cases = await self._load_case_snapshots(
            session, baseline_run, strategy
        )
        current_cases, current_dataset_cases = await self._load_case_snapshots(
            session, current_run, strategy
        )

        return ComparisonInput(
            baseline_cases=baseline_cases,
            current_cases=current_cases,
            context=ComparisonContext(
                baseline=self._run_context(baseline_run, baseline_version, baseline_dataset),
                current=self._run_context(current_run, current_version, current_dataset),
                agent_model_changed=baseline_run.agent_model != current_run.agent_model,
                judge_model_changed=baseline_run.judge_model != current_run.judge_model,
                application_version_changed=(
                    baseline_run.application_version_id != current_run.application_version_id
                ),
                configuration_changes=diff_configurations(
                    baseline_run.configuration_snapshot, current_run.configuration_snapshot
                ),
            ),
            baseline_dataset_cases=baseline_dataset_cases,
            current_dataset_cases=current_dataset_cases,
        )

    @staticmethod
    def _validate_runs(baseline_run: EvaluationRun, current_run: EvaluationRun) -> None:
        non_completed = [
            f"{role} run {run.id} has status {run.status.value!r}"
            for role, run in (("baseline", baseline_run), ("current", current_run))
            if run.status is not RunStatus.COMPLETED
        ]
        if non_completed:
            raise RegressionValidationError(
                "Both runs must be completed to be compared; "
                + "; ".join(non_completed)
                + ". Partial or non-executed runs would produce misleading metrics."
            )
        if baseline_run.application_id != current_run.application_id:
            raise RegressionValidationError(
                "Runs belong to different applications "
                f"({baseline_run.application_id} vs {current_run.application_id}) and "
                "cannot be compared."
            )

    @staticmethod
    def _run_context(
        run: EvaluationRun, version: DatasetVersion, dataset: Dataset
    ) -> ComparisonRunContext:
        return ComparisonRunContext(
            run_id=run.id,
            status=run.status.value,
            agent_model=run.agent_model,
            judge_model=run.judge_model,
            application_id=run.application_id,
            application_version_id=run.application_version_id,
            dataset_version_id=run.dataset_version_id,
            dataset_version=version.version,
            dataset_name=dataset.name,
            configuration_snapshot=sanitize_configuration(run.configuration_snapshot),
        )

    @staticmethod
    async def _load_case_snapshots(
        session: AsyncSession,
        run: EvaluationRun,
        strategy: str,
    ) -> tuple[list[CaseResultSnapshot], dict[str, str]]:
        """Case snapshots for a run plus the dataset version's identity→name map.

        ``strategy`` is ``"test_case_id"`` (same version) or ``"case_name"``
        (cross-version within one dataset).
        """
        dataset_cases = (
            (await session.execute(
                select(TestCase).where(TestCase.dataset_version_id == run.dataset_version_id)
            ))
            .scalars()
            .all()
        )
        names_by_case_id = {case.id: case.name for case in dataset_cases}
        if strategy == "test_case_id":
            identities = {str(case_id): name for case_id, name in names_by_case_id.items()}
        else:
            identities = {name: name for name in names_by_case_id.values()}

        case_results = await EvaluationRepository().list_case_results(session, run.id)
        guardrails_by_case: dict[uuid.UUID, list[GuardrailResultSnapshot]] = defaultdict(list)
        for guardrail in await EvaluationRepository().list_guardrail_results_for_run(
            session, run.id
        ):
            guardrails_by_case[guardrail.evaluation_case_result_id].append(
                GuardrailResultSnapshot(name=guardrail.name, status=guardrail.status)
            )

        snapshots: list[CaseResultSnapshot] = []
        for case_result in case_results:
            name = names_by_case_id.get(case_result.test_case_id, str(case_result.test_case_id))
            identity = (
                str(case_result.test_case_id) if strategy == "test_case_id" else name
            )
            snapshots.append(
                CaseResultSnapshot(
                    identity=identity,
                    case_result_id=case_result.id,
                    test_case_id=case_result.test_case_id,
                    name=name,
                    status=case_result.status,
                    latency_ms=case_result.latency_ms,
                    guardrails=guardrails_by_case.get(case_result.id, []),
                )
            )
        return snapshots, identities


def _summary_payload(report: RegressionReport) -> dict:
    """JSON-safe report content for the ``summary`` JSONB column.

    ``comparison_id``/``created_at`` are excluded — they are columns on the
    artifact row and are re-injected when the report is reloaded.
    """
    payload = report.model_dump(mode="json")
    payload.pop("comparison_id", None)
    payload.pop("created_at", None)
    return payload


def comparison_to_report(comparison: RegressionComparison) -> RegressionReport:
    """Rebuild the typed report from a persisted artifact."""
    payload = dict(comparison.summary)
    payload["comparison_id"] = comparison.id
    payload["created_at"] = comparison.created_at
    payload["comparison_version"] = comparison.comparison_version
    return RegressionReport.model_validate(payload)
