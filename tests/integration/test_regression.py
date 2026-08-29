"""Regression integration tests — live PostgreSQL (5433).

No network and no LLM: runs/case results/guardrail results are seeded
directly through repositories, then compared through RegressionService.
The Celery/Redis layer is deliberately not involved (spec: regression
logic is independent of the worker).
"""

import uuid

import pytest
from sqlalchemy import text
from sqlalchemy.exc import IntegrityError

from evalyx.core.config import (
    Settings,  # noqa: F401 — parity with other integration tests
)
from evalyx.db.models import (
    CaseStatus,
    ComparisonResult,
    GuardrailStatus,
    RunStatus,
)
from evalyx.db.repositories import (
    ApplicationRepository,
    DatasetRepository,
    EvaluationRepository,
    NotFoundError,
    RegressionRepository,
)
from evalyx.db.session import DatabaseManager
from evalyx.evaluation.regression import (
    RegressionService,
    RegressionThresholds,
    RegressionValidationError,
    policy_fingerprint,
)

pytestmark = pytest.mark.integration

DOMAIN_TABLES = (
    "regression_comparisons",
    "guardrail_results",
    "evaluation_case_results",
    "evaluation_runs",
    "test_cases",
    "dataset_versions",
    "datasets",
    "application_versions",
    "applications",
)


async def seed_scenario(
    db: DatabaseManager,
    *,
    with_secrets: bool = False,
) -> tuple[uuid.UUID, uuid.UUID]:
    """Seed the ten-case regression scenario; return (baseline, current) ids.

    Baseline: ok0-6 passed, stable-fail failed, to-fix failed, err error.
    Current:  ok0-4 passed, ok5 failed (new pii failure), ok6 error,
              stable-fail failed, to-fix passed, err error.
    """
    apps, datasets, evaluations = (
        ApplicationRepository(),
        DatasetRepository(),
        EvaluationRepository(),
    )
    async with db.session() as session:
        app = await apps.create(session, name=f"regression-app-{uuid.uuid4().hex[:8]}")
        dataset = await datasets.create(session, name=f"regression-ds-{uuid.uuid4().hex[:8]}")
        version = await datasets.create_version(session, dataset_id=dataset.id, version=1)
        case_ids: dict[str, uuid.UUID] = {}
        for name in [
            *(f"ok{i}" for i in range(7)),
            "stable-fail",
            "to-fix",
            "err",
        ]:
            case = await datasets.add_test_case(
                session, dataset_version_id=version.id, name=name, input={"prompt": name}
            )
            case_ids[name] = case.id

        baseline_config = {"temperature": 0.2, "max_tokens": 512}
        current_config = {"temperature": 0.7, "max_tokens": 512}
        if with_secrets:
            baseline_config["api_key"] = "baseline-secret-value"
            current_config["auth_token"] = "current-secret-value"

        baseline = await evaluations.create_run(
            session,
            application_id=app.id,
            dataset_version_id=version.id,
            agent_model="agent-a",
            judge_model="judge-a",
            configuration_snapshot=baseline_config,
        )
        current = await evaluations.create_run(
            session,
            application_id=app.id,
            dataset_version_id=version.id,
            agent_model="agent-a",
            judge_model="judge-a",
            configuration_snapshot=current_config,
        )

        baseline_outcomes: dict[str, tuple[CaseStatus, int]] = {
            **{f"ok{i}": (CaseStatus.PASSED, 450) for i in range(7)},
            "stable-fail": (CaseStatus.FAILED, 450),
            "to-fix": (CaseStatus.FAILED, 450),
            "err": (CaseStatus.ERROR, 0),
        }
        current_outcomes: dict[str, tuple[CaseStatus, int]] = {
            **{f"ok{i}": (CaseStatus.PASSED, 600) for i in range(5)},
            "ok5": (CaseStatus.FAILED, 600),
            "ok6": (CaseStatus.ERROR, 0),
            "stable-fail": (CaseStatus.FAILED, 600),
            "to-fix": (CaseStatus.PASSED, 600),
            "err": (CaseStatus.ERROR, 0),
        }
        # pii/safety verdicts per case per run; error cases get no rows.
        baseline_guardrails: dict[str, tuple[GuardrailStatus, GuardrailStatus]] = {
            **{f"ok{i}": (GuardrailStatus.PASSED, GuardrailStatus.PASSED) for i in range(7)},
            "stable-fail": (GuardrailStatus.PASSED, GuardrailStatus.FAILED),
            "to-fix": (GuardrailStatus.PASSED, GuardrailStatus.FAILED),
        }
        current_guardrails: dict[str, tuple[GuardrailStatus, GuardrailStatus]] = {
            **{f"ok{i}": (GuardrailStatus.PASSED, GuardrailStatus.PASSED) for i in range(5)},
            "ok5": (GuardrailStatus.FAILED, GuardrailStatus.PASSED),
            "stable-fail": (GuardrailStatus.PASSED, GuardrailStatus.FAILED),
            "to-fix": (GuardrailStatus.PASSED, GuardrailStatus.PASSED),
        }

        for run, outcomes, guardrail_plan in (
            (baseline, baseline_outcomes, baseline_guardrails),
            (current, current_outcomes, current_guardrails),
        ):
            for name, (status, latency) in outcomes.items():
                case_result = await evaluations.add_case_result(
                    session,
                    evaluation_run_id=run.id,
                    test_case_id=case_ids[name],
                    input={"prompt": name},
                    status=status,
                    actual_output=None if status is CaseStatus.ERROR else f"reply:{name}",
                    latency_ms=latency if status is not CaseStatus.ERROR else None,
                    error="provider exploded" if status is CaseStatus.ERROR else None,
                )
                if status is CaseStatus.ERROR:
                    continue  # error cases produce no guardrail rows (harness semantics)
                for index, guardrail_name in enumerate(("pii", "safety")):
                    verdict = guardrail_plan[name][index]
                    await evaluations.add_guardrail_result(
                        session,
                        evaluation_case_result_id=case_result.id,
                        name=guardrail_name,
                        passed=verdict is GuardrailStatus.PASSED,
                        status=verdict,
                    )

            await evaluations.update_status(session, run, RunStatus.COMPLETED)

    return baseline.id, current.id


@pytest.fixture
def service(db_manager: DatabaseManager) -> RegressionService:
    return RegressionService(db_manager.session_factory)


# -- persistence -----------------------------------------------------------------


async def test_compare_persists_artifact_with_full_report(clean_db, service):
    baseline_id, current_id = await seed_scenario(clean_db)

    report = await service.compare_runs(baseline_id, current_id)

    assert report.comparison_id is not None
    assert report.created_at is not None
    assert report.result is ComparisonResult.REGRESSION_DETECTED
    assert report.regression_detected is True
    assert report.comparison_version == "1"
    assert report.matched_cases == 10
    assert [f.name for f in report.newly_failed_cases] == ["ok5"]
    assert [f.name for f in report.fixed_cases] == ["to-fix"]
    assert report.baseline.pass_rate == pytest.approx(77.777778)
    assert report.current.pass_rate == pytest.approx(75.0)
    assert {v.metric for v in report.threshold_violations} == {
        "pass_rate",
        "error_rate",
        "guardrail_failure_rate",
        "latency",
    }

    async with clean_db.session() as session:
        row = await RegressionRepository().get(session, report.comparison_id)
        assert row is not None
        assert row.baseline_run_id == baseline_id
        assert row.current_run_id == current_id
        assert row.regression_detected is True
        assert row.thresholds["max_pass_rate_drop_pp"] == 2.0
        assert row.summary["baseline_run_id"] == str(baseline_id)
        assert row.summary["newly_failed_cases"][0]["name"] == "ok5"
        assert row.summary["guardrail_comparisons"][0]["name"] == "pii"


async def test_threshold_snapshot_is_persisted(clean_db, service):
    baseline_id, current_id = await seed_scenario(clean_db)
    thresholds = RegressionThresholds(
        max_pass_rate_drop_pp=50.0,
        max_error_rate_increase_pp=50.0,
        max_guardrail_failure_rate_increase_pp=50.0,
        max_latency_increase_percent=None,
    )
    report = await service.compare_runs(baseline_id, current_id, thresholds)
    assert report.result is ComparisonResult.NO_REGRESSION  # thresholds forbid any violation

    reloaded = await service.get_report(report.comparison_id)
    assert reloaded.threshold_violations == []
    async with clean_db.session() as session:
        row = await RegressionRepository().get(session, report.comparison_id)
        assert row.thresholds["max_pass_rate_drop_pp"] == 50.0
        assert row.thresholds["max_latency_increase_percent"] is None
        assert row.policy_fingerprint == policy_fingerprint(thresholds)


async def test_persisted_summary_is_sanitized(clean_db, service):
    baseline_id, current_id = await seed_scenario(clean_db, with_secrets=True)
    report = await service.compare_runs(baseline_id, current_id)

    async with clean_db.session() as session:
        row = await RegressionRepository().get(session, report.comparison_id)
        summary_text = str(row.summary)
    assert "baseline-secret-value" not in summary_text
    assert "current-secret-value" not in summary_text
    assert "api_key" not in summary_text
    assert "auth_token" not in summary_text
    # non-secret config change preserved: temperature 0.2 → 0.7
    changes = {c.path: c for c in report.context.configuration_changes}
    assert changes["temperature"].baseline == 0.2
    assert changes["temperature"].current == 0.7


async def test_get_report_round_trip(clean_db, service):
    baseline_id, current_id = await seed_scenario(clean_db)
    created = await service.compare_runs(baseline_id, current_id)
    reloaded = await service.get_report(created.comparison_id)
    assert reloaded.model_dump(exclude={"created_at"}) == created.model_dump(
        exclude={"created_at"}
    )
    assert reloaded.created_at == created.created_at


# -- idempotency & reproducibility -------------------------------------------------


async def test_repeated_comparison_is_idempotent(clean_db, service):
    baseline_id, current_id = await seed_scenario(clean_db)
    first = await service.compare_runs(baseline_id, current_id)
    second = await service.compare_runs(baseline_id, current_id)

    assert second.comparison_id == first.comparison_id
    assert second.created_at == first.created_at
    assert second.model_dump(exclude={"created_at"}) == first.model_dump(exclude={"created_at"})

    async with clean_db.session() as session:
        rows = await RegressionRepository().list_for_run(session, baseline_id)
    assert len(rows) == 1


async def test_different_thresholds_create_distinct_artifacts(clean_db, service):
    baseline_id, current_id = await seed_scenario(clean_db)
    strict = await service.compare_runs(baseline_id, current_id, RegressionThresholds())
    lax = await service.compare_runs(
        baseline_id,
        current_id,
        RegressionThresholds(
            max_pass_rate_drop_pp=99.0,
            max_error_rate_increase_pp=99.0,
            max_guardrail_failure_rate_increase_pp=99.0,
            max_latency_increase_percent=None,
        ),
    )
    assert lax.comparison_id != strict.comparison_id
    assert lax.regression_detected is False
    async with clean_db.session() as session:
        rows = await RegressionRepository().list_for_run(session, baseline_id)
    assert len(rows) == 2
    assert {r.result for r in rows} == {
        ComparisonResult.REGRESSION_DETECTED,
        ComparisonResult.NO_REGRESSION,
    }


async def test_reproducibility_and_evaluation_immutability(clean_db, service):
    baseline_id, current_id = await seed_scenario(clean_db)

    async def evaluation_snapshot():
        async with clean_db.engine.connect() as conn:
            runs = (
                await conn.execute(text(
                    "SELECT id, status, agent_model, configuration_snapshot, started_at, "
                    "completed_at FROM evaluation_runs ORDER BY id"
                ))
            ).all()
            cases = (
                await conn.execute(text(
                    "SELECT id, evaluation_run_id, test_case_id, status, latency_ms, actual_output "
                    "FROM evaluation_case_results ORDER BY id"
                ))
            ).all()
            guardrails = (
                await conn.execute(text(
                    "SELECT id, evaluation_case_result_id, name, status, passed "
                    "FROM guardrail_results ORDER BY id"
                ))
            ).all()
        return runs, cases, guardrails

    before = await evaluation_snapshot()
    first = await service.compare_runs(baseline_id, current_id)
    second = await service.compare_runs(baseline_id, current_id)
    after = await evaluation_snapshot()

    assert before == after  # historical evaluation data untouched
    assert first.model_dump(exclude={"created_at"}) == second.model_dump(exclude={"created_at"})


# -- database guarantees -----------------------------------------------------------


async def test_referenced_run_deletion_is_restricted(clean_db, service):
    baseline_id, current_id = await seed_scenario(clean_db)
    await service.compare_runs(baseline_id, current_id)

    async with clean_db.engine.begin() as conn:
        with pytest.raises(IntegrityError):
            await conn.execute(text("DELETE FROM evaluation_runs WHERE id = :id"), {"id": baseline_id})

    # after removing the artifact, the runs are deletable again
    async with clean_db.engine.begin() as conn:
        await conn.execute(text("DELETE FROM regression_comparisons"))
        await conn.execute(text("DELETE FROM evaluation_runs WHERE id IN (:b, :c)"),
                           {"b": baseline_id, "c": current_id})


async def test_database_check_constraint_rejects_self_comparison(clean_db):
    baseline_id, _ = await seed_scenario(clean_db)
    async with clean_db.engine.begin() as conn:
        with pytest.raises(IntegrityError):
            await conn.execute(
                text(
                    "INSERT INTO regression_comparisons "
                    "(id, baseline_run_id, current_run_id, result, regression_detected, "
                    " comparison_version, policy_fingerprint, thresholds, summary) "
                    "VALUES (gen_random_uuid(), :r, :r, 'no_regression', false, '1', 'x', '{}', '{}')"
                ),
                {"r": baseline_id},
            )


# -- compatibility rejection ---------------------------------------------------------


async def make_minimal_run(clean_db, *, app_name: str, dataset_name: str) -> uuid.UUID:
    apps, datasets, evaluations = (
        ApplicationRepository(),
        DatasetRepository(),
        EvaluationRepository(),
    )
    async with clean_db.session() as session:
        app = await apps.create(session, name=app_name)
        dataset = await datasets.create(session, name=dataset_name)
        version = await datasets.create_version(session, dataset_id=dataset.id, version=1)
        run = await evaluations.create_run(
            session,
            application_id=app.id,
            dataset_version_id=version.id,
            agent_model="agent-a",
        )
    return run.id


async def test_rejects_same_run_as_baseline_and_current(clean_db, service):
    run_id, _ = await seed_scenario(clean_db)
    with pytest.raises(RegressionValidationError, match="itself"):
        await service.compare_runs(run_id, run_id)


@pytest.mark.parametrize(
    ("baseline_status", "current_status"),
    [
        (RunStatus.PENDING, RunStatus.COMPLETED),
        (RunStatus.RUNNING, RunStatus.COMPLETED),
        (RunStatus.FAILED, RunStatus.COMPLETED),
        (RunStatus.CANCELLED, RunStatus.COMPLETED),
        (RunStatus.COMPLETED, RunStatus.PENDING),
        (RunStatus.COMPLETED, RunStatus.RUNNING),
        (RunStatus.COMPLETED, RunStatus.FAILED),
        (RunStatus.COMPLETED, RunStatus.CANCELLED),
    ],
)
async def test_rejects_non_completed_runs(clean_db, service, baseline_status, current_status):
    apps, datasets, evaluations = (
        ApplicationRepository(),
        DatasetRepository(),
        EvaluationRepository(),
    )
    async with clean_db.session() as session:
        app = await apps.create(session, name=f"app-{uuid.uuid4().hex[:8]}")
        dataset = await datasets.create(session, name=f"ds-{uuid.uuid4().hex[:8]}")
        version = await datasets.create_version(session, dataset_id=dataset.id, version=1)
        baseline = await evaluations.create_run(
            session, application_id=app.id, dataset_version_id=version.id, agent_model="m"
        )
        current = await evaluations.create_run(
            session, application_id=app.id, dataset_version_id=version.id, agent_model="m"
        )
        await evaluations.update_status(session, baseline, baseline_status)
        await evaluations.update_status(session, current, current_status)

    with pytest.raises(RegressionValidationError, match="completed"):
        await service.compare_runs(baseline.id, current.id)


async def test_rejects_cross_application_comparisons(clean_db, service):
    run_a = await make_minimal_run(
        clean_db, app_name=f"a-{uuid.uuid4().hex[:8]}", dataset_name=f"ad-{uuid.uuid4().hex[:8]}"
    )
    run_b = await make_minimal_run(
        clean_db, app_name=f"b-{uuid.uuid4().hex[:8]}", dataset_name=f"bd-{uuid.uuid4().hex[:8]}"
    )
    async with clean_db.session() as session:
        run_a_row = await EvaluationRepository().get_run(session, run_a)
        run_b_row = await EvaluationRepository().get_run(session, run_b)
        await EvaluationRepository().update_status(session, run_a_row, RunStatus.COMPLETED)
        await EvaluationRepository().update_status(session, run_b_row, RunStatus.COMPLETED)
    with pytest.raises(RegressionValidationError, match="different applications"):
        await service.compare_runs(run_a, run_b)


async def test_rejects_different_datasets(clean_db, service):
    # same application, two different datasets
    apps, datasets, evaluations = (
        ApplicationRepository(),
        DatasetRepository(),
        EvaluationRepository(),
    )
    async with clean_db.session() as session:
        app = await apps.create(session, name=f"app-{uuid.uuid4().hex[:8]}")
        ds1 = await datasets.create(session, name=f"ds1-{uuid.uuid4().hex[:8]}")
        ds2 = await datasets.create(session, name=f"ds2-{uuid.uuid4().hex[:8]}")
        v1 = await datasets.create_version(session, dataset_id=ds1.id, version=1)
        v2 = await datasets.create_version(session, dataset_id=ds2.id, version=1)
        baseline = await evaluations.create_run(
            session, application_id=app.id, dataset_version_id=v1.id, agent_model="m"
        )
        current = await evaluations.create_run(
            session, application_id=app.id, dataset_version_id=v2.id, agent_model="m"
        )
        await evaluations.update_status(session, baseline, RunStatus.COMPLETED)
        await evaluations.update_status(session, current, RunStatus.COMPLETED)

    with pytest.raises(RegressionValidationError, match="different datasets"):
        await service.compare_runs(baseline.id, current.id)


async def test_missing_run_raises_typed_error(clean_db, service):
    with pytest.raises(NotFoundError):
        await service.compare_runs(uuid.uuid4(), uuid.uuid4())


# -- dataset versioning ---------------------------------------------------------------


async def test_cross_version_comparison_matches_cases_by_name(clean_db, service):
    """v1 → baseline, v2 → current: common cases compare by name; v2's new
    case is reported separately; v1's dropped case counts as removed."""
    apps, datasets, evaluations = (
        ApplicationRepository(),
        DatasetRepository(),
        EvaluationRepository(),
    )
    async with clean_db.session() as session:
        app = await apps.create(session, name=f"app-{uuid.uuid4().hex[:8]}")
        dataset = await datasets.create(session, name=f"ds-{uuid.uuid4().hex[:8]}")
        v1 = await datasets.create_version(session, dataset_id=dataset.id, version=1)
        v2 = await datasets.create_version(session, dataset_id=dataset.id, version=2)
        v1_cases: dict[str, uuid.UUID] = {}
        for name in ["c0", "c1", "c2", "c3", "c4"]:
            case = await datasets.add_test_case(
                session, dataset_version_id=v1.id, name=name, input={"prompt": name}
            )
            v1_cases[name] = case.id
        v2_cases: dict[str, uuid.UUID] = {}
        for name in ["c0", "c1", "c2", "c3", "c-new"]:
            case = await datasets.add_test_case(
                session, dataset_version_id=v2.id, name=name, input={"prompt": name}
            )
            v2_cases[name] = case.id

        baseline = await evaluations.create_run(
            session, application_id=app.id, dataset_version_id=v1.id, agent_model="m"
        )
        current = await evaluations.create_run(
            session, application_id=app.id, dataset_version_id=v2.id, agent_model="m"
        )
        outcomes = {
            baseline: {"c0": CaseStatus.PASSED, "c1": CaseStatus.PASSED,
                       "c2": CaseStatus.PASSED, "c3": CaseStatus.PASSED,
                       "c4": CaseStatus.PASSED},
            current: {"c0": CaseStatus.PASSED, "c1": CaseStatus.PASSED,
                      "c2": CaseStatus.FAILED, "c3": CaseStatus.PASSED,
                      "c-new": CaseStatus.FAILED},
        }
        for run, run_outcomes in outcomes.items():
            cases = v1_cases if run is baseline else v2_cases
            for name, status in run_outcomes.items():
                await evaluations.add_case_result(
                    session,
                    evaluation_run_id=run.id,
                    test_case_id=cases[name],
                    input={"prompt": name},
                    status=status,
                )
            await evaluations.update_status(session, run, RunStatus.COMPLETED)

    report = await service.compare_runs(baseline.id, current.id)

    assert report.matched_cases == 4  # c0..c3 matched by name
    assert report.new_cases == ["c-new"]
    assert report.removed_cases == ["c4"]  # baseline result with no current counterpart
    assert [f.name for f in report.newly_failed_cases] == ["c2"]
    assert report.baseline.pass_rate == 100.0
    assert report.current.pass_rate == 75.0
    assert report.result is ComparisonResult.REGRESSION_DETECTED


# -- degenerate data ------------------------------------------------------------------


async def test_not_comparable_when_current_run_has_no_results(clean_db, service):
    apps, datasets, evaluations = (
        ApplicationRepository(),
        DatasetRepository(),
        EvaluationRepository(),
    )
    async with clean_db.session() as session:
        app = await apps.create(session, name=f"app-{uuid.uuid4().hex[:8]}")
        dataset = await datasets.create(session, name=f"ds-{uuid.uuid4().hex[:8]}")
        version = await datasets.create_version(session, dataset_id=dataset.id, version=1)
        case = await datasets.add_test_case(
            session, dataset_version_id=version.id, name="c0", input={"prompt": "x"}
        )
        baseline = await evaluations.create_run(
            session, application_id=app.id, dataset_version_id=version.id, agent_model="m"
        )
        current = await evaluations.create_run(
            session, application_id=app.id, dataset_version_id=version.id, agent_model="m"
        )
        await evaluations.add_case_result(
            session,
            evaluation_run_id=baseline.id,
            test_case_id=case.id,
            input={"prompt": "x"},
            status=CaseStatus.PASSED,
        )
        await evaluations.update_status(session, baseline, RunStatus.COMPLETED)
        await evaluations.update_status(session, current, RunStatus.COMPLETED)

    report = await service.compare_runs(baseline.id, current.id)
    assert report.result is ComparisonResult.NOT_COMPARABLE
    assert report.regression_detected is False
    assert report.not_comparable_reason is not None
    assert report.matched_cases == 0
    assert report.missing_case_results["current"] == ["c0"]


async def test_list_for_run_returns_both_sides(clean_db, service):
    baseline_id, current_id = await seed_scenario(clean_db)
    await service.compare_runs(baseline_id, current_id)
    async with clean_db.session() as session:
        as_baseline = await RegressionRepository().list_for_run(session, baseline_id)
        as_current = await RegressionRepository().list_for_run(session, current_id)
    assert len(as_baseline) == 1
    assert len(as_current) == 1
    assert as_baseline[0].id == as_current[0].id
