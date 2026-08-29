"""Submit a background evaluation through Celery.

Local development workflow (see README):

    Terminal 1: docker compose up -d          # PostgreSQL + Redis
    Terminal 2: uv run celery -A evalyx.worker.celery_app worker --loglevel=INFO
    Terminal 3: uv run python examples/submit_background_evaluation.py

Creates an application, dataset, version with two test cases, and a pending
`EvaluationRun` in PostgreSQL, then enqueues only the run id. The worker
executes the existing evaluation pipeline and persists results; this script
just waits for the task result and prints the summary. Detailed results live
in PostgreSQL (EvaluationRun / EvaluationCaseResult / GuardrailResult).
"""

import asyncio

from evalyx.core.config import get_settings
from evalyx.db.repositories import (
    ApplicationRepository,
    DatasetRepository,
    EvaluationRepository,
)
from evalyx.db.session import DatabaseManager
from evalyx.worker.tasks import run_evaluation


async def create_pending_run(db: DatabaseManager, settings):
    """Seed a tiny dataset and create the run record (status: pending)."""
    async with db.session() as session:
        apps = ApplicationRepository()
        datasets = DatasetRepository()
        evaluations = EvaluationRepository()

        app = await apps.create(session, name="demo-support-assistant")
        dataset = await datasets.create(session, name="demo-support-dataset")
        version = await datasets.create_version(session, dataset_id=dataset.id, version=1)
        await datasets.add_test_case(
            session,
            dataset_version_id=version.id,
            name="greeting",
            input={"prompt": "Say hello to a customer in one short sentence."},
        )
        await datasets.add_test_case(
            session,
            dataset_version_id=version.id,
            name="refund-policy",
            input={"prompt": "A customer asks how to return an item. Answer briefly."},
        )
        run = await evaluations.create_run(
            session,
            application_id=app.id,
            dataset_version_id=version.id,
            agent_model=settings.evalyx_agent_model,
            judge_model=settings.evalyx_judge_model,
        )
    return run.id


async def main() -> None:
    settings = get_settings()
    db = DatabaseManager(settings)
    try:
        run_id = await create_pending_run(db, settings)
        print(f"Created EvaluationRun {run_id} (pending). Submitting to Celery...")
    finally:
        await db.dispose()

    # Only the run id crosses the queue — never secrets, sessions, or ORM objects.
    result = run_evaluation.apply_async(args=[str(run_id)])
    print(f"Queued task {result.id}. Waiting for the worker...")

    summary = result.get()  # small JSON dict; details stay in PostgreSQL
    print(f"Task finished: {summary}")


if __name__ == "__main__":
    asyncio.run(main())
