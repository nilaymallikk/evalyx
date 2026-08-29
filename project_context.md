# Evalyx — Project Context

## 1. Project Identity

**Project name:** Evalyx

**Repository description:**

> An AI evaluation and reliability platform for testing, observing, debugging, and regression-testing LLM applications and agents.

Evalyx is a portfolio-grade, production-style AI engineering project designed to demonstrate that its developer can build and maintain an existing AI application/service rather than merely call an LLM API.

The project should feel like an internal platform that an AI engineering team could use to evaluate an LLM-powered application before and after changes.

---

## 2. Why This Project Exists

The target job profile values fresh AI engineers who can maintain, evolve, debug, and improve an existing Azure-hosted AI application.

Evalyx should therefore demonstrate:

- Maintainable backend architecture
- API development
- Database design
- Background/asynchronous processing
- LLM integration
- Guardrails
- Evaluation methodology
- Regression testing
- Observability
- Error handling
- Testing
- Documentation
- Dockerized development
- Azure-ready architecture

The goal is not to build a toy chatbot. The goal is to build the tooling around an AI application that helps a team answer:

> "Did our AI application get better or worse after this change, and why?"

---

## 3. Core Product Idea

A developer registers an LLM application/agent, creates a versioned evaluation dataset, selects an evaluation policy, and starts an evaluation run.

Evalyx then:

1. Executes test cases against the target LLM application/model.
2. Captures inputs, outputs, latency, errors, and metadata.
3. Runs deterministic guardrails.
4. Optionally uses an LLM judge for semantic evaluation.
5. Calculates scores.
6. Stores detailed results.
7. Compares the run with a baseline.
8. Detects regressions.
9. Makes failures easy to inspect and debug.

The central domain object is an **Evaluation Run**.

---

## 4. Initial Demonstration Scenario

Use a fictional customer-support AI assistant as the primary demo.

The demo should contain normal, adversarial, and edge-case test cases.

Important scenarios:

- Prompt injection
- Sensitive/PII leakage
- Unsafe output
- Hallucination / unsupported claims
- Instruction-following failures
- Normal successful customer-support questions
- Error/timeout handling

Example regression story:

Baseline:

- 100 test cases
- 94% pass rate

New version:

- 100 test cases
- 86% pass rate

Evalyx should identify:

- Overall regression
- Newly failed cases
- Which guardrails failed
- Score changes
- Relevant model/configuration/version metadata

This regression demonstration is a key recruiter-facing feature.

---

## 5. Important Scope Decision

Evalyx is **not just a guardrail library**.

It is an:

> AI evaluation and reliability platform.

Guardrails are one component of the evaluation system.

The platform should eventually support:

- Test datasets
- Dataset versions
- Evaluation runs
- Guardrail policies
- Model/application configurations
- Deterministic checks
- LLM-as-a-judge checks
- Scoring
- Regression detection
- Failure analysis
- Observability
- Audit/history

---

## 6. Free Model Constraint

The project must initially use **free models through OpenRouter only**.

Do not introduce a paid-model dependency.

Current intended models:

### Agent / system under test

```text
nvidia/nemotron-3-ultra-550b-a55b:free
```

### Judge / evaluator

```text
minimax/minimax-m3:free
```

### Alternative experimental agent

```text
poolside/laguna-s-2.1:free
```

The architecture must NOT hard-code these models throughout the codebase.

Use a provider/model abstraction and configuration so models can be selected per evaluation run.

The application should gracefully handle:

- Model unavailable
- Rate limits
- Provider errors
- Timeouts
- Malformed responses
- Temporary upstream failures

The project may later support Azure OpenAI, OpenAI, or other providers, but these are not required for the initial free-model implementation.

---

## 7. Recommended Technology Stack

### Language

- Python 3.12+ (use the repository's configured Python version if already established)

### API/backend

- FastAPI
- Pydantic v2
- SQLAlchemy 2.x
- asyncpg
- Alembic

### Database

- PostgreSQL

### Background processing

- Redis
- Celery

### LLM access

- OpenRouter
- Provider abstraction/interface
- Optional local Ollama support

### Testing

- pytest
- pytest-asyncio where needed

### Quality/tooling

- Ruff
- Type checking where practical
- uv for dependency/environment management

### Containers

- Docker
- Docker Compose

### Observability

- Structured Python logging
- OpenTelemetry where appropriate
- Metrics support where useful
- Do not attempt to recreate Datadog/Grafana/etc. unnecessarily

---

## 8. Current Repository State

The repository currently looks approximately like:

```text
evalyx/
├── .claude-flow/
├── .venv/
├── docs/
├── evals/
├── examples/
├── src/
│   └── evalyx/
│       └── __init__.py
├── tests/
├── .env
├── .env.example
├── .github/
├── .gitignore
├── .python-version
├── docker-compose.yml
├── main.py
├── pyproject.toml
├── README.md
├── test_openrouter.py
└── uv.lock
```

This is the **existing repository**. Do not blindly delete or rewrite it.

Before making structural changes, inspect:

- `pyproject.toml`
- `docker-compose.yml`
- `main.py`
- `test_openrouter.py`
- `.env.example`
- `.gitignore`
- `README.md`
- existing `docs/`, `evals/`, `examples/`, and `tests/`

Preserve useful work and improve it incrementally.

The `.venv/` directory is local environment state and should not be committed.

---

## 9. Existing Environment Configuration

The intended local configuration is:

```env
APP_ENV=development
LOG_LEVEL=INFO

DATABASE_URL=postgresql+asyncpg://evalyx:evalyx@localhost:5432/evalyx

REDIS_URL=redis://localhost:6379/0

EVALYX_SECRET_KEY=<generated-secret>

OPENROUTER_API_KEY=<local-secret>

EVALYX_AGENT_MODEL=nvidia/nemotron-3-ultra-550b-a55b:free
EVALYX_JUDGE_MODEL=minimax/minimax-m3:free

OLLAMA_BASE_URL=http://localhost:11434
```

Important:

- `.env` contains secrets and must never be committed.
- `.env.example` contains placeholders and should be committed.
- `EVALYX_SECRET_KEY=change-me` is not acceptable for a real running configuration.
- Generate a secure random secret locally.
- Never print or expose API keys in source code, logs, tests, screenshots, commits, or documentation.
- The previously visible OpenRouter key must be considered compromised and rotated.

---

## 10. PostgreSQL and Redis

The existing URLs are local development connection strings, not externally supplied personal credentials.

The recommended development architecture is to run PostgreSQL and Redis through Docker Compose.

Expected local services:

```text
PostgreSQL
host: localhost
port: 5432
database: evalyx
user: evalyx
password: evalyx
```

and:

```text
Redis
host: localhost
port: 6379
database: 0
```

The actual `docker-compose.yml` must be inspected before changing it.

Do not install or configure a cloud database merely for the initial project.

---

## 11. High-Level Architecture

```text
                   LLM Application / Agent
                             |
                             v
                    +------------------+
                    |      Evalyx      |
                    | Evaluation API   |
                    +--------+---------+
                             |
             +---------------+---------------+
             |               |               |
             v               v               v
         Dataset         Guardrails       LLM Judge
             |               |               |
             +---------------+---------------+
                             |
                             v
                    Evaluation Engine
                             |
             +---------------+---------------+
             |               |               |
             v               v               v
          Quality         Safety        Reliability
           Scores         Checks          Metrics
             |               |               |
             +---------------+---------------+
                             |
                             v
                    Regression Engine
                             |
                    +--------+--------+
                    |                 |
                    v                 v
                 Baseline        Current Run
                    |                 |
                    +--------+--------+
                             |
                             v
                     Regression Report
```

---

## 12. Evaluation Pipeline

A typical evaluation run should follow:

```text
Create Evaluation Run
        |
        v
Load Dataset Version
        |
        v
Resolve Application / Model Configuration
        |
        v
Execute Test Cases
        |
        v
Capture Output + Latency + Errors
        |
        +----------------------+
        |                      |
        v                      v
Deterministic Checks       LLM Judge
        |                      |
        +----------+-----------+
                   |
                   v
             Score Results
                   |
                   v
          Store Case Results
                   |
                   v
           Aggregate Metrics
                   |
                   v
          Compare Baseline
                   |
                   v
        Detect Regressions
                   |
                   v
             Final Report
```

---

## 13. Guardrails

Initial guardrails should be limited to a useful, demonstrable set.

### Prompt Injection

Detect whether the application can be manipulated by instructions such as attempts to ignore system instructions, reveal hidden prompts, or bypass policy.

### PII / Sensitive Data Leakage

Detect sensitive information such as:

- Email addresses
- Phone numbers
- Credit-card-like patterns
- API keys/secrets
- Other configured sensitive patterns

This should initially rely on deterministic checks where possible.

### Unsafe Content

Evaluate whether the application generates content violating the configured safety policy.

Use semantic evaluation where deterministic rules are insufficient.

### Hallucination / Unsupported Claims

Evaluate whether responses make claims unsupported by supplied context or expected facts.

### Instruction Following

Evaluate whether the response satisfies a requested output format, required fields, constraints, or expected behavior.

---

## 14. Hybrid Evaluation Strategy

Do not make every guardrail an LLM call.

Use:

### Deterministic evaluation

For:

- Regex/pattern checks
- PII detection
- Secret detection
- Schema validation
- Required fields
- Maximum length
- Simple policy rules

### Semantic / LLM evaluation

For:

- Hallucination
- Groundedness
- Instruction following
- Semantic safety
- Prompt-injection effectiveness where necessary
- Overall qualitative response evaluation

The platform should make it clear which evaluator produced each result.

---

## 15. LLM-as-a-Judge

The judge should receive enough context to evaluate the response responsibly.

Potential inputs:

```text
Test input
Expected behavior
Optional context
Actual model output
Evaluation criteria
```

The judge should produce structured output such as:

```json
{
  "score": 0.0,
  "passed": true,
  "reason": "..."
}
```

The implementation must validate judge output and handle malformed judge responses safely.

Do not blindly trust arbitrary judge text.

---

## 16. Core Data Model Concepts

The exact schema should be designed during the database phase, but the domain should include concepts similar to:

```text
Application
ApplicationVersion / Configuration
Dataset
DatasetVersion
TestCase
GuardrailPolicy
EvaluationRun
EvaluationCaseResult
EvaluationMetric
RegressionComparison
ModelConfiguration
Audit / execution metadata
```

Every evaluation result should retain enough information to reproduce or debug what happened.

Useful metadata includes:

```text
evaluation_id
test_case_id
dataset_version
application/version
model
judge model
timestamp
latency
status
error
input
output
guardrail results
scores
```

Avoid storing secrets unnecessarily.

---

## 17. Regression Testing

Regression detection is a core feature.

Compare a current evaluation run against a selected baseline.

Possible metrics:

- Pass rate
- Safety score
- Quality score
- Groundedness
- Instruction-following score
- Guardrail failure rate
- Latency
- Error rate

Example:

```text
Baseline pass rate: 94%
Current pass rate:  86%
Change:             -8 percentage points

Regression: DETECTED
```

The system should identify newly failing or significantly degraded cases.

Regression thresholds must be explicit/configurable rather than hidden magic numbers.

---

## 18. Observability

Capture structured execution information.

At minimum:

```text
request_id
evaluation_id
test_case_id
model
latency
status
error
timestamp
token usage when available
guardrail outcomes
judge score
```

Use structured logs.

Add tracing/metrics only where they materially improve the system.

Do not build unnecessary infrastructure solely to make the architecture look complicated.

---

## 19. Async Processing

Evaluation runs can contain many test cases and should not block an HTTP request.

Preferred flow:

```text
POST /evaluation-runs
        |
        v
Create run record
        |
        v
Queue background task
        |
        v
Redis
        |
        v
Celery worker
        |
        v
Execute evaluation
        |
        v
Persist results
```

The API should provide a way to retrieve run status/results.

Handle:

- Worker failures
- Individual test failures
- Retryable upstream failures
- Timeouts
- Partial failures

A single failed test case should not necessarily destroy the entire run.

---

## 20. Provider Abstraction

Use a clean interface around LLM calls.

Conceptually:

```text
LLMProvider
   |
   +-- OpenRouterProvider
   |
   +-- OllamaProvider
   |
   +-- Future Azure/OpenAI provider
```

The evaluation engine should depend on the abstraction, not directly on OpenRouter-specific implementation details.

This is important for Azure readiness.

---

## 21. Azure Readiness

Azure is a target deployment/integration direction, not an initial requirement.

Do not force Azure services into local development.

The architecture should make it possible to later replace or add:

```text
Azure OpenAI
Azure Container Apps / App Service
Azure Database for PostgreSQL
Azure Cache for Redis
Application Insights / Azure Monitor
```

without rewriting the evaluation domain.

The recruiter story should be:

> Evalyx was designed as a provider-agnostic evaluation platform that can locally test free OpenRouter models and can later evaluate Azure OpenAI-backed applications.

---

## 22. Target Source Structure

The final structure may evolve, but a reasonable target is:

```text
src/evalyx/
├── api/
│   ├── routes/
│   └── dependencies.py
├── core/
│   ├── config.py
│   ├── logging.py
│   └── security.py
├── db/
│   ├── models/
│   ├── repositories/
│   └── session.py
├── evaluation/
│   ├── engine.py
│   ├── runner.py
│   ├── scoring.py
│   └── regression.py
├── guardrails/
│   ├── base.py
│   ├── injection.py
│   ├── pii.py
│   ├── safety.py
│   ├── hallucination.py
│   └── instruction.py
├── llm/
│   ├── base.py
│   ├── openrouter.py
│   ├── ollama.py
│   └── judge.py
├── datasets/
│   ├── loader.py
│   └── versioning.py
├── workers/
│   └── tasks.py
├── observability/
│   ├── tracing.py
│   └── metrics.py
└── __init__.py
```

Do not create every directory merely because this document shows it. Add structure when a phase requires it.

---

## 23. Development Principles

The AI coding assistant must follow these principles:

1. Inspect before modifying.
2. Preserve useful existing work.
3. Do not rewrite the entire repository without justification.
4. Work phase-by-phase.
5. Keep changes small and reviewable.
6. Prefer simple, maintainable implementations.
7. Avoid premature abstraction.
8. Do not add dependencies without a reason.
9. Keep provider-specific logic isolated.
10. Never hard-code secrets.
11. Add tests with meaningful functionality.
12. Validate changes locally.
13. Update documentation as architecture changes.
14. Explain important design decisions.
15. Do not claim a feature works unless it has actually been tested.
16. Prefer explicit configuration over hidden behavior.
17. Design for failure, not just the happy path.
18. Preserve reproducibility of evaluation runs.
19. Treat evaluation data and results as first-class domain objects.
20. Build something a recruiter can clone, run, understand, and discuss.

---

## 24. Development Phases

Implementation should proceed sequentially.

### Phase 1 — Repository Audit & Architecture

Inspect the existing repository and establish the final architecture without unnecessary rewriting.

Deliverables:

- Repository assessment
- Architecture decision
- Cleanup only where justified
- Development conventions
- Updated project documentation

### Phase 2 — Configuration & Infrastructure

Make local configuration reliable.

Deliverables:

- Settings/config module
- Environment validation
- Docker Compose
- PostgreSQL
- Redis
- Health checks
- Secure secret handling

### Phase 3 — Database & Domain Model

Build the persistent domain model.

Deliverables:

- SQLAlchemy models
- Alembic migrations
- Database session
- Repositories
- Initial schema
- Database tests

### Phase 4 — LLM Provider Layer

Implement model/provider abstraction.

Deliverables:

- Provider interface
- OpenRouter implementation
- Optional Ollama implementation
- Agent model configuration
- Judge model configuration
- Timeout/error handling
- Provider tests

### Phase 5 — Evaluation Engine

Build the core evaluation pipeline.

Deliverables:

- Dataset loading
- Test case execution
- Result storage
- Scoring
- Run lifecycle
- Partial failure handling

### Phase 6 — Guardrails

Implement the initial hybrid guardrail system.

Deliverables:

- Guardrail interface
- PII detection
- Prompt-injection evaluation
- Safety evaluation
- Hallucination/groundedness evaluation
- Instruction-following evaluation
- Structured guardrail results

### Phase 7 — Async Evaluation Workers

Move long-running evaluation workloads into background processing.

Deliverables:

- Celery integration
- Redis queue
- Worker tasks
- Job status
- Retry/timeout behavior
- Failure handling

### Phase 8 — Regression Testing

Build baseline/current-run comparison.

Deliverables:

- Baselines
- Metric comparison
- Thresholds
- Newly failed case detection
- Regression report

### Phase 9 — API

Expose the platform through FastAPI.

Deliverables:

- Applications
- Datasets
- Test cases
- Policies
- Evaluation runs
- Results
- Regression endpoints
- Health endpoints

### Phase 10 — Observability

Add useful production-style visibility.

Deliverables:

- Structured logging
- Request IDs
- Evaluation IDs
- Metrics
- Optional tracing
- Error context

### Phase 11 — Demo Application

Create a local customer-support AI application/agent that can be evaluated.

Deliverables:

- Demo agent
- Normal cases
- Adversarial cases
- Deliberately vulnerable behavior where appropriate for demonstration
- Evaluation dataset

### Phase 12 — Tests & Reliability

Strengthen the project.

Deliverables:

- Unit tests
- Integration tests
- Evaluation tests
- Failure-path tests
- API tests
- Regression tests
- Quality checks

### Phase 13 — Documentation & Recruiter Demo

Polish the project.

Deliverables:

- README
- Architecture documentation
- Setup guide
- Example evaluation
- Regression example
- API usage examples
- Design decisions
- Limitations
- Future Azure deployment path

---

## 25. Definition of Done

A phase is not complete merely because code was written.

For every phase, the coding assistant must:

1. Inspect relevant existing files.
2. Explain the intended change briefly.
3. Implement the change.
4. Run relevant tests/checks.
5. Fix failures.
6. Update documentation if needed.
7. Summarize:
   - What changed
   - Why
   - Files changed
   - Tests executed
   - Known limitations
   - What the next phase needs

Do not automatically continue into the next phase unless explicitly instructed.

---

## 26. AI Coding Assistant Rules

The coding assistant receiving this file should treat it as project context, not as permission to implement everything immediately.

At the start of each phase:

- Read this file.
- Inspect the current repository.
- Compare actual state against this context.
- Identify discrepancies.
- Do not assume the documented target structure already exists.
- Ask only when a decision cannot reasonably be made from the repository/context.
- Prefer implementing the smallest correct change.

Never:

- expose secrets
- commit `.env`
- hard-code API keys
- invent test results
- silently replace working components
- add Azure dependencies without a current phase requirement
- require paid models
- make the entire platform depend on one model
- treat an LLM judge as infallible
- skip tests because an AI-generated implementation "looks correct"

---

## 27. Current Priority

The project is currently in the **pre-build/setup stage**.

Before full implementation:

1. Rotate the previously exposed OpenRouter key.
2. Generate a real `EVALYX_SECRET_KEY`.
3. Inspect and verify `docker-compose.yml`.
4. Ensure PostgreSQL and Redis can run locally.
5. Verify the Python/uv environment.
6. Verify OpenRouter access with the selected free model.
7. Verify database connectivity.
8. Verify Redis connectivity.
9. Only then begin the phased implementation.

Do not start by building the dashboard or a large frontend.

The first priority is a reliable backend evaluation pipeline.

---

## 28. Recruiter-Facing Story

The final project should support this concise explanation:

> "Evalyx is an AI evaluation and reliability platform I built to test LLM applications and agents before and after changes. It runs versioned test datasets asynchronously, applies deterministic and LLM-based guardrails, records latency/errors/results, and compares runs against baselines to detect regressions. I designed the provider layer so it can run locally with free OpenRouter models while remaining ready for Azure OpenAI."

The strongest demo should show:

```text
Baseline Evaluation
       |
       v
94% Pass Rate
       |
       v
Change model/prompt/application
       |
       v
Run Evalyx Again
       |
       v
86% Pass Rate
       |
       v
REGRESSION DETECTED
       |
       +--> 7 newly failed cases
       +--> PII guardrail failure
       +--> prompt-injection failure
       +--> latency increased
       +--> detailed case-level evidence
```

That is the core product story.

---

## 29. Final Instruction

Build Evalyx as a serious but finishable portfolio project.

Optimize for:

**correctness > maintainability > explainability > unnecessary complexity.**

The project should demonstrate that the developer understands how AI systems are evaluated, monitored, debugged, secured, tested, and evolved—not merely how to send a prompt to an LLM.
