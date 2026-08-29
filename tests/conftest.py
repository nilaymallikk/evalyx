"""Shared pytest fixtures and skips for the Evalyx test suite."""

import os

import pytest

# Provide a safe test default so Settings can always be constructed in the
# test environment, regardless of the developer's local `.env`.
os.environ.setdefault("EVALYX_SECRET_KEY", "test-only-secret-key")


def pytest_collection_modifyitems(config, items):
    """Skip integration tests unless explicitly enabled."""
    if os.getenv("EVALYX_RUN_INTEGRATION_TESTS") == "1":
        return
    skip_integration = pytest.mark.skip(
        reason="integration test; set EVALYX_RUN_INTEGRATION_TESTS=1 to run"
    )
    for item in items:
        if "integration" in item.keywords:
            item.add_marker(skip_integration)
