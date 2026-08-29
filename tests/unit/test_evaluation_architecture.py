"""Architectural test: the evaluation engine is provider-independent."""

from pathlib import Path

import evalyx

EVALUATION_PACKAGE_DIR = Path(evalyx.__file__).parent / "evaluation"

FORBIDDEN_MODULES = ("openrouter", "ollama", "factory")


def test_evaluation_package_never_imports_concrete_providers():
    """The engine must depend on LLMProvider only — never on provider modules."""
    offenders: list[str] = []
    for source_file in EVALUATION_PACKAGE_DIR.glob("*.py"):
        for line in source_file.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if stripped.startswith(("import ", "from ")) and any(
                module in stripped for module in FORBIDDEN_MODULES
            ):
                offenders.append(f"{source_file.name}: {stripped}")
    assert offenders == []
