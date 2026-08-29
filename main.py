"""Development entry point for Evalyx.

Run the local API with:

    uv run python main.py

or, for auto-reload during development:

    uv run uvicorn evalyx.api.app:app --reload
"""

import uvicorn

from evalyx.core.config import get_settings


def main() -> None:
    settings = get_settings()
    uvicorn.run(
        "evalyx.api.app:app",
        host="127.0.0.1",
        port=8000,
        log_level=settings.log_level.lower(),
    )


if __name__ == "__main__":
    main()

