import os
import sys

import requests
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("OPENROUTER_API_KEY")
AGENT_MODEL = os.getenv(
    "EVALYX_AGENT_MODEL",
    "nvidia/nemotron-3-ultra-550b-a55b:free",
)
JUDGE_MODEL = os.getenv(
    "EVALYX_JUDGE_MODEL",
    "minimax/minimax-m3:free",
)

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"


def call_model(model: str, prompt: str) -> str:
    if not API_KEY:
        raise RuntimeError("OPENROUTER_API_KEY is missing from .env")

    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": prompt,
            }
        ],
        "temperature": 0.2,
        "max_tokens": 200,
    }

    response = requests.post(
        OPENROUTER_URL,
        headers=headers,
        json=payload,
        timeout=60,
    )

    response.raise_for_status()

    data = response.json()

    try:
        return data["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise RuntimeError(
            f"Unexpected OpenRouter response:\n{data}"
        ) from exc


def test_model(name: str, model: str, prompt: str) -> bool:
    print(f"\n{'=' * 60}")
    print(f"Testing {name}")
    print(f"Model: {model}")
    print("=" * 60)

    try:
        result = call_model(model, prompt)

        print("\nResponse:")
        print(result)

        print(f"\n✅ {name} test passed")
        return True

    except requests.HTTPError as exc:
        print(f"\n❌ {name} HTTP error:")
        print(exc)

        if exc.response is not None:
            print(exc.response.text)

        return False

    except requests.RequestException as exc:
        print(f"\n❌ {name} request failed:")
        print(exc)
        return False

    except Exception as exc:
        print(f"\n❌ {name} test failed:")
        print(exc)
        return False


def main() -> int:
    print("Evalyx — OpenRouter Connectivity Test")

    agent_ok = test_model(
        "Agent Model",
        AGENT_MODEL,
        (
            "You are the AI agent being evaluated by Evalyx. "
            "Answer briefly: What is the purpose of an AI evaluation "
            "platform?"
        ),
    )

    judge_ok = test_model(
        "Judge Model",
        JUDGE_MODEL,
        (
            "You are an evaluation judge. "
            "Evaluate this response:\n\n"
            "Question: What is 2 + 2?\n"
            "Response: 4\n\n"
            "Return a short assessment explaining whether the response "
            "is correct."
        ),
    )

    print(f"\n{'=' * 60}")
    print("SUMMARY")
    print("=" * 60)
    print(f"Agent model: {'✅ PASS' if agent_ok else '❌ FAIL'}")
    print(f"Judge model: {'✅ PASS' if judge_ok else '❌ FAIL'}")

    if agent_ok and judge_ok:
        print("\n🎉 OpenRouter is ready for Evalyx.")
        return 0

    print("\n⚠️ One or more OpenRouter tests failed.")
    return 1


if __name__ == "__main__":
    sys.exit(main())