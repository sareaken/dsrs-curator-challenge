"""Run the Chapter 4 example questions and record LLM usage."""

from __future__ import annotations

import json
import time
from pathlib import Path

from agents.answer import main as answer_question
from agents.llm import usage


QUESTIONS = [
    "Which manager held the largest Apple position in 2026 Q2?",
    "Which manager added the most Nvidia shares between 2026 Q1 and 2026 Q2?",
    "How many distinct issuers did Renaissance Technologies LLC report in 2026 Q2?",
    "What was the total reported value of Citadel Advisors LLC's holdings in 2026 Q1?",
    "Which managers in the roster filed a 13F-NT instead of a 13F-HR for 2026 Q2?",
    "Did Pershing Square Capital Management report any Microsoft holdings directly in 2026 Q2?",
    "Which manager reported the most call options in 2026 Q2?",
    "What was Third Point LLC's largest position by value in 2026 Q1, and what was it?",
    "Which manager held Tesla in both 2026 Q1 and 2026 Q2, and did the position grow or shrink?",
    "What was the average portfolio value across all managers in 2026 Q3?",
]


def usage_delta(
    before: dict[str, int],
    after: dict[str, int],
) -> dict[str, int]:
    """Calculate LLM usage attributable to one question."""
    return {
        "calls": after["calls"] - before["calls"],
        "prompt_tokens": (
            after["prompt_tokens"] - before["prompt_tokens"]
        ),
        "completion_tokens": (
            after["completion_tokens"] - before["completion_tokens"]
        ),
    }


def main() -> None:
    question_usage = []

    starting_usage = usage()

    for i, question in enumerate(QUESTIONS, start=1):
        print(f"\n{'=' * 70}")
        print(f"QUESTION {i}")
        print(question)

        before = usage()
        start = time.perf_counter()

        try:
            result = answer_question(question)
        except Exception as exc:
            result = {
                "answer": None,
                "unit": "NONE",
                "sources": [],
            }
            print(f"ERROR: {exc}")

        elapsed = time.perf_counter() - start
        after = usage()

        delta = usage_delta(before, after)

        print("\nANSWER")
        print(json.dumps(result, indent=2))

        print("\nUSAGE")
        print(
            f"calls={delta['calls']}, "
            f"prompt_tokens={delta['prompt_tokens']}, "
            f"completion_tokens={delta['completion_tokens']}, "
            f"elapsed_seconds={elapsed:.3f}"
        )

        question_usage.append(
            {
                "question": question,
                "calls": delta["calls"],
                "prompt_tokens": delta["prompt_tokens"],
                "completion_tokens": delta["completion_tokens"],
                "elapsed_seconds": round(elapsed, 3),
            }
        )

    ending_usage = usage()

    manifest = {
        "questions": question_usage,
        "totals": {
            "calls": (
                ending_usage["calls"]
                - starting_usage["calls"]
            ),
            "prompt_tokens": (
                ending_usage["prompt_tokens"]
                - starting_usage["prompt_tokens"]
            ),
            "completion_tokens": (
                ending_usage["completion_tokens"]
                - starting_usage["completion_tokens"]
            ),
        },
    }

    output_path = (
        Path(__file__).resolve().parents[1]
        / "output"
        / "agent_usage.json"
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with output_path.open(
        "w",
        encoding="utf-8",
    ) as fh:
        json.dump(
            manifest,
            fh,
            indent=2,
        )
        fh.write("\n")

    print(f"\n{'=' * 70}")
    print("COMPLETE")
    print(f"Wrote usage manifest to: {output_path}")
    print(json.dumps(manifest["totals"], indent=2))


if __name__ == "__main__":
    main()