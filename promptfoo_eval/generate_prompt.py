from __future__ import annotations

import json
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parent.parent
CASES_FILE = PROJECT_ROOT / "prompt_cases.json"
SYSTEM_PROMPT_FILE = PROJECT_ROOT / "system_prompt.txt"


def create_prompt(context: dict[str, Any]) -> str:
    case_id = str(
        context.get("vars", {}).get(
            "case_id", "latency_20page_ava_and_the_glowing_pebble"
        )
    )
    cases = json.loads(CASES_FILE.read_text(encoding="utf-8"))
    try:
        case = next(item for item in cases if item["id"] == case_id)
    except StopIteration as exc:
        raise ValueError(f"Unknown storybook case id: {case_id}") from exc

    messages = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT_FILE.read_text(encoding="utf-8").rstrip("\n"),
        },
        {"role": "user", "content": str(case["input"])},
    ]
    return json.dumps(messages, ensure_ascii=False)
