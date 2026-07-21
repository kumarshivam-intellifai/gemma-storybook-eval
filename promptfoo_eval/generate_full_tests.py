from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parent.parent
CASES_FILE = PROJECT_ROOT / "prompt_cases.json"


MORAL_TERM_STEMS = {
    "sharing makes a small light brighter": ["shar", "light", "bright"],
    "patience helps wonderful things grow": ["patien", "grow"],
    "asking for help can solve a hard problem": ["ask", "help", "solv"],
    "telling the truth keeps friendships strong": ["truth", "friend", "strong"],
    "careful listening helps everyone feel seen": ["listen", "feel", "seen"],
    "small brave steps can lead to big discoveries": [
        "brave",
        "step",
        "discover",
    ],
    "cleaning up together makes work feel lighter": [
        "clean",
        "together",
        "work",
    ],
    "kind words can change a gloomy day": ["kind", "word", "gloom"],
}


def _capture(pattern: str, text: str, field: str) -> str:
    match = re.search(pattern, text, flags=re.IGNORECASE | re.MULTILINE)
    if not match:
        raise ValueError(f"Could not derive {field} from case input")
    return match.group(1).strip()


def _case_variables(case: dict[str, Any]) -> dict[str, Any]:
    case_input = str(case["input"])
    summary = _capture(r"^Story Summary:\s*(.+)$", case_input, "summary")
    child_name = _capture(
        r"\bA child named\s+([A-Za-z][A-Za-z'-]*)\b",
        summary,
        "child character",
    )
    helper_phrase = _capture(
        r"\bWith help from\s+(.+?),\s+"
        rf"{re.escape(child_name)}\s+learns that\b",
        summary,
        "helper character",
    )
    helper_name = helper_phrase.rsplit(maxsplit=1)[-1]
    found_object = _capture(
        rf"\b{re.escape(child_name)}\s+finds\s+(.+?)\s+in\s+the\b",
        summary,
        "story object",
    )
    moral = _capture(r"\blearns that\s+(.+?)\.\s*$", summary, "moral")
    moral_key = moral.lower()
    if moral_key not in MORAL_TERM_STEMS:
        raise ValueError(f"Unknown moral in {case['id']}: {moral}")

    return {
        "case_id": str(case["id"]),
        "expected_pages": int(
            _capture(
                r"^Number of Pages:\s*(\d+)\s*$",
                case_input,
                "page count",
            )
        ),
        "expected_title": _capture(
            r"^Story Title:\s*(.+)$",
            case_input,
            "title",
        ),
        "expected_author": _capture(
            r"^Author Name:\s*(.+)$",
            case_input,
            "author",
        ),
        "expected_characters": f"{child_name},{helper_name}",
        "expected_object": found_object,
        "expected_moral_stems": ",".join(MORAL_TERM_STEMS[moral_key]),
    }


def create_tests() -> list[dict[str, Any]]:
    cases = json.loads(CASES_FILE.read_text(encoding="utf-8"))
    tests: list[dict[str, Any]] = []

    for case in cases:
        variables = _case_variables(case)
        tests.append(
            {
                "description": (
                    f"{variables['expected_title']} "
                    f"({variables['expected_pages']} pages)"
                ),
                "vars": variables,
                "metadata": {
                    "case_id": case["id"],
                    "difficulty": case["difficulty"],
                    "pages": variables["expected_pages"],
                    "max_tokens": case["max_tokens"],
                },
                # Promptfoo merges test options into provider options. This
                # keeps each request's cap identical to prompt_cases.json.
                "options": {"max_tokens": int(case["max_tokens"])},
            }
        )

    return tests
