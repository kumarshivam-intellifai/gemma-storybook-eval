from __future__ import annotations

import csv
import json
import math
import statistics
import sys
from collections import Counter
from pathlib import Path
from typing import Any

from generate_full_tests import create_tests
from storybook_assertions_full import evaluate_storybook


def _percentile(values: list[float], percentile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = (len(ordered) - 1) * percentile
    lower = math.floor(index)
    upper = math.ceil(index)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (index - lower)


def main() -> int:
    if len(sys.argv) != 3:
        print(
            "Usage: regrade_full_results.py RESULTS_JSON OUTPUT_DIRECTORY",
            file=sys.stderr,
        )
        return 2

    source_path = Path(sys.argv[1]).resolve()
    output_directory = Path(sys.argv[2]).resolve()
    source = json.loads(source_path.read_text(encoding="utf-8"))
    expected_vars = {
        test["vars"]["case_id"]: test["vars"] for test in create_tests()
    }

    cases: list[dict[str, Any]] = []
    detailed_results: list[dict[str, Any]] = []
    component_failures: Counter[str] = Counter()
    component_passes: Counter[str] = Counter()

    for source_result in source["results"]["results"]:
        case_id = str(source_result["vars"]["case_id"])
        variables = expected_vars[case_id]
        rubric = evaluate_storybook(
            str(source_result["response"]["output"]),
            {"vars": variables},
        )
        latency_ms = int(source_result.get("latencyMs", 0) or 0)
        latency_pass = latency_ms < 240000
        passed = bool(rubric["pass"] and latency_pass)
        promptfoo_score = (float(rubric["score"]) + float(latency_pass)) / 2
        token_usage = source_result["response"].get("tokenUsage", {})
        failed_components: list[str] = []
        for component in rubric["componentResults"]:
            name = component["reason"].split(":", 1)[0]
            if component["pass"]:
                component_passes[name] += 1
            else:
                component_failures[name] += 1
                failed_components.append(name)
        if not latency_pass:
            component_failures["Response under 240 seconds"] += 1
            failed_components.append("Response under 240 seconds")
        else:
            component_passes["Response under 240 seconds"] += 1

        case_row = {
            "case_id": case_id,
            "title": variables["expected_title"],
            "pages": int(variables["expected_pages"]),
            "passed": passed,
            "rubric_score": float(rubric["score"]),
            "promptfoo_score": promptfoo_score,
            "latency_ms": latency_ms,
            "finish_reason": source_result["response"].get(
                "finishReason",
                "",
            ),
            "prompt_tokens": int(token_usage.get("prompt", 0) or 0),
            "completion_tokens": int(token_usage.get("completion", 0) or 0),
            "total_tokens": int(token_usage.get("total", 0) or 0),
            "failed_components": "; ".join(failed_components),
        }
        cases.append(case_row)
        detailed_results.append(
            {
                **case_row,
                "reason": rubric["reason"],
                "componentResults": rubric["componentResults"],
            }
        )

    latencies = [float(case["latency_ms"]) for case in cases]
    passed_count = sum(bool(case["passed"]) for case in cases)
    summary = {
        "source_results": str(source_path),
        "regrade_note": (
            "Saved model outputs regraded after making format-equivalent "
            "headings and article-free story-object matching rubric-valid."
        ),
        "total_cases": len(cases),
        "passed": passed_count,
        "failed": len(cases) - passed_count,
        "pass_rate": passed_count / len(cases) if cases else 0.0,
        "average_rubric_score": (
            statistics.fmean(float(case["rubric_score"]) for case in cases)
            if cases
            else 0.0
        ),
        "average_promptfoo_score": (
            statistics.fmean(float(case["promptfoo_score"]) for case in cases)
            if cases
            else 0.0
        ),
        "latency_ms": {
            "min": min(latencies) if latencies else None,
            "mean": statistics.fmean(latencies) if latencies else None,
            "p50": _percentile(latencies, 0.50),
            "p95": _percentile(latencies, 0.95),
            "max": max(latencies) if latencies else None,
        },
        "tokens": {
            "prompt": sum(int(case["prompt_tokens"]) for case in cases),
            "completion": sum(
                int(case["completion_tokens"]) for case in cases
            ),
            "total": sum(int(case["total_tokens"]) for case in cases),
        },
        "finish_reasons": dict(
            Counter(str(case["finish_reason"]) for case in cases)
        ),
        "component_passes": dict(component_passes),
        "component_failures": dict(component_failures.most_common()),
        "passed_cases": [
            {
                "case_id": case["case_id"],
                "pages": case["pages"],
                "latency_ms": case["latency_ms"],
            }
            for case in cases
            if case["passed"]
        ],
        "failed_cases": [
            {
                "case_id": case["case_id"],
                "pages": case["pages"],
                "rubric_score": case["rubric_score"],
                "latency_ms": case["latency_ms"],
                "failed_components": case["failed_components"],
            }
            for case in cases
            if not case["passed"]
        ],
    }

    output_directory.mkdir(parents=True, exist_ok=True)
    (output_directory / "regraded_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )
    (output_directory / "regraded_results.json").write_text(
        json.dumps(detailed_results, indent=2) + "\n",
        encoding="utf-8",
    )
    with (output_directory / "regraded_cases.csv").open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(cases[0]) if cases else [])
        if cases:
            writer.writeheader()
            writer.writerows(cases)

    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
