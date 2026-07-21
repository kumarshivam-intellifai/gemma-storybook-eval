from __future__ import annotations

import csv
import json
import math
import statistics
import sys
from collections import Counter
from pathlib import Path
from typing import Any


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


def _rows(payload: dict[str, Any]) -> list[dict[str, Any]]:
    results = payload.get("results", {}).get("results", [])
    rows: list[dict[str, Any]] = []
    for result in results:
        variables = result.get("vars", {})
        grading = result.get("gradingResult") or {}
        component_results = grading.get("componentResults", [])
        if component_results and isinstance(
            component_results[0].get("componentResults"),
            list,
        ):
            component_results = component_results[0]["componentResults"]
        token_usage = result.get("response", {}).get("tokenUsage") or {}
        rows.append(
            {
                "case_id": variables.get("case_id", ""),
                "title": variables.get("expected_title", ""),
                "pages": int(variables.get("expected_pages", 0)),
                "passed": bool(result.get("success", False)),
                "score": float(result.get("score", 0.0)),
                "latency_ms": int(result.get("latencyMs", 0) or 0),
                "prompt_tokens": int(token_usage.get("prompt", 0) or 0),
                "completion_tokens": int(token_usage.get("completion", 0) or 0),
                "total_tokens": int(token_usage.get("total", 0) or 0),
                "reason": grading.get("reason", result.get("error", "")),
                "failed_components": "; ".join(
                    component.get("reason", "").split(":", 1)[0]
                    for component in component_results
                    if not component.get("pass", False)
                ),
            }
        )
    return rows


def main() -> int:
    if len(sys.argv) != 3:
        print(
            "Usage: summarize_full_results.py RESULTS_JSON OUTPUT_DIRECTORY",
            file=sys.stderr,
        )
        return 2

    results_path = Path(sys.argv[1]).resolve()
    output_directory = Path(sys.argv[2]).resolve()
    payload = json.loads(results_path.read_text(encoding="utf-8"))
    rows = _rows(payload)
    latencies = [float(row["latency_ms"]) for row in rows]
    passed = sum(bool(row["passed"]) for row in rows)
    failed = len(rows) - passed
    total_tokens = sum(int(row["total_tokens"]) for row in rows)
    component_failures: Counter[str] = Counter()
    for row in rows:
        for component in str(row["failed_components"]).split("; "):
            if component:
                component_failures[component] += 1

    summary = {
        "total_cases": len(rows),
        "passed": passed,
        "failed": failed,
        "pass_rate": passed / len(rows) if rows else 0.0,
        "average_score": (
            statistics.fmean(float(row["score"]) for row in rows)
            if rows
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
            "prompt": sum(int(row["prompt_tokens"]) for row in rows),
            "completion": sum(int(row["completion_tokens"]) for row in rows),
            "total": total_tokens,
        },
        "component_failures": dict(component_failures.most_common()),
        "failed_cases": [
            {
                "case_id": row["case_id"],
                "pages": row["pages"],
                "score": row["score"],
                "latency_ms": row["latency_ms"],
                "failed_components": row["failed_components"],
                "reason": row["reason"],
            }
            for row in rows
            if not row["passed"]
        ],
    }

    output_directory.mkdir(parents=True, exist_ok=True)
    (output_directory / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )
    with (output_directory / "cases.csv").open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]) if rows else [])
        if rows:
            writer.writeheader()
            writer.writerows(rows)

    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
