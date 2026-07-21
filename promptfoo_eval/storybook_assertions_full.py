from __future__ import annotations

import re
from typing import Any


def _result(
    name: str,
    passed: bool,
    reason: str,
    score: float | None = None,
) -> dict[str, Any]:
    return {
        "pass": passed,
        "score": float(passed) if score is None else max(0.0, min(1.0, score)),
        "reason": f"{name}: {reason}",
    }


def _section(
    output: str,
    start_pattern: str,
    end_pattern: str | None,
) -> str:
    starts = list(
        re.finditer(
            start_pattern,
            output,
            flags=re.IGNORECASE | re.MULTILINE,
        )
    )
    if not starts:
        return ""
    start = starts[-1]
    tail = output[start.end() :]
    if end_pattern:
        end = re.search(
            end_pattern,
            tail,
            flags=re.IGNORECASE | re.MULTILINE,
        )
        if end:
            tail = tail[: end.start()]
    return tail


def _page_numbers(section: str) -> list[int]:
    return [
        int(value)
        for value in re.findall(
            r"(?im)^\s*(?:#{1,6}\s*)?(?:\*\*)?"
            r"Page\s+(\d+)(?:\*\*)?\b",
            section,
        )
    ]


def _exact_pages_component(
    name: str,
    section: str,
    expected_pages: int,
) -> dict[str, Any]:
    numbers = _page_numbers(section)
    expected = list(range(1, expected_pages + 1))
    passed = numbers == expected
    unique_valid = len(set(numbers).intersection(expected))
    score = unique_valid / expected_pages if expected_pages else 0.0
    return _result(
        name,
        passed,
        f"found page sequence {numbers}; expected {expected}",
        score,
    )


def _contains_stem(text: str, stem: str) -> bool:
    return bool(
        re.search(
            rf"(?i)\b{re.escape(stem)}[A-Za-z'-]*\b",
            text,
        )
    )


def _asset_heading(pattern: str, block: str) -> re.Match[str] | None:
    return re.search(
        rf"(?i)\*{{0,2}}{pattern}:\*{{0,2}}",
        block,
    )


def evaluate_storybook(output: str, context: dict[str, Any]) -> dict[str, Any]:
    variables = context.get("vars", {})
    expected_pages = int(variables["expected_pages"])
    expected_title = str(variables["expected_title"])
    expected_author = str(variables["expected_author"])
    expected_object = str(variables["expected_object"])
    expected_object_terms = [
        term.lower()
        for term in re.findall(r"[A-Za-z][A-Za-z'-]*", expected_object)
        if term.lower() not in {"a", "an", "the"}
    ]
    expected_characters = [
        name.strip()
        for name in str(variables["expected_characters"]).split(",")
        if name.strip()
    ]
    expected_moral_stems = [
        stem.strip()
        for stem in str(variables["expected_moral_stems"]).split(",")
        if stem.strip()
    ]

    stage_1 = _section(
        output,
        r"(?im)^\s*#{0,6}\s*Stage\s*1\b[^\n]*",
        r"(?im)^\s*#{0,6}\s*Stage\s*2\b",
    )
    stage_2 = _section(
        output,
        r"(?im)^\s*#{0,6}\s*Stage\s*2\b[^\n]*",
        r"(?im)^\s*#{0,6}\s*Stage\s*3\b",
    )
    stage_3 = _section(
        output,
        r"(?im)^\s*#{0,6}\s*Stage\s*3\b[^\n]*",
        None,
    )
    page_output_section = _section(
        stage_2,
        r"^\s*#{1,6}\s*Step\s*2\b[^\n]*",
        None,
    )
    if not page_output_section:
        page_output_section = stage_2

    reasoning_markers = [
        r"(?im)^\s*Thinking Process\s*:",
        r"(?i)<think>",
        r"(?im)^\s*\d+\.\s+\*\*Analyze the Request",
        r"(?im)^\s*\*Self-Correction",
    ]
    leaked_markers = [
        marker for marker in reasoning_markers if re.search(marker, output)
    ]

    character_sheet_missing = [
        name
        for name in expected_characters
        if not re.search(
            rf"(?i)\b{re.escape(name)}\b",
            stage_2,
        )
    ]
    character_sheet_present = bool(
        re.search(r"(?i)Character Design Sheet", stage_2)
        and not character_sheet_missing
    )

    page_heading_matches = list(
        re.finditer(
            r"(?im)^\s*(?:[-*+]\s+)?(?:#{1,6}\s*)?(?:\*\*)?"
            r"Page\s+(\d+)(?:\*\*)?(?:\s*[:—-].*)?\s*$",
            page_output_section,
        )
    )
    page_blocks: list[str] = []
    for index, match in enumerate(page_heading_matches):
        block_end = (
            page_heading_matches[index + 1].start()
            if index + 1 < len(page_heading_matches)
            else len(page_output_section)
        )
        page_blocks.append(page_output_section[match.end() : block_end])

    story_text_count = 0
    image_prompt_count = 0
    after_prompt_count = 0
    style_count = 0
    for block in page_blocks:
        image_heading = _asset_heading(r"(?:Page\s+)?Image Prompt", block)
        if not image_heading:
            continue
        image_prompt_count += 1
        story_text = block[: image_heading.start()]
        story_text = re.sub(
            r"(?im)^\s*(?:[-*+]\s+)?"
            r"\*{0,2}(?:Page\s+)?Story Text:\*{0,2}\s*",
            "",
            story_text,
            count=1,
        ).strip()
        if len(re.findall(r"\b\w+\b", story_text)) >= 5:
            story_text_count += 1
        if _asset_heading(
            r"(?:After Image Prompt|Character Descriptions?|Characters?)",
            block,
        ):
            after_prompt_count += 1
        if re.search(
            r"(?i)(?:\*\*)?Style:\s*(?:\*\*)?\s*3d-Pixar Style",
            block,
        ):
            style_count += 1
    coverage_counts = [
        story_text_count,
        image_prompt_count,
        after_prompt_count,
        style_count,
    ]
    coverage_score = sum(
        min(count, expected_pages) / expected_pages for count in coverage_counts
    ) / len(coverage_counts)
    page_assets_complete = all(
        count == expected_pages for count in coverage_counts
    )

    cover_present = bool(
        stage_3
        and re.search(r"(?i)Cover Illustration Prompt", stage_3)
        and expected_title.lower() in stage_3.lower()
        and expected_author.lower() in stage_3.lower()
        and re.search(
            r"(?i)(?:\*\*)?Style:\s*(?:\*\*)?\s*3d-Pixar Style",
            stage_3,
        )
    )

    lower_output = output.lower()
    missing_story_terms = [
        term
        for term in expected_characters
        if term.lower() not in lower_output
    ]
    missing_object_terms = [
        term for term in expected_object_terms if term not in lower_output
    ]
    missing_moral_stems = [
        stem
        for stem in expected_moral_stems
        if not _contains_stem(output, stem)
    ]
    moral_stems_found = len(expected_moral_stems) - len(missing_moral_stems)
    moral_terms_present = moral_stems_found >= min(
        2,
        len(expected_moral_stems),
    )
    story_continuity = (
        not missing_story_terms
        and not missing_object_terms
        and moral_terms_present
    )

    missing_expected_characters = [
        name
        for name in expected_characters
        if not re.search(rf"(?i)\b{re.escape(name)}\b", output)
    ]
    character_description_text = "\n".join(
        block[
            heading.end() :
        ]
        for block in page_blocks
        if (
            heading := _asset_heading(
                r"(?:After Image Prompt|Character Descriptions?|Characters?)",
                block,
            )
        )
    )
    described_character_names = sorted(
        set(
            re.findall(
                r"(?im)^\s*(?:[-*+]\s+)?"
                r"\*\*([A-Z][A-Za-z' -]+):\*\*",
                character_description_text,
            )
        )
    )
    non_character_labels = {
        "after image prompt",
        "character description",
        "character descriptions",
        "characters",
        "image prompt",
        "page image prompt",
        "page story text",
        "story text",
        "style",
    }
    described_character_names = [
        name
        for name in described_character_names
        if name.strip().lower() not in non_character_labels
    ]
    unexpected_characters = [
        name
        for name in described_character_names
        if not any(
            re.search(rf"(?i)\b{re.escape(expected)}\b", name)
            for expected in expected_characters
        )
    ]
    character_limit_passed = (
        not missing_expected_characters and not unexpected_characters
    )

    forbidden_page = expected_pages + 1
    has_extra_page = bool(
        re.search(
            rf"(?im)^\s*(?:#{{1,6}}\s*)?(?:\*\*)?"
            rf"Page\s+{forbidden_page}(?:\*\*)?\b",
            stage_1 + "\n" + stage_2,
        )
    )

    components = [
        _result(
            "No hidden reasoning",
            not leaked_markers,
            (
                "no reasoning markers found"
                if not leaked_markers
                else f"found markers {leaked_markers}"
            ),
        ),
        _exact_pages_component(
            "Stage 1 exact page count",
            stage_1,
            expected_pages,
        ),
        _exact_pages_component(
            "Stage 2 exact page count",
            stage_2,
            expected_pages,
        ),
        _result(
            "Character design sheet",
            character_sheet_present,
            (
                f"contains designs for {expected_characters}"
                if character_sheet_present
                else f"missing sheet characters {character_sheet_missing}"
            ),
        ),
        _result(
            "Page asset coverage",
            page_assets_complete,
            (
                f"story_text={story_text_count}, "
                f"image_prompt={image_prompt_count}, "
                f"after_prompt={after_prompt_count}, "
                f"style_marker={style_count}; expected {expected_pages} each"
            ),
            coverage_score,
        ),
        _result(
            "Cover requirements",
            cover_present,
            (
                "cover has exact title, author, prompt label, and style"
                if cover_present
                else "cover is missing or lacks title/author/style"
            ),
        ),
        _result(
            "Story continuity and moral",
            story_continuity,
            (
                "characters, story object, and moral terms are present"
                if story_continuity
                else (
                    f"missing characters={missing_story_terms}, "
                    f"missing object terms={missing_object_terms}, "
                    f"moral stems found={moral_stems_found}/"
                    f"{len(expected_moral_stems)}, "
                    f"missing moral stems={missing_moral_stems}"
                )
            ),
        ),
        _result(
            "Character limit and identity",
            character_limit_passed,
            (
                f"expected only {expected_characters}"
                if character_limit_passed
                else (
                    f"missing expected={missing_expected_characters}, "
                    f"unexpected described={unexpected_characters}, "
                    f"all described={described_character_names}"
                )
            ),
        ),
        _result(
            f"No Page {forbidden_page}",
            not has_extra_page,
            (
                f"Page {forbidden_page} is absent"
                if not has_extra_page
                else f"unexpected Page {forbidden_page} found"
            ),
        ),
    ]

    passed = all(component["pass"] for component in components)
    score = sum(component["score"] for component in components) / len(components)
    failed_names = [
        component["reason"].split(":", 1)[0]
        for component in components
        if not component["pass"]
    ]
    reason = (
        "All storybook requirements passed"
        if passed
        else f"Failed components: {', '.join(failed_names)}"
    )
    return {
        "pass": passed,
        "score": score,
        "reason": reason,
        "componentResults": components,
    }
