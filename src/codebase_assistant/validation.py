"""Citation and unsupported-claim validation."""

from pathlib import Path

from .models import Chunk, ValidationReport


def validate_answer(answer: dict, chunks: list[Chunk]) -> ValidationReport:
    valid_ranges = {c.file: set(range(c.start_line, c.end_line + 1)) for c in chunks}
    errors: list[str] = []
    warnings: list[str] = []
    evidence = answer.get("evidence", [])
    if not evidence:
        errors.append("The answer contains no evidence citations.")

    for item in evidence:
        candidate = str(item.get("file", "")).replace("\\", "/")
        matches = [path for path in valid_ranges if path == candidate or Path(path).name == Path(candidate).name]
        if len(matches) != 1:
            errors.append(f"Cited file is not uniquely present in context: {item.get('file')}")
            continue
        lines = item.get("lines", [])
        if isinstance(lines, int):
            lines = [lines]
        invalid = [line for line in lines if line not in valid_ranges[matches[0]]]
        if invalid:
            errors.append(f"Invalid line citation for {matches[0]}: {invalid}")

    context_text = " ".join(c.text for c in chunks).lower()
    answer_text = str(answer.get("answer", "")).lower()
    if "database" in answer_text and "database" not in context_text:
        warnings.append("The answer mentions a database that is not shown in context.")
    warnings.extend(str(item) for item in answer.get("unsupported_claims", []))
    return ValidationReport("PASS" if not errors and not warnings else "REVIEW", errors, warnings)
