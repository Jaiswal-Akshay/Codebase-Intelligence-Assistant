"""Lesson 13: validate citations and unsupported claims in RAG answers."""

import argparse
import json
import re
from pathlib import Path
from urllib.request import Request, urlopen

import numpy as np
from fastembed import TextEmbedding


SUPPORTED_EXTENSIONS = {
    ".py": "Python", ".js": "JavaScript", ".jsx": "JavaScript",
    ".ts": "TypeScript", ".tsx": "TypeScript", ".java": "Java",
    ".go": "Go", ".rs": "Rust", ".sql": "SQL", ".md": "Markdown",
    ".json": "JSON", ".yaml": "YAML", ".yml": "YAML", ".toml": "TOML",
}
IGNORED_DIRECTORIES = {
    ".git", ".venv", "venv", "node_modules", "__pycache__",
    "dist", "build", "target", ".next", "coverage",
}
CHUNK_LINES = 120


def words(text):
    stop_words = {
        "a", "an", "and", "are", "at", "be", "but", "by", "can", "do",
        "for", "from", "how", "i", "in", "is", "it", "of", "on", "or",
        "the", "this", "to", "us", "what", "where", "which", "with",
    }
    return set(re.findall(r"[a-zA-Z0-9_]+", text.lower())) - stop_words


def load_chunks(repository):
    chunks = []
    for path in repository.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            continue
        if any(part in IGNORED_DIRECTORIES for part in path.parts):
            continue
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
        except (UnicodeDecodeError, OSError):
            continue

        relative_path = path.relative_to(repository).as_posix()
        for start in range(0, len(lines), CHUNK_LINES):
            end = min(start + CHUNK_LINES, len(lines))
            chunks.append({
                "file": relative_path,
                "language": SUPPORTED_EXTENSIONS[path.suffix.lower()],
                "start_line": start + 1,
                "end_line": end,
                "text": "\n".join(lines[start:end]),
            })
    return chunks


def cosine_similarity(vector, matrix):
    return (matrix @ vector) / (
        np.linalg.norm(matrix, axis=1) * np.linalg.norm(vector)
    )


def called_functions(code):
    names = re.findall(r"\b([A-Za-z_]\w*)\s*\(", code)
    return set(names) - {"if", "for", "while", "print", "return"}


def defines_function(code, function_name):
    patterns = [
        rf"\bdef\s+{re.escape(function_name)}\s*\(",
        rf"\bfunction\s+{re.escape(function_name)}\s*\(",
        rf"\b(?:const|let|var)\s+{re.escape(function_name)}\s*=",
    ]
    return any(re.search(pattern, code) for pattern in patterns)


def retrieve_context(question, chunks):
    model = TextEmbedding()
    embeddings = np.array(list(model.embed([
        f"passage: {chunk['text']}" for chunk in chunks
    ])))
    question_embedding = np.array(list(model.embed([
        f"query: {question}"
    ]))[0])
    semantic_scores = cosine_similarity(question_embedding, embeddings)
    question_words = words(question)

    results = []
    for semantic_score, chunk in zip(semantic_scores, chunks):
        matching_words = question_words.intersection(words(chunk["text"]))
        keyword_score = len(matching_words) / max(len(question_words), 1)
        score = (0.7 * float(semantic_score)) + (0.3 * keyword_score)
        results.append((score, chunk))
    results.sort(key=lambda item: item[0], reverse=True)

    seed_score, seed_chunk = results[0]
    references = called_functions(seed_chunk["text"])
    selected = [(seed_score, seed_chunk)]

    for score, chunk in results:
        if score >= 0.60 and chunk not in [item[1] for item in selected]:
            selected.append((score, chunk))
        if any(defines_function(chunk["text"], name) for name in references):
            if chunk not in [item[1] for item in selected]:
                selected.append((score, chunk))

    context_parts = []
    for score, chunk in selected:
        numbered_code = "\n".join(
            f"Line {line_number}: {line}"
            for line_number, line in enumerate(
                chunk["text"].splitlines(), start=chunk["start_line"]
            )
        )
        context_parts.append(
            f"FILE: {chunk['file']}\n"
            f"LINES: {chunk['start_line']}-{chunk['end_line']}\n"
            f"CODE:\n{numbered_code}"
        )

    return "\n\n---\n\n".join(context_parts), selected


def ask_ollama(prompt):
    body = json.dumps({
        "model": "llama3:latest",
        "prompt": prompt,
        "format": "json",
        "stream": False,
        "options": {"temperature": 0.1},
    }).encode("utf-8")
    request = Request(
        "http://localhost:11434/api/generate",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urlopen(request, timeout=180) as response:
        return json.loads(response.read().decode("utf-8"))["response"]


def normalize_file_name(name, valid_files):
    candidate = str(name).replace("\\", "/")
    if candidate in valid_files:
        return candidate
    matches = [path for path in valid_files if Path(path).name == Path(candidate).name]
    return matches[0] if len(matches) == 1 else None


def validate_answer(answer, selected):
    valid_ranges = {
        chunk["file"]: set(range(chunk["start_line"], chunk["end_line"] + 1))
        for _, chunk in selected
    }
    errors = []
    warnings = []

    evidence_items = answer.get("evidence", [])
    if not evidence_items:
        errors.append("The answer contains no evidence citations.")

    for item in evidence_items:
        file_name = normalize_file_name(item.get("file", ""), valid_ranges)
        if not file_name:
            errors.append(f"Cited file is not in context: {item.get('file')}")
            continue

        lines = item.get("lines", [])
        if isinstance(lines, int):
            lines = [lines]
        invalid_lines = [line for line in lines if line not in valid_ranges[file_name]]
        if invalid_lines:
            errors.append(
                f"Invalid line citation for {file_name}: {invalid_lines}"
            )

    answer_text = str(answer.get("answer", ""))
    context_files = " ".join(valid_ranges).lower()
    context_text = " ".join(
        chunk["text"] for _, chunk in selected
    ).lower()
    if "database" in answer_text.lower() and "database" not in context_text:
        warnings.append("The answer mentions a database that is not shown in context.")
    if answer.get("unsupported_claims"):
        warnings.extend(answer["unsupported_claims"])

    return errors, warnings


parser = argparse.ArgumentParser(description="Validated RAG codebase assistant.")
parser.add_argument("repository", type=Path)
args = parser.parse_args()
repository = args.repository.resolve()

if not repository.is_dir():
    parser.error(f"Repository does not exist: {repository}")

chunks = load_chunks(repository)
if not chunks:
    parser.error("No supported text files were found.")

print(f"Loaded {len(chunks)} searchable chunks from {repository}")
question = input("Ask a question about the project: ")
print("Searching the repository...")
context, selected = retrieve_context(question, chunks)

prompt = f"""You are Codebase Compass, a strict software codebase assistant.
Return valid JSON with exactly these fields:
{{"answer": "...", "evidence": [{{"file": "relative/path", "lines": [1], "claim": "..."}}], "unsupported_claims": []}}

Rules:
- Use only the code in CONTEXT.
- Every factual claim must have an evidence item.
- Use only files and line numbers present in CONTEXT.
- Do not mention a database, API, or external behavior unless shown in CONTEXT.
- Put missing or uncertain claims in unsupported_claims.

CONTEXT:
{context}

USER QUESTION:
{question}
"""

print("\nValidated context files:")
for _, chunk in selected:
    print(f"- {chunk['file']}:{chunk['start_line']}-{chunk['end_line']}")

print("\nAsking the local Ollama model...\n")
raw_response = ask_ollama(prompt)

try:
    answer = json.loads(raw_response)
except json.JSONDecodeError:
    answer = {
        "answer": raw_response,
        "evidence": [],
        "unsupported_claims": ["The model did not return valid JSON."],
    }

errors, warnings = validate_answer(answer, selected)
status = "PASS" if not errors and not warnings else "REVIEW"

print(f"Validation status: {status}\n")
print(f"Answer:\n{answer.get('answer', '')}\n")
print("Evidence:")
for item in answer.get("evidence", []):
    print(f"- {item.get('file')} lines {item.get('lines')}: {item.get('claim')}")

if errors:
    print("\nValidation errors:")
    for error in errors:
        print(f"- {error}")
if warnings:
    print("\nValidation warnings:")
    for warning in warnings:
        print(f"- {warning}")
