"""Lesson 11: ingest multiple repository file types with relative paths."""

import argparse
import json
import re
from pathlib import Path
from urllib.request import Request, urlopen

import numpy as np
from fastembed import TextEmbedding


SUPPORTED_EXTENSIONS = {
    ".py": "Python",
    ".js": "JavaScript",
    ".jsx": "JavaScript",
    ".ts": "TypeScript",
    ".tsx": "TypeScript",
    ".java": "Java",
    ".go": "Go",
    ".rs": "Rust",
    ".sql": "SQL",
    ".md": "Markdown",
    ".json": "JSON",
    ".yaml": "YAML",
    ".yml": "YAML",
    ".toml": "TOML",
}

IGNORED_DIRECTORIES = {
    ".git", ".venv", "venv", "node_modules", "__pycache__",
    "dist", "build", "target", ".next", "coverage",
}
MAX_FILE_BYTES = 500_000
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
        if path.stat().st_size > MAX_FILE_BYTES:
            print(f"Skipped large file: {path.relative_to(repository)}")
            continue

        try:
            lines = path.read_text(encoding="utf-8").splitlines()
        except (UnicodeDecodeError, OSError):
            continue

        relative_path = path.relative_to(repository).as_posix()
        language = SUPPORTED_EXTENSIONS[path.suffix.lower()]

        for start in range(0, len(lines), CHUNK_LINES):
            end = min(start + CHUNK_LINES, len(lines))
            chunks.append({
                "file": relative_path,
                "language": language,
                "start_line": start + 1,
                "end_line": end,
                "text": "\n".join(lines[start:end]),
            })

    return chunks


def cosine_similarity(vector, matrix):
    return (matrix @ vector) / (
        np.linalg.norm(matrix, axis=1) * np.linalg.norm(vector)
    )


def ask_ollama(prompt):
    body = json.dumps({
        "model": "llama3:latest",
        "prompt": prompt,
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


parser = argparse.ArgumentParser(
    description="Ask questions about a multi-language repository."
)
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
embedding_model = TextEmbedding()
chunk_embeddings = np.array(list(embedding_model.embed([
    f"passage: {chunk['text']}" for chunk in chunks
])))
question_embedding = np.array(list(embedding_model.embed([
    f"query: {question}"
]))[0])
semantic_scores = cosine_similarity(question_embedding, chunk_embeddings)
question_words = words(question)

results = []
for semantic_score, chunk in zip(semantic_scores, chunks):
    matching_words = question_words.intersection(words(chunk["text"]))
    keyword_score = len(matching_words) / max(len(question_words), 1)
    combined_score = (0.7 * float(semantic_score)) + (0.3 * keyword_score)
    results.append((combined_score, chunk))

results.sort(key=lambda item: item[0], reverse=True)
selected = results[:4]

context_parts = []
for score, chunk in selected:
    numbered_code = "\n".join(
        f"Line {line_number}: {line}"
        for line_number, line in enumerate(
            chunk["text"].splitlines(), start=chunk["start_line"]
        )
    )
    context_parts.append(
        f"FILE: {chunk['file']} ({chunk['language']})\n"
        f"RELEVANCE: {score:.3f}\n"
        f"CODE:\n{numbered_code}"
    )

context = "\n\n---\n\n".join(context_parts)
print("\nRetrieved context:")
for _, chunk in selected:
    print(f"- {chunk['file']}:{chunk['start_line']}-{chunk['end_line']}")

prompt = f"""You are Codebase Compass, a strict software codebase assistant.
Answer using only the code in CONTEXT.
Explain what is directly shown and cite file paths and line numbers.
Do not invent behavior that is not shown.
If the context is insufficient, say so clearly.

CONTEXT:
{context}

USER QUESTION:
{question}
"""

print("\nAsking the local Ollama model...\n")
print(ask_ollama(prompt))
