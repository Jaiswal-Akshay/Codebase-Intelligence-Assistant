"""Lesson 4: show exact source-code line references."""

import re
from pathlib import Path


STOP_WORDS = {
    "a", "an", "and", "are", "at", "be", "but", "by", "can", "do",
    "for", "from", "how", "i", "in", "is", "it", "of", "on", "or",
    "the", "this", "to", "us", "what", "where", "which", "with",
}


def words(text):
    all_words = set(re.findall(r"[a-zA-Z0-9_]+", text.lower()))
    return all_words - STOP_WORDS


def load_documents(folder):
    documents = []

    for file_path in Path(folder).rglob("*.py"):
        documents.append({
            "name": str(file_path),
            "text": file_path.read_text(encoding="utf-8"),
        })

    return documents


def retrieve(question, documents):
    question_words = words(question)
    results = []

    for document in documents:
        document_words = words(document["text"])
        matching_words = question_words.intersection(document_words)

        if not matching_words:
            continue

        matching_lines = []
        for line_number, line in enumerate(document["text"].splitlines(), start=1):
            if words(line).intersection(matching_words):
                matching_lines.append((line_number, line.strip()))

        results.append({
            **document,
            "matching_words": ", ".join(sorted(matching_words)),
            "score": len(matching_words),
            "matching_lines": matching_lines,
        })

    return sorted(results, key=lambda result: result["score"], reverse=True)


project_folder = "sample_project"
documents = load_documents(project_folder)

print(f"Loaded {len(documents)} Python files from '{project_folder}'.")
question = input("Ask a question about the project: ")
matches = retrieve(question, documents)

if matches:
    print("\nRelevant project information:\n")

    for match in matches:
        print(f"- {match['name']}")
        print(f"  Score: {match['score']}")
        print(f"  Matching words: {match['matching_words']}")
        print("  Relevant lines:")

        for line_number, line in match["matching_lines"]:
            print(f"    Line {line_number}: {line}")
else:
    print("\nI could not find relevant information.")
