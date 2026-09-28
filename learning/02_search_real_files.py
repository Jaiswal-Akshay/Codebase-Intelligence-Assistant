"""Lesson 3: retrieve information from real project files."""

import re
from pathlib import Path


STOP_WORDS = {
    "a", "an", "and", "are", "at", "be", "but", "by", "can", "do",
    "for", "from", "how", "i", "in", "is", "it", "of", "on", "or",
    "the", "this", "to", "us", "what", "where", "which", "with",
}


def words(text):
    """Convert text into meaningful lowercase words."""
    all_words = set(re.findall(r"[a-zA-Z0-9_]+", text.lower()))
    return all_words - STOP_WORDS


def load_documents(folder):
    """Read Python files from a project folder."""
    documents = []
    folder_path = Path(folder)

    for file_path in folder_path.rglob("*.py"):
        documents.append({
            "name": str(file_path),
            "text": file_path.read_text(encoding="utf-8"),
        })

    return documents


def retrieve(question, documents):
    """Return project files ranked by meaningful matching words."""
    question_words = words(question)
    results = []

    for document in documents:
        document_words = words(document["text"])
        matching_words = question_words.intersection(document_words)

        if matching_words:
            results.append({
                **document,
                "matching_words": ", ".join(sorted(matching_words)),
                "score": len(matching_words),
            })

    return sorted(results, key=lambda result: result["score"], reverse=True)


project_folder = "sample_project"
documents = load_documents(project_folder)

if not documents:
    print(f"No Python files were found in '{project_folder}'.")
    print("Check that the folder exists and contains .py files.")
else:
    print(f"Loaded {len(documents)} Python files from '{project_folder}'.")
    question = input("Ask a question about the project: ")
    matches = retrieve(question, documents)

    if matches:
        print("\nRelevant project information:\n")
        for match in matches:
            print(f"- {match['name']}")
            print(f"  Score: {match['score']}")
            print(f"  Matching words: {match['matching_words']}")
            print(f"  Preview: {match['text'][:250].strip()}\n")
    else:
        print("\nI could not find relevant information.")
