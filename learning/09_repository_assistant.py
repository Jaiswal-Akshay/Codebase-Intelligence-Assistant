"""Lesson 10: run the dependency-aware assistant on any local repository."""

import argparse
import json
import re
from pathlib import Path
from urllib.request import Request, urlopen

import numpy as np
from fastembed import TextEmbedding


STOP_WORDS = {
    "a", "an", "and", "are", "at", "be", "but", "by", "can", "do",
    "for", "from", "how", "i", "in", "is", "it", "of", "on", "or",
    "the", "this", "to", "us", "what", "where", "which", "with",
}


def words(text):
    return set(re.findall(r"[a-zA-Z0-9_]+", text.lower())) - STOP_WORDS


def load_documents(folder):
    documents = []
    ignored = {".git", ".venv", "venv", "node_modules", "__pycache__"}

    for path in Path(folder).rglob("*.py"):
        if any(part in ignored for part in path.parts):
            continue
        try:
            documents.append({
                "name": str(path),
                "text": path.read_text(encoding="utf-8"),
            })
        except UnicodeDecodeError:
            print(f"Skipped non-text file: {path}")

    return documents


def cosine_similarity(vector, matrix):
    return (matrix @ vector) / (
        np.linalg.norm(matrix, axis=1) * np.linalg.norm(vector)
    )


def find_called_functions(code):
    names = re.findall(r"\b([A-Za-z_]\w*)\s*\(", code)
    python_keywords = {"if", "for", "while", "print", "return"}
    return set(names) - python_keywords


def find_function_definitions(function_names, documents):
    related = []
    for document in documents:
        if any(
            re.search(rf"\bdef\s+{re.escape(name)}\s*\(", document["text"])
            for name in function_names
        ):
            related.append(document)
    return related


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
    description="Ask questions about a local Python repository."
)
parser.add_argument("repository", type=Path)
args = parser.parse_args()
repository = args.repository.resolve()

if not repository.is_dir():
    parser.error(f"Repository does not exist: {repository}")

documents = load_documents(repository)
if not documents:
    parser.error("No readable Python files were found in the repository.")

print(f"Loaded {len(documents)} Python files from {repository}")
question = input("Ask a question about the project: ")

print("Searching the codebase...")
embedding_model = TextEmbedding()
document_embeddings = np.array(list(embedding_model.embed([
    f"passage: {document['text']}" for document in documents
])))
question_embedding = np.array(list(embedding_model.embed([
    f"query: {question}"
]))[0])
scores = cosine_similarity(question_embedding, document_embeddings)

seed_index = int(np.argmax(scores))
seed_document = documents[seed_index]
called_functions = find_called_functions(seed_document["text"])
dependency_documents = find_function_definitions(called_functions, documents)

selected_documents = [seed_document]
for document in dependency_documents:
    if document not in selected_documents:
        selected_documents.append(document)

print("\nDependency-aware context files:")
for document in selected_documents:
    print(f"- {document['name']}")

context_parts = []
for document in selected_documents:
    numbered_code = "\n".join(
        f"Line {line_number}: {line}"
        for line_number, line in enumerate(document["text"].splitlines(), start=1)
    )
    context_parts.append(f"FILE: {document['name']}\nCODE:\n{numbered_code}")

context = "\n\n---\n\n".join(context_parts)
prompt = f"""You are Codebase Compass, a strict software codebase assistant.
Answer the user's question using only the code in CONTEXT.
Explain what is directly shown and cite file paths and line numbers.
Do not invent behavior whose implementation is not shown.
If information is missing, say so clearly.

CONTEXT:
{context}

USER QUESTION:
{question}
"""

print("\nAsking the local Ollama model...\n")
print(ask_ollama(prompt))
