"""Lesson 8: generate a grounded answer with a local Ollama model."""

import json
import re
from pathlib import Path
from urllib.error import URLError
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
    for file_path in Path(folder).rglob("*.py"):
        documents.append({
            "name": str(file_path),
            "text": file_path.read_text(encoding="utf-8"),
        })
    return documents


def cosine_similarity(vector, matrix):
    return (matrix @ vector) / (
        np.linalg.norm(matrix, axis=1) * np.linalg.norm(vector)
    )


def retrieve_context(question, documents, embedding_model):
    document_embeddings = np.array(list(embedding_model.embed([
        f"passage: {document['text']}" for document in documents
    ])))
    question_embedding = np.array(list(embedding_model.embed([
        f"query: {question}"
    ]))[0])
    semantic_scores = cosine_similarity(question_embedding, document_embeddings)
    question_words = words(question)

    results = []
    for semantic_score, document in zip(semantic_scores, documents):
        matching_words = question_words.intersection(words(document["text"]))
        keyword_score = len(matching_words) / max(len(question_words), 1)
        combined_score = (0.7 * float(semantic_score)) + (0.3 * keyword_score)
        results.append((combined_score, document))

    selected = [result for result in results if result[0] >= 0.58]
    selected.sort(key=lambda result: result[0], reverse=True)

    context_parts = []
    for score, document in selected:
        numbered_code = "\n".join(
            f"Line {line_number}: {line}"
            for line_number, line in enumerate(
                document["text"].splitlines(), start=1
            )
        )
        context_parts.append(
            f"FILE: {document['name']}\n"
            f"RELEVANCE SCORE: {score:.3f}\n"
            f"CODE:\n{numbered_code}"
        )

    return "\n\n---\n\n".join(context_parts)


def ask_ollama(prompt, model="llama3:latest"):
    request_body = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": 0.1},
    }
    request = Request(
        "http://localhost:11434/api/generate",
        data=json.dumps(request_body).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    with urlopen(request, timeout=180) as response:
        return json.loads(response.read().decode("utf-8"))["response"]


documents = load_documents("sample_project")
if not documents:
    print("No Python files were found in 'sample_project'.")
    raise SystemExit

question = input("Ask a question about the project: ")
print("Searching the codebase...")
embedding_model = TextEmbedding()
context = retrieve_context(question, documents, embedding_model)

if not context:
    print("I could not find enough relevant code to answer confidently.")
    raise SystemExit

prompt = f"""You are Codebase Compass, a software codebase assistant.
Answer the user's question using only the provided CONTEXT.
Explain the implementation in plain language.
Cite the relevant file path and line number.
Do not invent behavior that is not shown in the context.
If the context is insufficient, say that clearly.

CONTEXT:
{context}

USER QUESTION:
{question}
"""

print("Asking the local Ollama model...")

try:
    answer = ask_ollama(prompt)
except URLError:
    print("Could not connect to Ollama at http://localhost:11434.")
    print("Make sure the Ollama application is running, then try again.")
    raise SystemExit(1)

print("\nAnswer:\n")
print(answer)
