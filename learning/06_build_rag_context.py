"""Lesson 7: package retrieved code as grounded RAG context."""

import re
from pathlib import Path

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


documents = load_documents("sample_project")
if not documents:
    print("No Python files were found in 'sample_project'.")
    raise SystemExit

embedding_model = TextEmbedding()
document_embeddings = np.array(list(embedding_model.embed([
    f"passage: {document['text']}" for document in documents
])))

question = input("Ask a question about the project: ")
question_words = words(question)
question_embedding = np.array(list(embedding_model.embed([
    f"query: {question}"
]))[0])
semantic_scores = cosine_similarity(question_embedding, document_embeddings)

results = []
for semantic_score, document in zip(semantic_scores, documents):
    matching_words = question_words.intersection(words(document["text"]))
    keyword_score = len(matching_words) / max(len(question_words), 1)
    combined_score = (0.7 * float(semantic_score)) + (0.3 * keyword_score)
    results.append((combined_score, document))

results.sort(key=lambda result: result[0], reverse=True)

selected = [result for result in results if result[0] >= 0.58]

if not selected:
    print("\nNo sufficiently relevant code was found.")
    print("The assistant should say it does not have enough evidence.")
    raise SystemExit

context_parts = []
for score, document in selected:
    numbered_code = "\n".join(
        f"Line {line_number}: {line}"
        for line_number, line in enumerate(document["text"].splitlines(), start=1)
    )
    context_parts.append(
        f"FILE: {document['name']}\n"
        f"RELEVANCE SCORE: {score:.3f}\n"
        f"CODE:\n{numbered_code}"
    )

context = "\n\n---\n\n".join(context_parts)

prompt = f"""You are a software codebase assistant.
Answer the user's question only using the code in the CONTEXT.
Explain the implementation clearly and cite file names and line numbers.
If the context does not contain enough evidence, say so.

CONTEXT:
{context}

USER QUESTION:
{question}
"""

print("\nGrounded RAG context prepared for an AI model:\n")
print(prompt)
