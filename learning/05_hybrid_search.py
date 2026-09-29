"""Lesson 6: combine keyword and semantic retrieval."""

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


def cosine_similarity(vector, matrix):
    vector_norm = np.linalg.norm(vector)
    matrix_norms = np.linalg.norm(matrix, axis=1)
    return (matrix @ vector) / (matrix_norms * vector_norm)


documents = load_documents("sample_project")
if not documents:
    print("No Python files were found in 'sample_project'.")
    raise SystemExit

print(f"Loaded {len(documents)} Python files.")
print("Loading the ONNX embedding model...")
embedding_model = TextEmbedding()

document_texts = [f"passage: {document['text']}" for document in documents]
document_embeddings = np.array(list(embedding_model.embed(document_texts)))

question = input("Ask a question about the project: ")
question_words = words(question)
question_embedding = np.array(
    list(embedding_model.embed([f"query: {question}"]))[0]
)
semantic_scores = cosine_similarity(question_embedding, document_embeddings)

results = []
for semantic_score, document in zip(semantic_scores, documents):
    matching_words = question_words.intersection(words(document["text"]))
    keyword_score = len(matching_words) / max(len(question_words), 1)

    # Semantic search handles meaning; keyword search rewards exact matches.
    combined_score = (0.7 * float(semantic_score)) + (0.3 * keyword_score)
    results.append((combined_score, float(semantic_score), matching_words, document))

results.sort(key=lambda result: result[0], reverse=True)

print("\nHybrid search results:\n")

for combined_score, semantic_score, matching_words, document in results:
    if combined_score < 0.58:
        continue

    print(f"- {document['name']}")
    print(f"  Combined score: {combined_score:.3f}")
    print(f"  Semantic score: {semantic_score:.3f}")
    print(f"  Matching keywords: {', '.join(sorted(matching_words)) or 'none'}")

    for line_number, line in enumerate(document["text"].splitlines(), start=1):
        print(f"    Line {line_number}: {line}")

    print()
