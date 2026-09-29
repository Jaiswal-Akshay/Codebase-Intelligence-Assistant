"""Semantic search using FastEmbed instead of PyTorch."""

from pathlib import Path

import numpy as np
from fastembed import TextEmbedding


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
print("Embedding model loaded.")

document_texts = [f"passage: {document['text']}" for document in documents]
document_embeddings = np.array(list(embedding_model.embed(document_texts)))

question = input("Ask a question about the project: ")
question_embedding = np.array(
    list(embedding_model.embed([f"query: {question}"]))[0]
)

scores = cosine_similarity(question_embedding, document_embeddings)
ranked_results = sorted(
    zip(scores, documents),
    key=lambda result: float(result[0]),
    reverse=True,
)

print("\nSemantic search results:\n")

for score, document in ranked_results:
    print(f"- {document['name']}")
    print(f"  Similarity score: {float(score):.3f}")

    for line_number, line in enumerate(document["text"].splitlines(), start=1):
        print(f"    Line {line_number}: {line}")

    print()
