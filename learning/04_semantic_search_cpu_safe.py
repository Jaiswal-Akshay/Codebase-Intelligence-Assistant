"""Semantic search with a conservative CPU/NumPy embedding path."""

import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

from pathlib import Path

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

torch.set_num_threads(1)
torch.set_num_interop_threads(1)


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


project_folder = "sample_project"
documents = load_documents(project_folder)

if not documents:
    print(f"No Python files were found in '{project_folder}'.")
    raise SystemExit

print(f"Loaded {len(documents)} Python files from '{project_folder}'.", flush=True)
print("Loading the embedding model on CPU...", flush=True)

model = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")
print("Embedding model loaded.", flush=True)

print("Creating document embeddings...", flush=True)
document_embeddings = model.encode(
    [document["text"] for document in documents],
    convert_to_numpy=True,
    batch_size=1,
    show_progress_bar=False,
)
print("Document embeddings created.", flush=True)

question = input("Ask a question about the project: ")
question_embedding = model.encode(
    question,
    convert_to_numpy=True,
    show_progress_bar=False,
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
