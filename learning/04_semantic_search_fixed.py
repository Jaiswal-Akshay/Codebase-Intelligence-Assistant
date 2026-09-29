"""Semantic code search forced to CPU with explicit progress messages."""

import os

# This project is intentionally CPU-only. Set this before importing PyTorch.
os.environ["CUDA_VISIBLE_DEVICES"] = ""

from pathlib import Path

from sentence_transformers import SentenceTransformer, util


def load_documents(folder):
    documents = []

    for file_path in Path(folder).rglob("*.py"):
        documents.append({
            "name": str(file_path),
            "text": file_path.read_text(encoding="utf-8"),
        })

    return documents


try:
    project_folder = "sample_project"
    documents = load_documents(project_folder)

    if not documents:
        print(f"No Python files were found in '{project_folder}'.")
        raise SystemExit

    print(f"Loaded {len(documents)} Python files from '{project_folder}'.", flush=True)
    print("Loading the embedding model on CPU...", flush=True)

    model = SentenceTransformer(
        "all-MiniLM-L6-v2",
        device="cpu",
    )
    print("Embedding model loaded.", flush=True)

    document_embeddings = model.encode(
        [document["text"] for document in documents],
        convert_to_tensor=True,
        show_progress_bar=False,
    )
    print("Project files embedded.", flush=True)

    question = input("Ask a question about the project: ")

    question_embedding = model.encode(
        question,
        convert_to_tensor=True,
        show_progress_bar=False,
    )

    similarity_scores = util.cos_sim(
        question_embedding,
        document_embeddings,
    )[0]

    ranked_results = sorted(
        zip(similarity_scores, documents),
        key=lambda result: float(result[0]),
        reverse=True,
    )

    print("\nSemantic search results:\n")

    for score, document in ranked_results:
        print(f"- {document['name']}")
        print(f"  Similarity score: {float(score):.3f}")

        for line_number, line in enumerate(
            document["text"].splitlines(),
            start=1,
        ):
            print(f"    Line {line_number}: {line}")

        print()

except Exception as error:
    print(f"\nThe program stopped because of this error: {type(error).__name__}: {error}")
    raise
