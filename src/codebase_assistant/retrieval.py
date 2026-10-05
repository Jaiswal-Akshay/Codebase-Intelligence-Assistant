"""Hybrid semantic/keyword retrieval with dependency expansion."""

import re

import numpy as np

from .config import STOP_WORDS
from .models import Chunk, RetrievalResult


def tokenize(text: str) -> set[str]:
    return set(re.findall(r"[a-zA-Z0-9_]+", text.lower())) - STOP_WORDS


def cosine_similarity(vector: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    return (matrix @ vector) / (np.linalg.norm(matrix, axis=1) * np.linalg.norm(vector))


def called_functions(code: str) -> set[str]:
    names = re.findall(r"\b([A-Za-z_]\w*)\s*\(", code)
    return set(names) - {"if", "for", "while", "print", "return"}


def defines_function(code: str, name: str) -> bool:
    patterns = [
        rf"\bdef\s+{re.escape(name)}\s*\(",
        rf"\bfunction\s+{re.escape(name)}\s*\(",
        rf"\b(?:const|let|var)\s+{re.escape(name)}\s*=",
    ]
    return any(re.search(pattern, code) for pattern in patterns)


class HybridRetriever:
    def __init__(self, chunks: list[Chunk]) -> None:
        try:
            from fastembed import TextEmbedding
        except ImportError as error:
            raise RuntimeError(
                "fastembed is required for retrieval. Install dependencies with "
                "'python -m pip install -e .'."
            ) from error
        self.chunks = chunks
        self.model = TextEmbedding()
        self.embeddings = np.array(list(self.model.embed([f"passage: {c.text}" for c in chunks])))

    def search(self, question: str, dependency_expansion: bool = True) -> list[RetrievalResult]:
        query_embedding = np.array(list(self.model.embed([f"query: {question}"]))[0])
        semantic_scores = cosine_similarity(query_embedding, self.embeddings)
        query_words = tokenize(question)
        results = []
        for semantic_score, chunk in zip(semantic_scores, self.chunks):
            keyword_score = len(query_words.intersection(tokenize(chunk.text))) / max(len(query_words), 1)
            results.append(RetrievalResult((0.7 * float(semantic_score)) + (0.3 * keyword_score), chunk))
        results.sort(key=lambda result: result.score, reverse=True)
        if not results or not dependency_expansion:
            return results[:4]

        selected = [results[0]]
        references = called_functions(results[0].chunk.text)
        for result in results:
            if result.score >= 0.60 and result.chunk not in [item.chunk for item in selected]:
                selected.append(result)
            if any(defines_function(result.chunk.text, name) for name in references):
                if result.chunk not in [item.chunk for item in selected]:
                    selected.append(result)
        return selected
