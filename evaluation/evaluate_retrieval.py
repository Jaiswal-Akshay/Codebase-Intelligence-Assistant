"""Lightweight retrieval evaluation without calling an LLM."""

from __future__ import annotations

import json
import sys
from pathlib import Path

from codebase_assistant.ingest import discover_chunks
from codebase_assistant.retrieval import HybridRetriever


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("Usage: python evaluation/evaluate_retrieval.py PATH_TO_REPOSITORY")

    repository = Path(sys.argv[1]).resolve()
    questions = json.loads(Path("evaluation/questions.json").read_text(encoding="utf-8"))
    chunks = discover_chunks(repository)
    retriever = HybridRetriever(chunks)

    file_hits = 0
    term_hits = 0
    for case in questions:
        results = retriever.search(case["question"])
        retrieved_files = {result.chunk.file for result in results}
        retrieved_text = " ".join(result.chunk.text.lower() for result in results)
        file_ok = all(file_name in retrieved_files for file_name in case["expected_files"])
        terms_ok = all(term.lower() in retrieved_text for term in case["expected_terms"])
        file_hits += int(file_ok)
        term_hits += int(terms_ok)
        print(f"{'PASS' if file_ok and terms_ok else 'REVIEW'}: {case['question']}")
        print(f"  Files: {sorted(retrieved_files)}")

    total = len(questions)
    print(f"\nFile retrieval accuracy: {file_hits}/{total} ({file_hits / total:.1%})")
    print(f"Evidence term accuracy: {term_hits}/{total} ({term_hits / total:.1%})")


if __name__ == "__main__":
    main()
