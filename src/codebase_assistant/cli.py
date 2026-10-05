"""Command-line interface for Codebase Compass."""

from __future__ import annotations

import argparse
from pathlib import Path

from .ollama import OllamaError
from .pipeline import CodebaseCompass


def main() -> None:
    parser = argparse.ArgumentParser(description="Ask questions about a codebase.")
    parser.add_argument("repository", type=Path)
    parser.add_argument("question", nargs="?")
    parser.add_argument("--model", default="llama3:latest")
    args = parser.parse_args()
    repository = args.repository.resolve()
    if not repository.is_dir():
        parser.error(f"Repository does not exist: {repository}")
    question = args.question or input("Ask a question about the project: ")
    try:
        answer, results, report = CodebaseCompass(repository, args.model).answer(question)
    except OllamaError as error:
        parser.error(str(error))
    print("\nRetrieved context:")
    for result in results:
        print(f"- {result.chunk.file}:{result.chunk.start_line}-{result.chunk.end_line} ({result.score:.3f})")
    print(f"\nValidation status: {report.status}\n\nAnswer:\n{answer.get('answer', '')}\n")
    print("Evidence:")
    for item in answer.get("evidence", []):
        print(f"- {item.get('file')} lines {item.get('lines')}: {item.get('claim')}")
    for label, items in (("Validation errors", report.errors), ("Validation warnings", report.warnings)):
        if items:
            print(f"\n{label}:")
            for item in items:
                print(f"- {item}")


if __name__ == "__main__":
    main()
