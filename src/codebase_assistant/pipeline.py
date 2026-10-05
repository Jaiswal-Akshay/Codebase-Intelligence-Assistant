"""End-to-end Codebase Compass pipeline."""

from pathlib import Path

from .ingest import discover_chunks
from .models import RetrievalResult
from .ollama import OllamaClient
from .retrieval import HybridRetriever
from .validation import validate_answer


def format_context(results: list[RetrievalResult]) -> str:
    parts = []
    for result in results:
        numbered = "\n".join(
            f"Line {line}: {text}"
            for line, text in enumerate(result.chunk.text.splitlines(), start=result.chunk.start_line)
        )
        parts.append(f"FILE: {result.chunk.file} ({result.chunk.language})\nCODE:\n{numbered}")
    return "\n\n---\n\n".join(parts)


def build_prompt(question: str, context: str) -> str:
    return f"""You are Codebase Compass, a strict software codebase assistant.
Return valid JSON with exactly these fields:
{{"answer":"...","evidence":[{{"file":"relative/path","lines":[1],"claim":"..."}}],"unsupported_claims":[]}}
Use only CONTEXT. Every factual claim needs evidence. Do not mention behavior not shown.

CONTEXT:
{context}

USER QUESTION:
{question}
"""


class CodebaseCompass:
    def __init__(self, repository: Path, model: str = "llama3:latest") -> None:
        self.repository = repository.resolve()
        self.model = model

    def answer(self, question: str):
        chunks = discover_chunks(self.repository)
        results = HybridRetriever(chunks).search(question)
        response = OllamaClient(model=self.model).generate(build_prompt(question, format_context(results)))
        report = validate_answer(response, [result.chunk for result in results])
        return response, results, report
