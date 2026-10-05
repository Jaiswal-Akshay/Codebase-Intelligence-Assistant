"""Shared data models for Codebase Compass."""

from dataclasses import dataclass


@dataclass(frozen=True)
class Chunk:
    file: str
    language: str
    start_line: int
    end_line: int
    text: str


@dataclass(frozen=True)
class RetrievalResult:
    score: float
    chunk: Chunk


@dataclass(frozen=True)
class ValidationReport:
    status: str
    errors: list[str]
    warnings: list[str]
