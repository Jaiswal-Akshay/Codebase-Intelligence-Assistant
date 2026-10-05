"""Repository discovery and line-aware chunking."""

from pathlib import Path

from .config import CHUNK_LINES, IGNORED_DIRECTORIES, MAX_FILE_BYTES, SUPPORTED_EXTENSIONS
from .models import Chunk


def discover_chunks(repository: Path) -> list[Chunk]:
    chunks: list[Chunk] = []
    repository = repository.resolve()
    for path in repository.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            continue
        if any(part in IGNORED_DIRECTORIES for part in path.parts):
            continue
        if path.stat().st_size > MAX_FILE_BYTES:
            continue
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
        except (OSError, UnicodeDecodeError):
            continue
        relative_path = path.relative_to(repository).as_posix()
        language = SUPPORTED_EXTENSIONS[path.suffix.lower()]
        for start in range(0, len(lines), CHUNK_LINES):
            end = min(start + CHUNK_LINES, len(lines))
            chunks.append(Chunk(relative_path, language, start + 1, end, "\n".join(lines[start:end])))
    return chunks
