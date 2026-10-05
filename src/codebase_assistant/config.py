"""Project configuration."""

SUPPORTED_EXTENSIONS = {
    ".py": "Python", ".js": "JavaScript", ".jsx": "JavaScript",
    ".ts": "TypeScript", ".tsx": "TypeScript", ".java": "Java",
    ".go": "Go", ".rs": "Rust", ".sql": "SQL", ".md": "Markdown",
    ".json": "JSON", ".yaml": "YAML", ".yml": "YAML", ".toml": "TOML",
}
IGNORED_DIRECTORIES = {
    ".git", ".venv", "venv", "node_modules", "__pycache__",
    "dist", "build", "target", ".next", "coverage",
}
STOP_WORDS = {
    "a", "an", "and", "are", "at", "be", "but", "by", "can", "do",
    "for", "from", "how", "i", "in", "is", "it", "of", "on", "or",
    "the", "this", "to", "us", "what", "where", "which", "with",
}
CHUNK_LINES = 120
MAX_FILE_BYTES = 500_000
DEFAULT_MODEL = "llama3:latest"
DEFAULT_OLLAMA_URL = "http://localhost:11434/api/generate"
