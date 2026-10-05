"""Small standard-library client for Ollama's local generate API."""

import json
from urllib.request import Request, urlopen

from .config import DEFAULT_MODEL, DEFAULT_OLLAMA_URL


class OllamaError(RuntimeError):
    pass


class OllamaClient:
    def __init__(self, model: str = DEFAULT_MODEL, url: str = DEFAULT_OLLAMA_URL) -> None:
        self.model, self.url = model, url

    def generate(self, prompt: str) -> dict:
        body = json.dumps({
            "model": self.model, "prompt": prompt, "format": "json",
            "stream": False, "options": {"temperature": 0.1},
        }).encode("utf-8")
        request = Request(self.url, data=body, headers={"Content-Type": "application/json"}, method="POST")
        try:
            with urlopen(request, timeout=180) as response:
                payload = json.loads(response.read().decode("utf-8"))
            return json.loads(payload["response"])
        except Exception as error:
            raise OllamaError(f"Could not get a valid response from Ollama model '{self.model}'.") from error
