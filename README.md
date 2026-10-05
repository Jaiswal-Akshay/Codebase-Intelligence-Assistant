Codebase Compass
Codebase Compass is a local RAG assistant that analyzes software repositories and explains code using grounded file and line citations.
Requirements
- Python 3.10+
- Ollama
- Ollama model: llama3:latest
Installation
python -m pip install -r requirements.txt
Ollama Setup
Make sure Ollama is running and the model is installed:
ollama pull llama3:latest
Run
From the project root:
$env:PYTHONPATH = "src"
python -m codebase_assistant .\sample_project
You can also provide a question directly:
python -m codebase_assistant .\sample_project "How are user credentials verified?"
Evaluation
$env:PYTHONPATH = "src"
python evaluation\evaluate_retrieval.py .\sample_project
Main Features
- Multi-format repository ingestion
- Semantic and keyword retrieval
- Dependency-aware context expansion
- Local Ollama generation
- Citation and unsupported-claim validation