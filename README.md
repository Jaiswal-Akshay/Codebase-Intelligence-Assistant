Codebase Compass
Codebase Compass is a local RAG assistant that analyzes software repositories and explains code using grounded file and line citations.

Installation
python -m pip install -r requirements.txt

Ollama Setup
Make sure Ollama is running and the model is installed:
ollama pull llama3:latest

Run
# Clone the GitHub repository into the repositories folder
git clone https://github.com/username/project-name.git .\repositories\project-name

# Tell Python where the Codebase Compass source code is located
$env:PYTHONPATH = "src"

# Start Codebase Compass and analyze the cloned repository
python -m codebase_assistant .\repositories\project-name

You can also provide a question directly:
python -m codebase_assistant .\repositories\project-name "Your Question"