# history-bot

This is a history bot developed using python, ollama model and langchain. This uses external vector store(db-knowledge base) using chroma db for historical figures.pdf data.
To run this project, firstly initialize the virtual environment using uv.

Install uv with our standalone installers:

# On macOS and Linux.

curl -LsSf https://astral.sh/uv/install.sh | sh

# On Windows.

powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"

To start virtual environment, run .venv\Scripts\activate
To install dependencies, uv pip install -r requirements.txt
To run python file, un run history_chatbot.py
