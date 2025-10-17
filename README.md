Project with examples of reasoning RAG and agent using FastAPI, Pydantic AI, and ChromaDB.

## Project Structure

```
├── main_example.py          # Main FastAPI app with RAG agent and tools
├── rag_examples.ipynb       # Jupyter notebook examples
├── .env                     # Environment variables (create this)
├── .env.example             # Environment template
├── pyproject.toml           # Project configuration and dependencies
├── uv.lock                  # Dependency lock file
├── requirements.txt         # Pip-compatible dependencies
└── README.md                # This file
```

## Prerequisites

- Python 3.12+
- [uv](https://github.com/astral-sh/uv) package manager

## Setup

### 1. Clone and Navigate

```bash
git clone https://github.com/syv-ai/it-center-fyn
```

### 2. Install Dependencies

```bash
uv sync
```

### 3. Configure Environment

Create a `.env` file in the project root:

```bash
cp .env.example .env
```

Edit `.env` and add your API key:

```env
SYVAI_API_KEY=your_api_key_here
```

## Running the Application

### FastAPI Server

Start the FastAPI server:

```bash
uv run uvicorn main_example:app --reload --port 8000
```

The API will be available at `http://localhost:8000`

### Jupyter Notebook

Install development dependencies (includes ipykernel):

```bash
uv sync --dev
```

Then open the notebook:

```bash
jupyter notebook rag_examples.ipynb
```

Or use VS Code / Cursor with Jupyter extension.

## API Documentation

Once running, visit:
- **Swagger UI**: http://localhost:8000/docs

## API Usage

### Send a Chat Request

```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "syvai/danskgpt-v2.1",
    "messages": [
      {"role": "user", "content": "Hvad er FastAPI?"}
    ],
    "session_id": 1
  }'
```

### Example with Python

```python
import requests

response = requests.post(
    "http://localhost:8000/v1/chat/completions",
    json={
        "model": "syvai/danskgpt-v2.1",
        "messages": [
            {"role": "user", "content": "Hvad er FastAPI?"}
        ],
        "session_id": 1
    }
)

print(response.json())
```

## Troubleshooting

**Issue**: `ValidationError` for API key
- **Solution**: Ensure `.env` file exists with `SYVAI_API_KEY=your_key`

**Issue**: Model not using reflect tool
- **Solution**: This is a model limitation. The infrastructure is in place but depends on model behavior.

**Issue**: Port already in use
- **Solution**: Change port: `uvicorn main_example:app --port 8001`

## Development

Run in debug mode:

```bash
uv run uvicorn main_example:app --reload --log-level debug
```
