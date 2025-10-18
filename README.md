# scripture-rag

Run RAG queries on the standard works using semantic search and LLM-powered question answering.

## Installation

This project uses [uv](https://docs.astral.sh/uv/) for dependency management. First, install uv if you haven't already:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Then install the project dependencies:

```bash
uv sync
```

## Usage

### Indexing Scripture Files

Before running queries, you need to index the scripture files into the vector database:

```bash
uv run scripture-rag index
```

Options:

- `--assets-dir PATH`: Specify a custom path to the assets directory (auto-detected by default)
- `--append`: Append to existing index instead of clearing it

### Querying

Run semantic search queries or ask questions about the scriptures:

```bash
uv run scripture-rag query "faith and works"
```

By default, this will:

- Use a cross-encoder reranker for improved accuracy
- Generate an LLM-powered answer using relevant passages
- Display the top 5 most relevant scripture passages

Options:

- `--top-k N`: Number of results to return (default: 5)
- `--no-answer`: Skip LLM answer generation, only show search results
- `--book BOOK`: Filter by book name (can be used multiple times, e.g., `--book Alma --book Moroni`)
- `--no-reranker`: Disable cross-encoder reranking (faster but less accurate)
- `--retrieval-factor N`: Multiplier for initial retrieval when reranking (default: 3.0)

### Examples

```bash
# Basic query with LLM answer and reranking
uv run scripture-rag query "What does the scripture say about faith?"

# Search only in specific books
uv run scripture-rag query "faith and works" --book James --book Alma

# Get more results without LLM answer
uv run scripture-rag query "prayer" --top-k 10 --no-answer

# Faster search without reranking
uv run scripture-rag query "temple" --no-reranker
```

## Development

### Running Tests

Run the full test suite with pytest:

```bash
uv run pytest
```

Run tests with coverage report:

```bash
uv run pytest --cov=src/scripture_rag --cov-report=term-missing
```

Run specific test files:

```bash
uv run pytest tests/test_parser.py
uv run pytest tests/test_query.py
```

### Code Formatting and Linting

Check code with ruff:

```bash
uv run ruff check .
```

Auto-fix issues:

```bash
uv run ruff check --fix .
```

Format code:

```bash
uv run ruff format .
```

### Project Structure

```
scripture-rag/
├── src/scripture_rag/
│   ├── cli.py              # Command-line interface
│   ├── indexer.py          # Scripture indexing logic
│   ├── query.py            # Query engine
│   ├── parser.py           # Scripture text parsing
│   ├── vector_store.py     # ChromaDB vector store interface
│   ├── reranker.py         # Cross-encoder reranking
│   ├── book_mapping.py     # Book name mappings
│   └── downloader.py       # Download scripture texts
├── assets/                 # Scripture text files
├── tests/                  # Test files
└── pyproject.toml          # Project configuration
```
