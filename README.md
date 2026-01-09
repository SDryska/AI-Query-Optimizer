# AI Query Optimizer

Automatic generation of 3 query variants to improve retrieval accuracy in RAG systems using multi-query technique (+20% recall).

## Table of Contents

- [Background](#background)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
- [Architecture](#architecture)
- [Dependencies](#dependencies)
- [License](#license)

## Background

RAG (Retrieval-Augmented Generation) systems often suffer from low search accuracy due to suboptimal query formulations. AI Query Optimizer addresses this by implementing the **multi-query retrieval** technique:

- Generates 3 semantically diverse query variants via Grok API
- Evaluates variants using similarity/diversity scores (sentence-transformers)
- Tests retrieval accuracy on a mock vector database (FAISS)
- Expected improvement: +20% recall (based on LangChain docs)

## Installation

### Prerequisites

- Python 3.8+
- Grok API key (obtain at [console.x.ai](https://console.x.ai))

### Steps

```bash
# Clone the repository
git clone https://github.com/sandr-flow/query-optimizer.git
cd query-optimizer

# Create virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Edit .env and add your GROK_API_KEY
```

## Configuration

Create a `.env` file in the project root (or copy from `.env.example`):

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `GROK_API_KEY` | Yes | — | Your Grok API key |
| `GROK_API_URL` | No | `https://api.x.ai/v1/chat/completions` | API endpoint URL |
| `GROK_MODEL` | No | `grok-4-fast-non-reasoning` | Model to use for generation |

## Usage

### Command Line

```bash
# Basic usage
python query_optimizer.py "Key risks in climate reports?"

# JSON output format
python query_optimizer.py "your query" --json

# Custom API key
python query_optimizer.py "your query" --api-key YOUR_API_KEY
```

### Interactive Mode

```bash
python query_optimizer.py
# Enter query (or Ctrl+C to exit):
# > Key risks in climate reports?
```

### Example Output

```
======================================================================
                    AI Query Optimizer
======================================================================

Original query: Key risks in climate reports?

Generated variants:
----------------------------------------------------------------------

Variant 1: What are the primary climate risk factors?
Similarity Score: 0.85
Retrieval Score: 0.72
----------------------------------------------------------------------

Variant 2: Climate change risks and vulnerabilities
Similarity Score: 0.82
Retrieval Score: 0.68
----------------------------------------------------------------------

Variant 3: Environmental hazards in climate documentation
Similarity Score: 0.79
Retrieval Score: 0.65
----------------------------------------------------------------------

Summary:
  • Average Similarity: 0.82
  • Diversity Score: 0.65
  • Recommendation: Use all variants for ensemble retrieval

======================================================================
```

## Architecture

```
┌─────────────────┐
│  User Query     │
│  (CLI input)    │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  GrokAPIClient  │  → Generates 3 query variants via Grok API (json_mode)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Similarity     │  → Computes similarity/diversity scores
│  Checker        │    (paraphrase-multilingual-MiniLM-L12-v2)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  MockRetrieval  │  → Tests retrieval on FAISS index (10 mock documents)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  QueryOptimizer │  → Coordinates components, formats recommendations
└─────────────────┘
```

### Key Components

| Component | Description |
|-----------|-------------|
| `GrokAPIClient` | Generates query variants via Grok API with json_mode support |
| `SimilarityChecker` | Computes similarity/diversity scores using cosine similarity on embeddings |
| `MockRetrieval` | Mock vector database for retrieval accuracy testing (FAISS index) |
| `QueryOptimizer` | Main orchestrator coordinating all components |

## Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| `requests` | ≥2.31.0 | HTTP client for Grok API |
| `sentence-transformers` | ≥2.2.2 | Text embeddings generation |
| `faiss-cpu` | ≥1.7.4 | Vector similarity search |
| `numpy` | ≥1.24.0 | Numerical computations for similarity scores |
| `python-dotenv` | ≥1.0.0 | Environment configuration loading |

## License

MIT License - see [LICENSE](LICENSE) file for details.
