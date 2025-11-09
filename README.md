# AI Query Optimizer

Автоматическая генерация 3 вариантов запроса для улучшения retrieval accuracy в RAG-системах (multi-query technique, +20% recall).

## Установка

```bash
pip install -r requirements.txt
export GROK_API_KEY="your_api_key_here"  # или через --api-key
```

## Использование

```bash
python query_optimizer.py "Key risks in climate reports?"
python query_optimizer.py "запрос" --json
```

## Архитектура

- **GrokAPIClient** - генерация вариантов через Grok API
- **SimilarityChecker** - similarity/diversity scores (sentence-transformers)
- **MockRetrieval** - тестирование accuracy (FAISS)
- **QueryOptimizer** - координация компонентов

## Зависимости

- requests
- sentence-transformers
- faiss-cpu

