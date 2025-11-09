# AI Query Optimizer

Автоматическая генерация 3 вариантов запроса для улучшения retrieval accuracy в RAG-системах (multi-query technique, +20% recall).

## Установка

```bash
pip install -r requirements.txt
```

### Настройка

Создайте файл `.env` в корне проекта со следующими параметрами:

```
GROK_API_KEY=your_api_key_here
GROK_API_URL=https://api.x.ai/v1/chat/completions
GROK_MODEL=grok-4-fast-non-reasoning
```

> **Примечание:** `GROK_API_URL` и `GROK_MODEL` имеют значения по умолчанию, но их можно переопределить в `.env` файле.

## Использование

```bash
# Базовое использование
python query_optimizer.py "Key risks in climate reports?"

# Вывод в формате JSON
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
- numpy
- python-dotenv

