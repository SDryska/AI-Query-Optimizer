# PRD: AI Query Optimizer

## Problem Statement

**Проблема**: RAG-системы имеют низкую точность поиска из-за неоптимальных формулировок запросов.

**Решение**: Автоматическая генерация 3 семантически разнообразных вариантов запроса (multi-query technique). Ожидаемый эффект: +20% recall (LangChain docs).

## User Stories

- **US-1**: Генерация 3 вариантов запроса через Grok API
- **US-2**: Вычисление similarity/diversity scores (sentence-transformers)
- **US-3**: Mock retrieval для оценки accuracy (FAISS)
- **US-4**: CLI интерфейс с JSON выводом

## Технические требования

**Вход**: Пользовательский запрос (до 500 символов)

**Выход**: 3 варианта запроса + similarity scores + diversity score

**Зависимости**: requests, sentence-transformers, faiss-cpu

**Ограничения**: Только внешние API (Grok), таймаут 30 сек

## Метрики успеха

- Recall improvement: +15-25%
- Diversity score: > 0.6
- Similarity score: > 0.8
- Latency: < 5 сек

## Wireframes

### CLI Интерфейс

#### Командная строка
```
$ python query_optimizer.py "Key risks in climate reports?"
```

#### Интерактивный режим
```
$ python query_optimizer.py
Введите запрос (или Ctrl+C для выхода):
> Key risks in climate reports?
```

#### Текстовый вывод (human-readable)
```
====================================================
                    AI Query Optimizer
====================================================

Исходный запрос: Key risks in climate reports?

Сгенерированные варианты:
----------------------------------------------------

Вариант 1: What are the primary climate risk factors?
Similarity Score: 0.85
Retrieval Score: 0.72
----------------------------------------------------

Вариант 2: Climate change risks and vulnerabilities
Similarity Score: 0.82
Retrieval Score: 0.68
----------------------------------------------------

Вариант 3: Environmental hazards in climate documentation
Similarity Score: 0.79
Retrieval Score: 0.65
----------------------------------------------------

Сводка:
  • Средний Similarity: 0.82
  • Diversity Score: 0.65
  • Рекомендация: Используйте все варианты для ensemble retrieval

====================================================
```

#### JSON вывод
```json
{
  "original_query": "Key risks in climate reports?",
  "variants": [
    {
      "text": "What are the primary climate risk factors?",
      "similarity_score": 0.85
    },
    {
      "text": "Climate change risks and vulnerabilities",
      "similarity_score": 0.82
    },
    {
      "text": "Environmental hazards in climate documentation",
      "similarity_score": 0.79
    }
  ],
  "diversity_score": 0.65,
  "average_similarity": 0.82,
  "retrieval_results": [
    {
      "variant": "What are the primary climate risk factors?",
      "top_results": [[0, 0.85], [2, 0.72], [1, 0.68]],
      "avg_retrieval_score": 0.75
    },
    ...
  ]
}
```

#### Процесс работы (Flow)
```
┌─────────────────┐
│  User Query     │
│  (CLI input)    │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Grok API       │
│  Generate 3     │
│  Variants       │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Similarity     │
│  Checker        │
│  (embeddings)   │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Mock Retrieval │
│  (FAISS)        │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Format Output  │
│  (Text/JSON)    │
└─────────────────┘
```

