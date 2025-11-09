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

## Описание

AI Query Optimizer реализует технику **multi-query retrieval** для улучшения recall в RAG-системах. Генерирует 3 семантически разнообразных варианта запроса через Grok API, оценивает их через similarity/diversity scores (sentence-transformers) и тестирует на mock векторной базе (FAISS).

Подробное описание проблемы, решения, архитектуры и wireframes см. в [PRD.md](PRD.md).

## Архитектура

- **GrokAPIClient** - генерация вариантов через Grok API с поддержкой json_mode
- **SimilarityChecker** - вычисление similarity/diversity scores через косинусное сходство embeddings (модель `paraphrase-multilingual-MiniLM-L12-v2`)
- **MockRetrieval** - тестирование retrieval accuracy на FAISS индексе с 10 предзагруженными документами
- **QueryOptimizer** - координация компонентов и формирование рекомендаций

## Использованные инструменты

### Из ТЗ

- **Grok API** - в соответствии с ТЗ.
- **sentence-transformers** - в соответствии с ТЗ. 
- **FAISS** - в соответствии с ТЗ.
- **requests** - в соответствии с ТЗ.

### Дополнительные

- **numpy** - критическая зависимость для sentence-transformers и FAISS. Модель `paraphrase-multilingual-MiniLM-L12-v2` создает embeddings размерностью 384 (каждый текст представлен вектором из 384 чисел). Для вычисления similarity scores требуется косинусное сходство между векторами: при обработке 4 текстов (1 оригинальный запрос + 3 варианта) выполняется ~1500 операций умножения и сложения над массивами. NumPy использует оптимизированные C-библиотеки (BLAS/LAPACK) и векторизацию CPU, что дает ускорение в 100-1000 раз по сравнению с встроенными списками Python. Без numpy вычисление similarity для одного запроса заняло бы секунды вместо миллисекунд, что неприемлемо для интерактивного использования. Альтернативы: встроенные списки Python (катастрофическое падение производительности).

- **python-dotenv** - безопасное хранение API-ключей вне кода, стандарт для конфигурации Python-проектов. Альтернатива: `os.getenv()` (требует ручной настройки в каждой среде).

## Зависимости

- **requests** (>=2.31.0) - HTTP-клиент для работы с Grok API
- **sentence-transformers** (>=2.2.2) - создание векторных представлений текста
- **faiss-cpu** (>=1.7.4) - векторный поиск для mock retrieval
- **numpy** (>=1.24.0) - численные вычисления для similarity scores
- **python-dotenv** (>=1.0.0) - загрузка конфигурации из .env файла

