"""
Конфигурационный файл для AI Query Optimizer
Содержит промпты, моки документов и другие изменяемые параметры
"""

import os
from dotenv import load_dotenv

# Загрузка переменных окружения из .env файла
load_dotenv()

# ============================================================================
# ПРОМПТЫ
# ============================================================================

# Плейсхолдеры: {num_variants} - количество вариантов, {query} - исходный запрос
QUERY_VARIANT_PROMPT_TEMPLATE = """You are an expert in optimizing search queries for RAG systems.

Task: Create {num_variants} semantically diverse variants of the following query that 
preserve the original meaning but are formulated differently to improve 
retrieval accuracy in vector databases.

Original query: "{query}"

Requirements:
1. Each variant must preserve the core meaning of the original
2. Variants should be semantically diverse (different phrasings, synonyms, structures)
3. Variants should be optimized for vector database search
4. Output only the variants, one per line, without numbering or additional comments

Variants:"""


# ============================================================================
# МОКИ ДОКУМЕНТОВ
# ============================================================================


MOCK_DOCUMENTS = [
    "Climate change poses significant risks to human and natural systems. The IPCC reports identify key risks including sea level rise, extreme weather events, and biodiversity loss.",
    "One of the primary risks highlighted in climate assessment reports is the increased frequency and intensity of extreme weather events such as heatwaves, droughts, and heavy precipitation.",
    "Climate reports emphasize the critical risk of sea level rise, which threatens coastal communities, infrastructure, and ecosystems worldwide.",
    "Biodiversity loss represents a major risk identified in climate reports, with species extinction rates accelerating due to habitat destruction and changing climate conditions.",
    "Food security risks are prominent in climate assessment documents, as changing precipitation patterns and temperature extremes threaten agricultural productivity globally.",
    "Water scarcity is a key risk documented in IPCC reports, with many regions facing increased drought frequency and reduced water availability.",
    "Climate reports identify health risks from climate change, including heat-related illnesses, vector-borne diseases, and air quality deterioration.",
    "Economic risks from climate change are substantial, with climate reports estimating significant GDP losses and increased costs for adaptation and disaster recovery.",
    "Infrastructure risks in climate reports include damage to transportation systems, energy grids, and buildings from extreme weather and sea level rise.",
    "The IPCC assessment reports highlight cascading risks where climate impacts in one system can trigger failures in interconnected systems, amplifying overall vulnerability."
]


# ============================================================================
# ПАРАМЕТРЫ API
# ============================================================================

# Параметры для Grok API (значения из .env имеют приоритет)
GROK_API_CONFIG = {
    "model": os.getenv("GROK_MODEL", "grok-4-fast-non-reasoning"),
    "temperature": 0.7,    
    "max_tokens": 500,     
    "api_url": os.getenv("GROK_API_URL", "https://api.x.ai/v1/chat/completions")
}


# ============================================================================
# ПАРАМЕТРЫ МОДЕЛЕЙ
# ============================================================================

# Параметры для модели embeddings
EMBEDDING_MODEL_CONFIG = {
    "model_name": "paraphrase-multilingual-MiniLM-L12-v2",  # Модель sentence-transformers
}


# ============================================================================
# ПАРАМЕТРЫ ОПТИМИЗАЦИИ
# ============================================================================

# Параметры для процесса оптимизации
OPTIMIZATION_CONFIG = {
    "num_variants": 3,      # Количество вариантов запроса для генерации
    "retrieval_top_k": 3,  # Количество топ результатов для retrieval тестирования
}


# ============================================================================
# ПАРАМЕТРЫ ОЦЕНКИ И РЕКОМЕНДАЦИЙ
# ============================================================================

RECOMMENDATION_THRESHOLDS = {
    "high_diversity": 0.6,      
    "high_similarity": 0.8,    
    "excellent_similarity": 0.85  
}

# Сообщения рекомендаций
RECOMMENDATION_MESSAGES = {
    "ensemble": "Используйте все 3 варианта для ensemble retrieval",
    "good": "Варианты хорошо сохраняют смысл, можно использовать все",
    "selective": "Рассмотрите использование вариантов с высоким similarity score"
}


# ============================================================================
# ПАРАМЕТРЫ ВЫВОДА
# ============================================================================

# Параметры форматирования вывода
OUTPUT_CONFIG = {
    "separator_width": 70,  # Ширина разделителей в текстовом выводе
    "similarity_precision": 3,  # Количество знаков после запятой для similarity scores
}

