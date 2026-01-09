"""
Configuration file for AI Query Optimizer.

Contains prompts, mock documents, and other configurable parameters.
"""

import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# ============================================================================
# PROMPTS
# ============================================================================

# Placeholders: {num_variants} - number of variants, {query} - original query
# Note: When using json_mode (response_format), prompt must explicitly request JSON object
QUERY_VARIANT_PROMPT_TEMPLATE = """You are an expert in optimizing search queries for RAG systems.

Task: Create {num_variants} semantically diverse variants of the following query that 
preserve the original meaning but are formulated differently to improve 
retrieval accuracy in vector databases.

Original query: "{query}"

Requirements:
1. Each variant must preserve the core meaning of the original
2. Variants should be semantically diverse (different phrasings, synonyms, structures)
3. Variants should be optimized for vector database search

You must respond with a valid JSON object containing a "variants" array with exactly {num_variants} query strings.

JSON format:
{{
  "variants": [
    "variant 1 text",
    "variant 2 text",
    "variant 3 text"
  ]
}}"""


# ============================================================================
# MOCK DOCUMENTS
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
# API PARAMETERS
# ============================================================================

# Grok API parameters (values from .env take priority)
GROK_API_CONFIG = {
    "model": os.getenv("GROK_MODEL", "grok-4-fast-non-reasoning"),
    "temperature": 0.7,    
    "max_tokens": 500,     
    "api_url": os.getenv("GROK_API_URL", "https://api.x.ai/v1/chat/completions"),
    "response_format": {"type": "json_object"}  # Enables json_mode for guaranteed JSON output
}


# ============================================================================
# MODEL PARAMETERS
# ============================================================================

# Embedding model parameters
EMBEDDING_MODEL_CONFIG = {
    "model_name": "paraphrase-multilingual-MiniLM-L12-v2",  # sentence-transformers model
}


# ============================================================================
# OPTIMIZATION PARAMETERS
# ============================================================================

# Parameters for the optimization process
OPTIMIZATION_CONFIG = {
    "num_variants": 3,      # Number of query variants to generate
    "retrieval_top_k": 3,   # Number of top results for retrieval testing
}


# ============================================================================
# EVALUATION AND RECOMMENDATION PARAMETERS
# ============================================================================

RECOMMENDATION_THRESHOLDS = {
    "high_diversity": 0.6,      
    "high_similarity": 0.8,    
    "excellent_similarity": 0.85  
}

# Recommendation messages
RECOMMENDATION_MESSAGES = {
    "ensemble": "Use all 3 variants for ensemble retrieval",
    "good": "Variants preserve meaning well, all can be used",
    "selective": "Consider using only variants with high similarity score"
}


# ============================================================================
# OUTPUT PARAMETERS
# ============================================================================

# Output formatting parameters
OUTPUT_CONFIG = {
    "separator_width": 70,       # Width of separators in text output
    "similarity_precision": 3,   # Decimal places for similarity scores
}
