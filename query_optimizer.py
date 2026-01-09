#!/usr/bin/env python3
"""
AI Query Optimizer - CLI tool for generating optimized query variants.

Uses Grok API for variant generation and sentence-transformers + FAISS for evaluation.
"""

import os
import sys
import json
import argparse
from typing import List, Dict, Tuple
import requests
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss
from dotenv import load_dotenv
import config

load_dotenv()


class GrokAPIClient:
    """Client for interacting with Grok API."""
    
    def __init__(self, api_key: str = None, api_url: str = None):
        """
        Initialize the Grok API client.

        Args:
            api_key: Grok API key. If not provided, reads from GROK_API_KEY env var.
            api_url: API endpoint URL. If not provided, uses config default.
        """
        self.api_key = api_key or os.getenv("GROK_API_KEY")
        self.api_url = api_url or config.GROK_API_CONFIG["api_url"]
        self.model = config.GROK_API_CONFIG["model"]
        
        if not self.api_key:
            raise ValueError(
                "GROK_API_KEY not set. "
                "Set the environment variable or pass api_key parameter."
            )
    
    def generate_query_variants(self, query: str, num_variants: int = None) -> List[str]:
        """
        Generate optimized query variants via Grok API.

        Args:
            query: Original user query.
            num_variants: Number of variants to generate. Defaults to config value.

        Returns:
            List of query variant strings.

        Raises:
            ValueError: If API call fails or response parsing fails.
        """
        if num_variants is None:
            num_variants = config.OPTIMIZATION_CONFIG["num_variants"]
        
        prompt = config.QUERY_VARIANT_PROMPT_TEMPLATE.format(
            num_variants=num_variants,
            query=query
        )

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "messages": [
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "model": self.model,
            "temperature": config.GROK_API_CONFIG["temperature"],
            "max_tokens": config.GROK_API_CONFIG["max_tokens"]
        }
        
        # Add response_format for json_mode if supported by API
        if "response_format" in config.GROK_API_CONFIG:
            payload["response_format"] = config.GROK_API_CONFIG["response_format"]
        
        try:
            response = requests.post(
                self.api_url,
                headers=headers,
                json=payload,
                timeout=30
            )
            response.raise_for_status()
            
            result = response.json()
            content = result["choices"][0]["message"]["content"].strip()
            
            # Parse JSON response
            # With json_mode (response_format), API guarantees valid JSON
            try:
                # Try to parse directly (json_mode should return clean JSON)
                json_str = content
                
                # Fallback: if API returned markdown code block (in case json_mode unsupported)
                if "```json" in content:
                    start = content.find("```json") + 7
                    end = content.find("```", start)
                    if end != -1:
                        json_str = content[start:end].strip()
                elif "```" in content and content.startswith("```"):
                    start = content.find("```") + 3
                    end = content.find("```", start)
                    if end != -1:
                        json_str = content[start:end].strip()
                
                parsed = json.loads(json_str)
                variants = parsed.get("variants", [])
                
                # Validate: check that we got a list of strings
                if not isinstance(variants, list):
                    raise ValueError("JSON must contain 'variants' array")
                
                # Filter empty strings and trim to required count
                variants = [v.strip() for v in variants if v and isinstance(v, str) and v.strip()]
                
                if not variants:
                    raise ValueError("No valid variants received")
                
                return variants[:num_variants] if variants else [query]
                
            except json.JSONDecodeError as e:
                error_msg = f"Error parsing JSON response from API: {e}"
                error_msg += f"\nReceived content: {content[:500]}"
                error_msg += "\n\nEnsure the API supports response_format or check the prompt."
                raise ValueError(error_msg) from e
            except (KeyError, ValueError) as e:
                error_msg = f"Error processing JSON response: {e}"
                error_msg += f"\nReceived content: {content[:500]}"
                raise ValueError(error_msg) from e
            
        except requests.exceptions.HTTPError as e:
            error_msg = f"Error calling Grok API: {e}"
            
            if e.response is not None:
                try:
                    error_data = e.response.json()
                    if "error" in error_data:
                        error_msg += f"\nError details: {error_data['error']}"
                    elif "message" in error_data:
                        error_msg += f"\nError details: {error_data['message']}"
                    
                    if e.response.status_code == 401 or e.response.status_code == 400:
                        if "API key" in str(error_data).lower() or "api key" in str(error_data).lower():
                            error_msg += "\n\nCheck the API key in GROK_API_KEY environment variable"
                            error_msg += "\nor pass it via --api-key parameter"
                            error_msg += "\nGet your API key at: https://console.x.ai"
                except (ValueError, KeyError):
                    error_msg += f"\nAPI response: {e.response.text[:200]}"
            
            raise ValueError(error_msg) from e
            
        except requests.exceptions.RequestException as e:
            error_msg = f"Network error calling Grok API: {e}"
            if hasattr(e, 'response') and e.response is not None:
                try:
                    error_data = e.response.json()
                    error_msg += f"\nAPI response: {error_data}"
                except (ValueError, AttributeError):
                    error_msg += f"\nAPI response: {e.response.text[:200]}"
            raise ValueError(error_msg) from e
            
        except (KeyError, IndexError) as e:
            error_msg = f"Error parsing API response: {e}"
            if 'result' in locals():
                error_msg += f"\nUnexpected response format: {json.dumps(result, ensure_ascii=False, indent=2)[:500]}"
            raise ValueError(error_msg) from e


class SimilarityChecker:
    """Computes similarity and diversity scores for query variants."""
    
    def __init__(self, model_name: str = None):
        """
        Initialize the embedding model.

        Args:
            model_name: Name of sentence-transformers model. Defaults to config value.
        """
        if model_name is None:
            model_name = config.EMBEDDING_MODEL_CONFIG["model_name"]
        print("Loading embedding model...", file=sys.stderr)
        self.model = SentenceTransformer(model_name)
        print("Model loaded.", file=sys.stderr)
    
    def compute_similarity(self, query: str, variants: List[str]) -> List[float]:
        """
        Compute similarity scores between original query and variants.

        Args:
            query: Original query.
            variants: List of query variants.

        Returns:
            List of similarity scores (0-1 range).
        """
        texts = [query] + variants
        embeddings = self.model.encode(texts, convert_to_numpy=True)
        
        query_embedding = embeddings[0]
        variant_embeddings = embeddings[1:]
        
        similarities = []
        for variant_emb in variant_embeddings:
            similarity = np.dot(query_embedding, variant_emb) / (
                np.linalg.norm(query_embedding) * np.linalg.norm(variant_emb)
            )
            similarities.append(float(similarity))
        
        return similarities
    
    def compute_diversity(self, variants: List[str]) -> float:
        """
        Compute diversity score among variants.

        Args:
            variants: List of query variants.

        Returns:
            Diversity score (0-1, where 1 = maximum diversity).
        """
        if len(variants) < 2:
            return 0.0
        
        embeddings = self.model.encode(variants, convert_to_numpy=True)
        
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        normalized_embeddings = embeddings / (norms + 1e-8)
        
        similarity_matrix = np.dot(normalized_embeddings, normalized_embeddings.T)
        
        n = len(embeddings)
        mask = np.triu(np.ones((n, n)), k=1).astype(bool)
        pairwise_similarities = similarity_matrix[mask]
        
        avg_similarity = np.mean(pairwise_similarities) if len(pairwise_similarities) > 0 else 0.0
        diversity = 1.0 - avg_similarity
        
        return float(diversity)


class MockRetrieval:
    """Mock vector database for testing retrieval accuracy."""
    
    def __init__(self, embedding_model):
        """
        Initialize the mock database.

        Args:
            embedding_model: Sentence-transformers model for creating embeddings.
        """
        self.model = embedding_model
        self.index = None
        self.documents = []
        self._initialize_mock_docs()
    
    def _initialize_mock_docs(self):
        """Initialize mock documents for testing."""
        self.documents = config.MOCK_DOCUMENTS.copy()
        
        embeddings = self.model.encode(self.documents, convert_to_numpy=True)
        dimension = embeddings.shape[1]
        
        faiss.normalize_L2(embeddings)
        
        self.index = faiss.IndexFlatIP(dimension)
        self.index.add(embeddings.astype('float32'))
    
    def search(self, query: str, top_k: int = 3) -> List[Tuple[int, float]]:
        """
        Search for relevant documents.

        Args:
            query: Search query.
            top_k: Number of results to return.

        Returns:
            List of tuples (document_index, score).
        """
        query_embedding = self.model.encode([query], convert_to_numpy=True)
        faiss.normalize_L2(query_embedding)
        
        scores, indices = self.index.search(query_embedding.astype('float32'), top_k)
        
        results = [
            (int(idx), float(score))
            for idx, score in zip(indices[0], scores[0])
            if idx >= 0
        ]
        
        return results


class QueryOptimizer:
    """Main class for query optimization."""
    
    def __init__(self, grok_api_key: str = None):
        """
        Initialize Query Optimizer.

        Args:
            grok_api_key: Grok API key. Can also be set via environment variable.
        """
        self.grok_client = GrokAPIClient(api_key=grok_api_key)
        self.similarity_checker = SimilarityChecker()
        self.mock_retrieval = MockRetrieval(self.similarity_checker.model)
    
    def optimize(self, query: str) -> Dict:
        """
        Optimize query by generating variants and evaluating them.

        Args:
            query: Original user query.

        Returns:
            Dictionary with optimization results including variants, scores, and recommendations.
        """
        print(f"Generating variants for query: '{query}'...", file=sys.stderr)
        variants = self.grok_client.generate_query_variants(query)
        
        print("Computing similarity scores...", file=sys.stderr)
        similarities = self.similarity_checker.compute_similarity(query, variants)
        
        print("Computing diversity score...", file=sys.stderr)
        diversity = self.similarity_checker.compute_diversity(variants)
        
        print("Testing retrieval accuracy...", file=sys.stderr)
        retrieval_results = []
        top_k = config.OPTIMIZATION_CONFIG["retrieval_top_k"]
        for variant in variants:
            results = self.mock_retrieval.search(variant, top_k=top_k)
            avg_score = np.mean([score for _, score in results]) if results else 0.0
            retrieval_results.append({
                "variant": variant,
                "top_results": results,
                "avg_retrieval_score": avg_score
            })
        
        return {
            "original_query": query,
            "variants": [
                {
                    "text": variant,
                    "similarity_score": sim_score
                }
                for variant, sim_score in zip(variants, similarities)
            ],
            "diversity_score": diversity,
            "average_similarity": np.mean(similarities),
            "retrieval_results": retrieval_results
        }


def format_output(results: Dict, json_output: bool = False):
    """
    Format and print optimization results.

    Args:
        results: Dictionary with optimization results.
        json_output: If True, output JSON format; otherwise human-readable text.
    """
    if json_output:
        print(json.dumps(results, ensure_ascii=False, indent=2))
        return
    
    separator_width = config.OUTPUT_CONFIG["separator_width"]
    precision = config.OUTPUT_CONFIG["similarity_precision"]
    
    print("\n" + "="*separator_width)
    print(" " * 20 + "AI Query Optimizer")
    print("="*separator_width + "\n")
    
    print(f"Original query: {results['original_query']}\n")
    print("Generated variants:")
    print("-" * separator_width)
    
    for i, variant_data in enumerate(results['variants'], 1):
        print(f"\nVariant {i}: {variant_data['text']}")
        print(f"Similarity Score: {variant_data['similarity_score']:.{precision}f}")
        
        retrieval = results['retrieval_results'][i-1]
        print(f"Retrieval Score: {retrieval['avg_retrieval_score']:.{precision}f}")
        print("-" * separator_width)
    
    print(f"\nSummary:")
    print(f"  • Average Similarity: {results['average_similarity']:.{precision}f}")
    print(f"  • Diversity Score: {results['diversity_score']:.{precision}f}")
    print(f"  • Recommendation: ", end="")
    
    thresholds = config.RECOMMENDATION_THRESHOLDS
    messages = config.RECOMMENDATION_MESSAGES
    
    if results['diversity_score'] > thresholds["high_diversity"] and results['average_similarity'] > thresholds["high_similarity"]:
        print(messages["ensemble"])
    elif results['average_similarity'] > thresholds["excellent_similarity"]:
        print(messages["good"])
    else:
        print(messages["selective"])
    
    print("\n" + "="*separator_width + "\n")


def main():
    """Main entry point for the CLI application."""
    parser = argparse.ArgumentParser(
        description="AI Query Optimizer - generate optimized query variants"
    )
    parser.add_argument(
        "query",
        nargs="?",
        help="User query to optimize"
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output in JSON format"
    )
    parser.add_argument(
        "--api-key",
        help="Grok API key (or use GROK_API_KEY environment variable)"
    )
    
    args = parser.parse_args()
    
    if args.query:
        query = args.query
    else:
        print("Enter query (or Ctrl+C to exit):")
        query = input().strip()
        if not query:
            print("Query cannot be empty", file=sys.stderr)
            sys.exit(1)
    
    try:
        optimizer = QueryOptimizer(grok_api_key=args.api_key)
        results = optimizer.optimize(query)
        format_output(results, json_output=args.json)
        
    except ValueError as e:
        print(f"\n❌ {e}", file=sys.stderr)
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n\nOperation cancelled by user", file=sys.stderr)
        sys.exit(130)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
