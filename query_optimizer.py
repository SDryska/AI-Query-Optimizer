#!/usr/bin/env python3
"""
AI Query Optimizer - CLI инструмент для генерации оптимизированных вариантов запросов
Использует Grok API для генерации вариантов и sentence-transformers + FAISS для оценки
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
    """Клиент для работы с Grok API"""
    
    def __init__(self, api_key: str = None, api_url: str = None):
        self.api_key = api_key or os.getenv("GROK_API_KEY")
        self.api_url = api_url or config.GROK_API_CONFIG["api_url"]
        self.model = config.GROK_API_CONFIG["model"]
        
        if not self.api_key:
            raise ValueError(
                "GROK_API_KEY не установлен. "
                "Установите переменную окружения или передайте api_key"
            )
    
    def generate_query_variants(self, query: str, num_variants: int = None) -> List[str]:
        """
        Генерирует варианты оптимизированного запроса через Grok API
        
        Args:
            query: Исходный пользовательский запрос
            num_variants: Количество вариантов для генерации (по умолчанию из конфига)
        
        Returns:
            Список вариантов запроса
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
            
            variants = [
                line.strip() 
                for line in content.split("\n") 
                if line.strip() and not line.strip().startswith("#")
            ]
            
            cleaned_variants = []
            for variant in variants:
                variant = variant.lstrip("0123456789.-) ").lstrip("Variant ").lstrip(": ")
                if variant:
                    cleaned_variants.append(variant)
            
            return cleaned_variants[:num_variants] if cleaned_variants else [query]
            
        except requests.exceptions.HTTPError as e:
            error_msg = f"Ошибка при вызове Grok API: {e}"
            
            if e.response is not None:
                try:
                    error_data = e.response.json()
                    if "error" in error_data:
                        error_msg += f"\nДетали ошибки: {error_data['error']}"
                    elif "message" in error_data:
                        error_msg += f"\nДетали ошибки: {error_data['message']}"
                    
                    if e.response.status_code == 401 or e.response.status_code == 400:
                        if "API key" in str(error_data).lower() or "api key" in str(error_data).lower():
                            error_msg += "\n\nПроверьте правильность API ключа в переменной окружения GROK_API_KEY"
                            error_msg += "\nили передайте ключ через параметр --api-key"
                            error_msg += "\nПолучить API ключ можно на: https://console.x.ai"
                except (ValueError, KeyError):
                    error_msg += f"\nОтвет API: {e.response.text[:200]}"
            
            raise ValueError(error_msg) from e
            
        except requests.exceptions.RequestException as e:
            error_msg = f"Ошибка сети при вызове Grok API: {e}"
            if hasattr(e, 'response') and e.response is not None:
                try:
                    error_data = e.response.json()
                    error_msg += f"\nОтвет API: {error_data}"
                except (ValueError, AttributeError):
                    error_msg += f"\nОтвет API: {e.response.text[:200]}"
            raise ValueError(error_msg) from e
            
        except (KeyError, IndexError) as e:
            error_msg = f"Ошибка при парсинге ответа API: {e}"
            if 'result' in locals():
                error_msg += f"\nНеожиданный формат ответа: {json.dumps(result, ensure_ascii=False, indent=2)[:500]}"
            raise ValueError(error_msg) from e


class SimilarityChecker:
    """Класс для вычисления similarity scores"""
    
    def __init__(self, model_name: str = None):
        """
        Инициализация модели для embeddings
        
        Args:
            model_name: Название модели sentence-transformers (опционально, по умолчанию из конфига)
        """
        if model_name is None:
            model_name = config.EMBEDDING_MODEL_CONFIG["model_name"]
        print("Загрузка модели для embeddings...", file=sys.stderr)
        self.model = SentenceTransformer(model_name)
        print("Модель загружена.", file=sys.stderr)
    
    def compute_similarity(self, query: str, variants: List[str]) -> List[float]:
        """
        Вычисляет similarity scores между оригинальным запросом и вариантами
        
        Args:
            query: Исходный запрос
            variants: Список вариантов
        
        Returns:
            Список similarity scores (0-1)
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
        Вычисляет diversity score между вариантами
        
        Args:
            variants: Список вариантов
        
        Returns:
            Diversity score (0-1, где 1 = максимальное разнообразие)
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
    """Mock векторная база для тестирования retrieval accuracy"""
    
    def __init__(self, embedding_model):
        """
        Инициализация mock базы данных
        
        Args:
            embedding_model: Модель для создания embeddings
        """
        self.model = embedding_model
        self.index = None
        self.documents = []
        self._initialize_mock_docs()
    
    def _initialize_mock_docs(self):
        """Инициализация mock документов для тестирования"""
        self.documents = config.MOCK_DOCUMENTS.copy()
        
        embeddings = self.model.encode(self.documents, convert_to_numpy=True)
        dimension = embeddings.shape[1]
        
        faiss.normalize_L2(embeddings)
        
        self.index = faiss.IndexFlatIP(dimension)
        self.index.add(embeddings.astype('float32'))
    
    def search(self, query: str, top_k: int = 3) -> List[Tuple[int, float]]:
        """
        Поиск релевантных документов
        
        Args:
            query: Поисковый запрос
            top_k: Количество результатов
        
        Returns:
            Список кортежей (индекс документа, score)
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
    """Основной класс для оптимизации запросов"""
    
    def __init__(self, grok_api_key: str = None):
        """
        Инициализация Query Optimizer
        
        Args:
            grok_api_key: API ключ для Grok (опционально, можно через env)
        """
        self.grok_client = GrokAPIClient(api_key=grok_api_key)
        self.similarity_checker = SimilarityChecker()
        self.mock_retrieval = MockRetrieval(self.similarity_checker.model)
    
    def optimize(self, query: str) -> Dict:
        """
        Оптимизирует запрос, генерируя варианты и оценивая их
        
        Args:
            query: Исходный пользовательский запрос
        
        Returns:
            Словарь с результатами оптимизации
        """
        print(f"Генерация вариантов для запроса: '{query}'...", file=sys.stderr)
        variants = self.grok_client.generate_query_variants(query)
        
        print("Вычисление similarity scores...", file=sys.stderr)
        similarities = self.similarity_checker.compute_similarity(query, variants)
        
        print("Вычисление diversity score...", file=sys.stderr)
        diversity = self.similarity_checker.compute_diversity(variants)
        
        print("Тестирование retrieval accuracy...", file=sys.stderr)
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
    """Форматирует вывод результатов"""
    if json_output:
        print(json.dumps(results, ensure_ascii=False, indent=2))
        return
    
    separator_width = config.OUTPUT_CONFIG["separator_width"]
    precision = config.OUTPUT_CONFIG["similarity_precision"]
    
    print("\n" + "="*separator_width)
    print(" " * 20 + "AI Query Optimizer")
    print("="*separator_width + "\n")
    
    print(f"Исходный запрос: {results['original_query']}\n")
    print("Сгенерированные варианты:")
    print("-" * separator_width)
    
    for i, variant_data in enumerate(results['variants'], 1):
        print(f"\nВариант {i}: {variant_data['text']}")
        print(f"Similarity Score: {variant_data['similarity_score']:.{precision}f}")
        
        retrieval = results['retrieval_results'][i-1]
        print(f"Retrieval Score: {retrieval['avg_retrieval_score']:.{precision}f}")
        print("-" * separator_width)
    
    print(f"\nСводка:")
    print(f"  • Средний Similarity: {results['average_similarity']:.{precision}f}")
    print(f"  • Diversity Score: {results['diversity_score']:.{precision}f}")
    print(f"  • Рекомендация: ", end="")
    
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
    parser = argparse.ArgumentParser(
        description="AI Query Optimizer - генерация оптимизированных вариантов запросов"
    )
    parser.add_argument(
        "query",
        nargs="?",
        help="Пользовательский запрос для оптимизации"
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Вывод в формате JSON"
    )
    parser.add_argument(
        "--api-key",
        help="Grok API ключ (или используйте переменную окружения GROK_API_KEY)"
    )
    
    args = parser.parse_args()
    
    if args.query:
        query = args.query
    else:
        print("Введите запрос (или Ctrl+C для выхода):")
        query = input().strip()
        if not query:
            print("Запрос не может быть пустым", file=sys.stderr)
            sys.exit(1)
    
    try:
        optimizer = QueryOptimizer(grok_api_key=args.api_key)
        results = optimizer.optimize(query)
        format_output(results, json_output=args.json)
        
    except ValueError as e:
        print(f"\n❌ {e}", file=sys.stderr)
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n\nОперация прервана пользователем", file=sys.stderr)
        sys.exit(130)
    except Exception as e:
        print(f"\n❌ Неожиданная ошибка: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

