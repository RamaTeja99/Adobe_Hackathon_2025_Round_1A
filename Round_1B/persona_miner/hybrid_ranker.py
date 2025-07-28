#!/usr/bin/env python3
"""
Hybrid ranking combining semantic similarity and BM25 lexical matching
"""

import numpy as np
from rank_bm25 import BM25Okapi
from typing import List, Dict, Tuple
import logging

logger = logging.getLogger(__name__)


class HybridRanker:
    """Combine semantic and lexical ranking"""

    def __init__(self, semantic_weight: float = 0.7, lexical_weight: float = 0.3):
        self.semantic_weight = semantic_weight
        self.lexical_weight = lexical_weight
        self.bm25 = None

    def build_bm25_index(self, texts: List[str]):
        """Build BM25 index from texts"""
        logger.info(f"Building BM25 index from {len(texts)} texts")


        tokenized_texts = [text.lower().split() for text in texts]
        self.bm25 = BM25Okapi(tokenized_texts)

    def rank_sections(self, query: str, semantic_results: List[Tuple[int, float]],
                      all_texts: List[str], k: int = 7) -> List[Tuple[int, float]]:
        if self.bm25 is None:
            self.build_bm25_index(all_texts)


        query_tokens = query.lower().split()
        bm25_scores = self.bm25.get_scores(query_tokens)


        if np.max(bm25_scores) > 0:
            bm25_scores = bm25_scores / np.max(bm25_scores)

        combined_scores = []

        for idx, semantic_score in semantic_results:
            lexical_score = bm25_scores[idx] if idx < len(bm25_scores) else 0.0

            combined_score = (
                    self.semantic_weight * semantic_score +
                    self.lexical_weight * lexical_score
            )

            combined_scores.append((idx, combined_score))

        combined_scores.sort(key=lambda x: x[1], reverse=True)
        return combined_scores[:k]

    def get_ranking_explanation(self, query: str, section_idx: int,
                                semantic_score: float, all_texts: List[str]) -> Dict[str, float]:
        """Get detailed scoring breakdown for a section"""
        if self.bm25 is None or section_idx >= len(all_texts):
            return {}

        query_tokens = query.lower().split()
        bm25_scores = self.bm25.get_scores(query_tokens)
        lexical_score = bm25_scores[section_idx] / np.max(bm25_scores) if np.max(bm25_scores) > 0 else 0

        combined_score = (
                self.semantic_weight * semantic_score +
                self.lexical_weight * lexical_score
        )

        return {
            'semantic_score': float(semantic_score),
            'lexical_score': float(lexical_score),
            'combined_score': float(combined_score),
            'semantic_weight': self.semantic_weight,
            'lexical_weight': self.lexical_weight
        }
