#!/usr/bin/env python3
"""
Text refinement using TextRank for extractive summarization
"""

import os
import nltk
import numpy as np
import networkx as nx
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Tuple
import logging

logger = logging.getLogger(__name__)


NLTK_DATA = os.getenv("NLTK_DATA", "/tmp/hf_cache")
nltk.data.path.append(NLTK_DATA)

def _ensure_nltk_data():
    """Ensure NLTK data is available with fallback"""
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        try:
            nltk.download('punkt', download_dir=NLTK_DATA, quiet=True)
        except Exception as e:
            logger.warning(f"Failed to download NLTK punkt: {e}")


_ensure_nltk_data()


class TextRefiner:
    """Refine section text using TextRank summarization"""

    def __init__(self, embedding_model, max_sentences: int = 3):
        self.embedding_model = embedding_model
        self.max_sentences = max_sentences

    def _safe_tokenize(self, text: str) -> List[str]:
        """Safe sentence tokenization with fallback"""
        try:
            return nltk.sent_tokenize(text)
        except Exception as e:
            logger.warning(f"NLTK tokenization failed: {e}, using simple fallback")
            import re
            sentences = re.split(r'[.!?]+', text)
            return [s.strip() for s in sentences if s.strip()]

    def refine_text(self, text: str) -> str:
        if not text or not text.strip():
            return text

        try:

            sentences = self._safe_tokenize(text)


            if len(sentences) <= self.max_sentences:
                return text.strip()


            sentence_embeddings = self.embedding_model.encode(sentences, show_progress=False)


            similarity_matrix = cosine_similarity(sentence_embeddings)


            np.fill_diagonal(similarity_matrix, 0)


            graph = nx.from_numpy_array(similarity_matrix)
            scores = nx.pagerank(graph)


            ranked_sentences = sorted(scores.items(), key=lambda x: x[1], reverse=True)
            top_sentence_indices = [idx for idx, score in ranked_sentences[:self.max_sentences]]
            top_sentence_indices.sort()
            refined_sentences = [sentences[idx] for idx in top_sentence_indices]
            refined_text = ' '.join(refined_sentences)

            logger.debug(f"Refined {len(sentences)} sentences to {len(refined_sentences)}")
            return refined_text.strip()

        except Exception as e:
            logger.error(f"TextRank refinement failed: {e}")
            sentences = self._safe_tokenize(text)
            return ' '.join(sentences[:self.max_sentences]).strip()

    def get_sentence_scores(self, text: str) -> List[Tuple[str, float]]:
        """Get sentence importance scores for analysis"""
        if not text or not text.strip():
            return []

        try:
            sentences = self._safe_tokenize(text)
            if len(sentences) <= 1:
                return [(sentences[0], 1.0)] if sentences else []

            sentence_embeddings = self.embedding_model.encode(sentences, show_progress=False)
            similarity_matrix = cosine_similarity(sentence_embeddings)
            np.fill_diagonal(similarity_matrix, 0)

            graph = nx.from_numpy_array(similarity_matrix)
            scores = nx.pagerank(graph)

            return [(sentences[idx], score) for idx, score in scores.items()]

        except Exception as e:
            logger.error(f"Failed to get sentence scores: {e}")
            sentences = self._safe_tokenize(text)
            return [(sent, 1.0 / len(sentences)) for sent in sentences]
