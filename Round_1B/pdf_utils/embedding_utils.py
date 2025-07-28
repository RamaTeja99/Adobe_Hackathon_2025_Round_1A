#!/usr/bin/env python3
"""
Embedding utilities using sentence-transformers
"""

import os
import numpy as np
from sentence_transformers import SentenceTransformer
from typing import List, Union
import time
import logging
import nltk

logger = logging.getLogger(__name__)

HF_CACHE = os.getenv("TRANSFORMERS_CACHE", "/tmp/hf_cache")
NLTK_DATA = os.getenv("NLTK_DATA", "/tmp/hf_cache")
os.makedirs(HF_CACHE, exist_ok=True)
os.makedirs(NLTK_DATA, exist_ok=True)
nltk.data.path.append(NLTK_DATA)


class EmbeddingModel:
    """Wrapper for sentence transformer embedding model"""

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.model = None
        self.embedding_dim = None
        self._ensure_nltk_data()

    def _ensure_nltk_data(self):
        """Ensure NLTK data is available"""
        try:
            nltk.data.find('tokenizers/punkt')
            logger.debug("NLTK punkt tokenizer already available")
        except LookupError:
            try:
                logger.info("Downloading NLTK punkt tokenizer...")
                nltk.download('punkt', download_dir=NLTK_DATA, quiet=True)
                logger.info("NLTK punkt tokenizer downloaded successfully")
            except Exception as e:
                logger.warning(f"Failed to download NLTK data: {e}")

    def load_model(self):
        """Load the sentence transformer model"""
        if self.model is None:
            start_time = time.time()
            logger.info(f"Loading embedding model: {self.model_name}")

            self.model = SentenceTransformer(self.model_name, cache_folder=HF_CACHE)
            self.embedding_dim = self.model.get_sentence_embedding_dimension()

            load_time = time.time() - start_time
            logger.info(f"Model loaded in {load_time:.2f}s, embedding dim: {self.embedding_dim}")

    def encode(self, texts: Union[str, List[str]], batch_size: int = 32,
               show_progress: bool = False, **kwargs) -> np.ndarray:
        """Encode texts into embeddings"""
        if self.model is None:
            self.load_model()

        if isinstance(texts, str):
            texts = [texts]

        start_time = time.time()
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
            **kwargs
        )

        encode_time = time.time() - start_time
        logger.debug(f"Encoded {len(texts)} texts in {encode_time:.2f}s")

        return embeddings

    def get_embedding_dimension(self) -> int:
        """Get the embedding dimension"""
        if self.embedding_dim is None and self.model is not None:
            self.embedding_dim = self.model.get_sentence_embedding_dimension()
        return self.embedding_dim or 384
