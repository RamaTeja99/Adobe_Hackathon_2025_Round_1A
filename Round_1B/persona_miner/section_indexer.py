#!/usr/bin/env python3
"""
Section indexing with FAISS for efficient similarity search
"""

import json
import numpy as np
import faiss
from pathlib import Path
from typing import List, Dict, Any, Tuple
import logging
import re

from pdf_utils import PDFReader, SectionTextExtractor, EmbeddingModel

logger = logging.getLogger(__name__)


class SectionIndexer:
    """Build and manage section embeddings index"""

    def __init__(self, embedding_model: EmbeddingModel):
        self.embedding_model = embedding_model
        self.text_extractor = SectionTextExtractor()
        self.sections = []
        self.embeddings = None
        self.index = None

    def _normalize_filename(self, filename: str) -> str:
        """Convert display names to likely PDF filenames"""
        if filename.endswith('.pdf'):
            filename = filename[:-4]

        # Replace spaces with hyphens and normalize
        normalized = filename.replace(' ', '-')
        normalized = re.sub(r'[^\w\-]', '', normalized)
        return normalized

    def _find_outline_file(self, doc: Dict[str, str], outline_dir: str) -> Path:
        """Find outline file using multiple naming strategies"""
        outline_dir_path = Path(outline_dir)

        if 'filename' in doc and doc['filename']:
            filename = doc['filename']
            stem = filename[:-4] if filename.endswith('.pdf') else filename
            outline_file = outline_dir_path / f"{stem}.json"
            if outline_file.exists():
                return outline_file

        if 'title' in doc and doc['title']:
            normalized_title = self._normalize_filename(doc['title'])
            outline_file = outline_dir_path / f"{normalized_title}.json"
            if outline_file.exists():
                return outline_file

        search_terms = []
        if 'title' in doc:
            search_terms.append(doc['title'].lower())
        if 'filename' in doc:
            search_terms.append(doc['filename'].lower().replace('.pdf', ''))

        for outline_file in outline_dir_path.glob("*.json"):
            outline_name = outline_file.stem.lower()
            for term in search_terms:
                # Word overlap matching
                term_words = set(re.findall(r'\w+', term))
                file_words = set(re.findall(r'\w+', outline_name))

                if len(term_words & file_words) >= max(1, len(term_words) * 0.7):
                    logger.info(f"Found outline by fuzzy match: {outline_file}")
                    return outline_file

        return None

    def build_index(self, documents: List[Dict[str, str]], outline_dir: str) -> Tuple[List[Dict], List[str]]:
        """Build FAISS index from document outlines"""
        logger.info(f"Building index from {len(documents)} documents")

        # Debug: List available outline files
        outline_dir_path = Path(outline_dir)
        if outline_dir_path.exists():
            available_files = list(outline_dir_path.glob("*.json"))
            logger.info(f"Available outline files: {[f.name for f in available_files]}")
        else:
            logger.error(f"Outline directory does not exist: {outline_dir}")
            return [], []

        sections = []
        texts = []

        for i, doc in enumerate(documents):
            logger.info(f"Processing document {i + 1}/{len(documents)}: {doc}")
            doc_sections, doc_texts = self._process_document(doc, outline_dir)
            sections.extend(doc_sections)
            texts.extend(doc_texts)

        if not texts:
            raise ValueError("No sections found to index")

        logger.info(f"Indexing {len(texts)} sections")

        # Generate embeddings and build FAISS index
        embeddings = self.embedding_model.encode(texts, batch_size=32, show_progress=False)

        embedding_dim = embeddings.shape[1]
        index = faiss.IndexFlatIP(embedding_dim)
        faiss.normalize_L2(embeddings)
        index.add(embeddings)

        self.sections = sections
        self.embeddings = embeddings
        self.index = index

        logger.info(f"Index built with {index.ntotal} vectors")
        return sections, texts

    def _process_document(self, doc: Dict[str, str], outline_dir: str) -> Tuple[List[Dict], List[str]]:
        """Process a single document's outline"""
        outline_file = self._find_outline_file(doc, outline_dir)

        if not outline_file:
            logger.warning(f"No outline found for document: {doc}")
            return [], []

        logger.info(f"Using outline file: {outline_file}")

        try:
            with open(outline_file, 'r', encoding='utf-8') as f:
                outline_data = json.load(f)
        except Exception as e:
            logger.error(f"Failed to load outline {outline_file}: {e}")
            return [], []

        headings = outline_data.get('outline', [])
        if not headings:
            logger.warning(f"No headings in outline: {outline_file}")
            return [], []

        sections = []
        texts = []
        doc_name = doc.get('filename', doc.get('title', 'Unknown'))

        for heading in headings:
            section = {
                'document': doc_name,
                'heading': heading['text'],
                'level': heading['level'],
                'page': heading['page'],
                'text': heading['text']
            }

            sections.append(section)
            texts.append(heading['text'])

        logger.info(f"Processed {len(sections)} sections from {doc_name}")
        return sections, texts

    def search(self, query: str, k: int = 30) -> List[Tuple[int, float]]:
        """Search for similar sections"""
        if self.index is None:
            raise ValueError("Index not built yet")

        query_embedding = self.embedding_model.encode([query], normalize_embeddings=True)
        similarities, indices = self.index.search(query_embedding, k)

        return [(int(idx), float(sim)) for idx, sim in zip(indices[0], similarities[0])]
