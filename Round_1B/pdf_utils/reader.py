#!/usr/bin/env python3
""" Lightweight PDF reader using PyMuPDF """

import fitz
from typing import List, Optional
import logging

logger = logging.getLogger(__name__)


class PDFReader:
    """Context manager for reading PDF files"""

    def __init__(self, path: str):
        self.path = path
        self.doc = None
        self.pages = []

    def __enter__(self):
        try:
            self.doc = fitz.open(self.path)
            self.pages = [self.doc.load_page(i) for i in range(self.doc.page_count)]
            logger.debug(f"Loaded PDF with {len(self.pages)} pages: {self.path}")
            return self
        except Exception as e:
            logger.error(f"Failed to open PDF {self.path}: {e}")
            raise

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.doc:
            self.doc.close()

    def get_page_count(self) -> int:
        return len(self.pages)

    def get_page_text(self, page_num: int) -> str:
        """Extract text from a specific page"""
        if 0 <= page_num < len(self.pages):
            return self.pages[page_num].get_text()
        return ""

    def get_text_in_bbox(self, page_num: int, bbox: List[float]) -> str:
        """Extract text within a bounding box"""
        if 0 <= page_num < len(self.pages):
            rect = fitz.Rect(bbox)
            return self.pages[page_num].get_textbox(rect)
        return ""
