#!/usr/bin/env python3
"""
Section text extraction utilities
"""

import re
from typing import List, Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


class SectionTextExtractor:
    """Extract text content for document sections"""

    def __init__(self, max_tokens: int = 200):
        self.max_tokens = max_tokens
        self.stats = {'sections_processed': 0, 'total_tokens': 0}

    def extract_section_text(self, pages: List, heading: Dict[str, Any],
                             next_heading: Optional[Dict[str, Any]] = None) -> str:
        try:
            page_num = heading.get('page', 0)
            if page_num >= len(pages):
                return ""

            page = pages[page_num]
            page_height = page.rect.height

            start_y = 0
            end_y = page_height

            if next_heading and next_heading.get('page') == page_num:
                end_y = min(end_y, page_height * 0.9)


            section_rect = page.rect
            section_rect.y0 = start_y
            section_rect.y1 = end_y

            text = page.get_textbox(section_rect)

            cleaned_text = self._clean_text(text)
            limited_text = self._limit_tokens(cleaned_text, self.max_tokens)

            self.stats['sections_processed'] += 1
            self.stats['total_tokens'] += len(limited_text.split())

            return limited_text

        except Exception as e:
            logger.error(f"Failed to extract section text: {e}")
            return ""

    def _clean_text(self, text: str) -> str:
        """Clean and normalize extracted text"""
        if not text:
            return ""

        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)

        # Remove page numbers and common artifacts
        text = re.sub(r'\b\d+\s*$', '', text)
        text = re.sub(r'^[^a-zA-Z]*', '', text)
        return text.strip()

    def _limit_tokens(self, text: str, max_tokens: int) -> str:
        """Limit text to maximum number of tokens"""
        words = text.split()
        if len(words) <= max_tokens:
            return text

        truncated = ' '.join(words[:max_tokens])


        last_period = truncated.rfind('.')
        if last_period > len(truncated) * 0.7:
            truncated = truncated[:last_period + 1]

        return truncated

    def get_stats(self) -> Dict[str, int]:
        """Get extraction statistics"""
        return self.stats.copy()
