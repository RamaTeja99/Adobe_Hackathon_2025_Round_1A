from .reader import PDFReader, get_text_blocks_vectorized, analyze_font_distribution, extract_blocks_from_page
from .heading_detect import HeadingDetector, FontClusterer
from .text_extract import TextExtractor, clean_heading_text

__version__ = "1.1.3"
__author__ = "Creative Codex"

__all__ = [
    "PDFReader",
    "get_text_blocks_vectorized", 
    "analyze_font_distribution",
    "extract_blocks_from_page",
    "HeadingDetector",
    "FontClusterer", 
    "TextExtractor",
    "clean_heading_text"
]