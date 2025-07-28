import fitz
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class PDFReader:
    def __init__(self, pdf_path: str):
        self.pdf_path = pdf_path
        self.doc = None
        self.pages = []
        self._load_document()

    def _load_document(self) -> None:
        try:
            self.doc = fitz.open(self.pdf_path)
            self.pages = [self.doc.load_page(i) for i in range(self.doc.page_count)]
            logger.info(f"Loaded PDF with {len(self.pages)} pages")
        except Exception as e:
            logger.error(f"Failed to load PDF {self.pdf_path}: {e}")
            raise

    def get_page_count(self) -> int:
        return len(self.pages)

    def get_page(self, page_num: int) -> fitz.Page:
        if 0 <= page_num < len(self.pages):
            return self.pages[page_num]
        raise IndexError(f"Page {page_num} out of range (0-{len(self.pages) - 1})")

    def get_title_from_metadata(self) -> Optional[str]:
        if self.doc and self.doc.metadata:
            title = self.doc.metadata.get("title", "").strip()
            return title if title else None
        return None

    def close(self):
        if self.doc:
            self.doc.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


def extract_blocks_from_page(page: fitz.Page) -> List[Dict[str, Any]]:
    blocks = []
    page_height = page.rect.height
    page_width = page.rect.width

    try:
        text_dict = page.get_text("dict")
        for block in text_dict.get("blocks", []):
            if "lines" not in block:
                continue

            for line in block["lines"]:
                for span in line["spans"]:
                    text = span.get("text", "").strip()
                    if not text:
                        continue

                    font_size = span.get("size", 12.0)
                    font_flags = span.get("flags", 0)
                    bbox = span.get("bbox", [0, 0, 0, 0])

                    relative_y = bbox[1] / page_height if page_height > 0 else 0

                    block_info = {
                        "text": text,
                        "bbox": bbox,
                        "font_name": span.get("font", "unknown"),
                        "font_size": font_size,
                        "flags": font_flags,
                        "page_height": page_height,
                        "page_width": page_width,
                        "relative_y": relative_y,
                        "is_bold": bool(font_flags & 16),
                        "is_italic": bool(font_flags & 2),
                    }
                    blocks.append(block_info)

    except Exception as e:
        logger.error(f"Error extracting blocks from page: {e}")

    return blocks


def get_text_blocks_vectorized(pages: List[fitz.Page]) -> Tuple[np.ndarray, List[Dict]]:
    """Extract blocks with improved vectorization and font analysis."""
    all_blocks = []
    font_sizes = []
    font_flags = []
    y_positions = []
    x_positions = []

    for page_num, page in enumerate(pages):
        blocks = extract_blocks_from_page(page)

        for block in blocks:
            block["page"] = page_num
            all_blocks.append(block)
            font_sizes.append(block["font_size"])
            font_flags.append(1 if block["is_bold"] else 0)
            y_positions.append(block["relative_y"])
            x_positions.append(block["bbox"][0])

    if font_sizes:
        features = np.column_stack([
            np.array(font_sizes, dtype=np.float32),
            np.array(font_flags, dtype=np.int32),
            np.array(y_positions, dtype=np.float32),
            np.array(x_positions, dtype=np.float32)
        ])
    else:
        features = np.empty((0, 4), dtype=np.float32)

    return features, all_blocks


def analyze_font_distribution(blocks: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Analyze font size distribution with improved statistics."""
    if not blocks:
        return {"body_size": 12.0, "sizes": [], "percentiles": {}}

    sizes = [block["font_size"] for block in blocks]
    sizes_array = np.array(sizes)

    stats = {
        "body_size": np.median(sizes_array),  
        "mean_size": np.mean(sizes_array),
        "std_size": np.std(sizes_array),
        "sizes": sizes,
        "unique_sizes": sorted(list(set(sizes))),
        "size_counts": {},
        "percentiles": {
            "25": np.percentile(sizes_array, 25),
            "50": np.percentile(sizes_array, 50),
            "75": np.percentile(sizes_array, 75),
            "85": np.percentile(sizes_array, 85),
            "90": np.percentile(sizes_array, 90),
            "95": np.percentile(sizes_array, 95)
        }
    }

    unique, counts = np.unique(sizes_array, return_counts=True)
    stats["size_counts"] = dict(zip(unique, counts))

    return stats


FONT_FLAGS = {
    "SUPERSCRIPT": 1 << 0,
    "ITALIC": 1 << 1,
    "SERIFED": 1 << 2,
    "MONOSPACED": 1 << 3,
    "BOLD": 1 << 4
}


def is_bold(flags: int) -> bool:
    """Check if text is bold based on font flags."""
    return bool(flags & FONT_FLAGS["BOLD"])


def is_italic(flags: int) -> bool:
    """Check if text is italic based on font flags."""
    return bool(flags & FONT_FLAGS["ITALIC"])