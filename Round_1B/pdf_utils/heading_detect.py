import numpy as np
import re
from typing import List, Dict, Any
from sklearn.cluster import MiniBatchKMeans
import logging

logger = logging.getLogger(__name__)
class FontClusterer:
    def __init__(self, n_clusters: int = 3, random_state: int = 42):
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.clusterer = None
        self.cluster_centers_ = None
        self.level_mapping = {}
        self.font_threshold = 1.0

    def fit(self, font_sizes: np.ndarray, body_size: float) -> 'FontClusterer':
        if len(font_sizes) == 0:
            logger.warning("No font sizes provided for clustering")
            return self
        heading_sizes = font_sizes[font_sizes > body_size + self.font_threshold]
        if len(heading_sizes) == 0:
            logger.warning("No clear heading sizes found")
            return self
        unique_sizes = np.unique(heading_sizes)
        n_clusters = min(self.n_clusters, len(unique_sizes))
        if n_clusters == 1:
            self.level_mapping = {0: "H1"}
            self.cluster_centers_ = np.array([unique_sizes[0]])
            return self
        sizes_2d = heading_sizes.reshape(-1, 1)
        self.clusterer = MiniBatchKMeans(
            n_clusters=n_clusters,
            random_state=self.random_state,
            batch_size=min(1000, len(heading_sizes))
        )
        self.clusterer.fit(sizes_2d)
        self.cluster_centers_ = self.clusterer.cluster_centers_.flatten()
        sorted_indices = np.argsort(self.cluster_centers_)[::-1]
        level_names = ["H1", "H2", "H3", "H4"]
        self.level_mapping = {}
        for i, cluster_idx in enumerate(sorted_indices):
            if i < len(level_names):
                self.level_mapping[cluster_idx] = level_names[i]
            else:
                self.level_mapping[cluster_idx] = "H3"
        logger.info(
            f"Font clusters: {dict(zip(self.cluster_centers_[sorted_indices], [self.level_mapping.get(i, f'H{i + 1}') for i in sorted_indices]))}")
        return self

    """Predict heading level with improved logic."""
    def predict_level(self, font_size: float, body_size: float) -> str:
        if self.clusterer is None:
            size_diff = font_size - body_size
            if size_diff >= 4.0:
                return "H1"
            elif size_diff >= 2.0:
                return "H2"
            elif size_diff >= 1.0:
                return "H3"
            else:
                return "H3"

        cluster = self.clusterer.predict([[font_size]])[0]
        return self.level_mapping.get(cluster, "H3")

    """Get information about the fitted clusters."""
    def get_cluster_info(self) -> Dict[str, Any]:
        if self.cluster_centers_ is None:
            return {}
        return {
            "centers": self.cluster_centers_.tolist(),
            "mapping": self.level_mapping,
            "n_clusters": len(self.cluster_centers_)
        }


class HeadingDetector:
    WEIGHTS = {
        'font_rank': 0.40,
        'bold_flag': 0.30,
        'line_position': 0.15,
        'cue_words': 0.15
    }

    PATTERNS = {
        'numbering': re.compile(r'^\s*(?:\d+(?:[.)]|\s+)|\d+(?:\.\d+)+\s*|[IVXLCDM]+\.\s*|[a-zA-Z]\.\s*)',
                                re.IGNORECASE),
        'section_numbering': re.compile(r'^\s*\d+(?:\.\d+)*\s+', re.IGNORECASE),
        'cue_words': re.compile(
            r'^\s*(?:chapter|section|appendix|introduction|conclusion|abstract|summary|references|bibliography|table\s+of\s+contents|acknowledgements?|revision\s+history|business\s+outcomes|content|background|timeline|milestones|approach|evaluation|phase|preamble|membership|requirements?|objectives?)\b',
            re.IGNORECASE),
        'japanese_chapter': re.compile(r'^第.*章$'),
        'table_like': re.compile(r'.*[:;]\s*$|^[^a-zA-Z]*$'),
        'stop_words': re.compile(r'^(?:the|a|an|and|or|but|in|on|at|to|for|of|with|by)\b', re.IGNORECASE),
        'page_numbers': re.compile(r'^\s*(?:page\s+)?\d+(?:\s+of\s+\d+)?\s*$', re.IGNORECASE),
        'copyright_footer': re.compile(r'©|copyright|\d{4}.*\d{4}|version\s+\d+', re.IGNORECASE)
    }

    def __init__(self, heading_threshold: float = 1.2):
        self.heading_threshold = heading_threshold
        self.font_clusterer = FontClusterer()
        self.body_font_size = 12.0
        self.font_stats = {}

    def analyze_document_fonts(self, blocks: List[Dict[str, Any]]) -> None:
        if not blocks:
            logger.warning("No blocks provided for font analysis")
            return
        font_sizes = np.array([block['font_size'] for block in blocks])
        self.body_font_size = np.median(font_sizes)
        self.font_stats = {
            'body_size': self.body_font_size,
            'mean_size': np.mean(font_sizes),
            'std_size': np.std(font_sizes),
            'percentiles': {
                '75': np.percentile(font_sizes, 75),
                '85': np.percentile(font_sizes, 85),
                '90': np.percentile(font_sizes, 90),
                '95': np.percentile(font_sizes, 95)
            }
        }
        heading_candidates = font_sizes[font_sizes > self.body_font_size * self.heading_threshold]
        if len(heading_candidates) > 0:
            self.font_clusterer.fit(heading_candidates, self.body_font_size)
        else:
            logger.info("No clear heading candidates found based on font size")

    def score_heading_candidate(self, block: Dict[str, Any]) -> float:
        """Improved scoring algorithm for heading candidates."""
        text = block['text'].strip()
        font_size = block['font_size']
        bbox = block['bbox']
        page_height = block.get('page_height', 792)
        is_bold = block.get('is_bold', False)
        if len(text) < 2:
            return 0.0
        if (self.PATTERNS['page_numbers'].match(text) or
                self.PATTERNS['copyright_footer'].search(text) or
                self.PATTERNS['table_like'].match(text)):
            return 0.0
        if font_size < self.body_font_size * self.heading_threshold:
            return 0.0
        font_multiplier = font_size / self.body_font_size
        font_rank = min(1.0, (font_multiplier - 1.0) / 2.0)
        bold_flag = 1.0 if is_bold else 0.0
        relative_y = bbox[1] / page_height if page_height > 0 else 0.5
        line_position = max(0.0, 1.0 - relative_y)
        cue_words = 1.0 if (self.PATTERNS['cue_words'].match(text) or
                            self.PATTERNS['japanese_chapter'].match(text)) else 0.0
        score = (
                self.WEIGHTS['font_rank'] * font_rank +
                self.WEIGHTS['bold_flag'] * bold_flag +
                self.WEIGHTS['line_position'] * line_position +
                self.WEIGHTS['cue_words'] * cue_words
        )

        return score

    def detect_headings(self, blocks: List[Dict[str, Any]],
                        max_headings_per_page: int = 10) -> List[Dict[str, Any]]:
        if not blocks:
            return []
        self.analyze_document_fonts(blocks)
        pages_blocks = {}
        for block in blocks:
            page_num = block.get('page', 0)
            if page_num not in pages_blocks:
                pages_blocks[page_num] = []
            pages_blocks[page_num].append(block)
        all_candidates = []
        for page_num, page_blocks in pages_blocks.items():
            scored_blocks = []
            for block in page_blocks:
                score = self.score_heading_candidate(block)
                if score > 0.1:
                    scored_blocks.append((score, block))
            scored_blocks.sort(reverse=True, key=lambda x: x[0])
            top_candidates = scored_blocks[:max_headings_per_page]
            for score, block in top_candidates:
                level = self.font_clusterer.predict_level(block['font_size'], self.body_font_size)
                candidate = {
                    'text': block['text'],
                    'level': level,
                    'page': page_num,
                    'score': score,
                    'font_size': block['font_size'],
                    'bbox': block['bbox']
                }
                all_candidates.append(candidate)
        all_candidates.sort(key=lambda x: (x['page'], x['bbox'][1]))
        filtered_headings = self._apply_hierarchy_rules(all_candidates)
        return filtered_headings

    def _apply_hierarchy_rules(self, candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if not candidates:
            return []
        unique_candidates = self._remove_duplicates(candidates)
        hierarchical_headings = self._enforce_hierarchy(unique_candidates)

        return hierarchical_headings

    def _remove_duplicates(self, candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        seen_texts = set()
        unique_candidates = []

        for candidate in candidates:
            text = candidate['text'].strip().lower()
            is_duplicate = False
            for seen_text in seen_texts:
                if text == seen_text or (len(text) > 10 and text in seen_text) or (
                        len(seen_text) > 10 and seen_text in text):
                    is_duplicate = True
                    break

            if not is_duplicate:
                seen_texts.add(text)
                unique_candidates.append(candidate)

        return unique_candidates

    def _enforce_hierarchy(self, candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if not candidates:
            return []

        result = []
        last_level_num = 0

        for candidate in candidates:
            current_level = candidate['level']
            level_num = int(current_level[1])
            if level_num > last_level_num + 2:
                level_num = last_level_num + 1
                candidate['level'] = f'H{level_num}'

            last_level_num = level_num
            result.append(candidate)

        return result

    def get_detection_stats(self) -> Dict[str, Any]:
        cluster_info = self.font_clusterer.get_cluster_info()
        return {
            'body_font_size': self.body_font_size,
            'font_stats': self.font_stats,
            'font_clusters': cluster_info,
            'weights': self.WEIGHTS,
            'threshold': self.heading_threshold
        }