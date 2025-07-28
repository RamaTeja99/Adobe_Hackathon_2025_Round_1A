#!/usr/bin/env python3
"""
Adobe Hackathon 2025 - Round 1B: Persona Section Miner
Main entry point for document collection analysis
"""

import os
import sys
import json
import time
import logging
from pathlib import Path
from typing import Dict, Any

from pdf_utils import EmbeddingModel
from persona_miner import SectionIndexer, HybridRanker, TextRefiner, OutputGenerator


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class PersonaSectionMiner:
    """Main application class for Round 1B"""

    def __init__(self):
        self.embedding_model = EmbeddingModel()
        self.section_indexer = SectionIndexer(self.embedding_model)
        self.hybrid_ranker = HybridRanker()
        self.text_refiner = None  # Will be initialized with embedding model
        self.output_generator = OutputGenerator()

        self.stats = {
            'start_time': time.time(),
            'documents_processed': 0,
            'sections_indexed': 0,
            'sections_selected': 0
        }

    def run(self, input_path: str, output_path: str, outline_dir: str = "input/outlines"):

        logger.info("=== Adobe Hackathon 2025 - Round 1B: Persona Section Miner ===")
        logger.info(f"Input: {input_path}")
        logger.info(f"Output: {output_path}")
        logger.info(f"Outline directory: {outline_dir}")

        try:

            config = self._load_input_config(input_path)


            query = self._build_query(config)
            logger.info(f"Query: {query}")


            sections, all_texts = self.section_indexer.build_index(
                config['documents'], outline_dir
            )
            self.stats['documents_processed'] = len(config['documents'])
            self.stats['sections_indexed'] = len(sections)

            semantic_results = self.section_indexer.search(query, k=30)


            ranked_results = self.hybrid_ranker.rank_sections(
                query, semantic_results, all_texts, k=7
            )
            self.stats['sections_selected'] = len(ranked_results)


            self.text_refiner = TextRefiner(self.embedding_model)
            refined_texts = []

            for section_idx, score in ranked_results:
                section_text = sections[section_idx]['text']
                refined_text = self.text_refiner.refine_text(section_text)
                refined_texts.append(refined_text)


            self.stats['total_time'] = time.time() - self.stats['start_time']
            self.output_generator.set_processing_stats(self.stats)

            output_data = self.output_generator.generate_output(
                config, ranked_results, sections, refined_texts, output_path
            )

            # Step 8: Validate output
            if self.output_generator.validate_output(output_data):
                logger.info("✅ Round 1B processing completed successfully")
                self._log_final_stats()
            else:
                logger.error("❌ Output validation failed")
                return False

            return True

        except Exception as e:
            logger.error(f"❌ Processing failed: {e}")
            return False

    def _load_input_config(self, input_path: str) -> Dict[str, Any]:
        """Load and validate input configuration"""
        if not os.path.exists(input_path):
            raise FileNotFoundError(f"Input file not found: {input_path}")

        with open(input_path, 'r', encoding='utf-8') as f:
            config = json.load(f)

        required_keys = ['persona', 'job_to_be_done', 'documents']
        for key in required_keys:
            if key not in config:
                raise KeyError(f"Missing required key in input: {key}")

        logger.info(f"Loaded configuration with {len(config['documents'])} documents")
        return config

    def _build_query(self, config: Dict[str, Any]) -> str:
        """Build search query from persona and job-to-be-done"""
        persona = config['persona']
        jtbd = config['job_to_be_done']

        query_parts = []

        # Add persona information
        if 'role' in persona:
            query_parts.append(persona['role'])

        if 'domain_expertise' in persona:
            query_parts.extend(persona['domain_expertise'])

        # Add job-to-be-done information
        if 'task' in jtbd:
            query_parts.append(jtbd['task'])

        if 'context' in jtbd:
            query_parts.append(jtbd['context'])

        if 'desired_outcome' in jtbd:
            query_parts.append(jtbd['desired_outcome'])

        query = ' '.join(filter(None, query_parts))
        return query

    def _log_final_stats(self):
        """Log final processing statistics"""
        logger.info("=== Processing Statistics ===")
        logger.info(f"Documents processed: {self.stats['documents_processed']}")
        logger.info(f"Sections indexed: {self.stats['sections_indexed']}")
        logger.info(f"Sections selected: {self.stats['sections_selected']}")
        logger.info(f"Total processing time: {self.stats['total_time']:.2f}s")
        logger.info(
            f"Average time per document: {self.stats['total_time'] / max(1, self.stats['documents_processed']):.2f}s")


def main():
    """Main entry point"""
    # Parse command line arguments
    input_path = sys.argv[1] if len(sys.argv) > 1 else "/app/input/challenge1b_input.json"
    output_path = sys.argv[2] if len(sys.argv) > 2 else "/app/output/challenge1b_output.json"
    outline_dir = sys.argv[3] if len(sys.argv) > 3 else "/app/input/outlines"

    # For local development, use relative paths
    if not input_path.startswith('/app'):
        input_path = os.path.join("input", "challenge1b_input.json")
        output_path = os.path.join("output", "challenge1b_output.json")
        outline_dir = os.path.join("input", "outlines")

    # Run the persona section miner
    miner = PersonaSectionMiner()
    success = miner.run(input_path, output_path, outline_dir)

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
