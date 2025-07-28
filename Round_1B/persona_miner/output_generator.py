#!/usr/bin/env python3
"""
Generate final JSON output for Round 1B
"""

import json
from datetime import datetime, timezone
from typing import List, Dict, Any,Tuple
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class OutputGenerator:
    """Generate schema-compliant output JSON"""

    def __init__(self):
        self.processing_stats = {}

    def generate_output(self,
                        input_config: Dict[str, Any],
                        selected_sections: List[Tuple[int, float]],
                        all_sections: List[Dict[str, Any]],
                        refined_texts: List[str],
                        output_path: str):



        persona = input_config.get('persona', {})
        job_to_be_done = input_config.get('job_to_be_done', {})
        documents = input_config.get('documents', [])


        extracted_sections = []
        subsection_analysis = []

        for rank, (section_idx, relevance_score) in enumerate(selected_sections, 1):
            section = all_sections[section_idx]
            refined_text = refined_texts[rank - 1] if rank - 1 < len(refined_texts) else section.get('text', '')


            doc_filename = section['document']
            if doc_filename.endswith('.pdf'):
                doc_filename = Path(doc_filename).name


            extracted_sections.append({
                "document": doc_filename,
                "section_title": section['heading'],
                "importance_rank": rank,
                "page_number": section['page'],
                "relevance_score": round(float(relevance_score), 3)
            })


            subsection_analysis.append({
                "document": doc_filename,
                "section_title": section['heading'],
                "refined_text": refined_text,
                "page_number": section['page'],
                "extraction_method": "TextRank"
            })

        output_data = {
            "metadata": {
                "input_documents": [doc.get('filename', doc.get('title', '')) for doc in documents],
                "persona": persona.get('role', ''),
                "job_to_be_done": job_to_be_done.get('task', ''),
                "processing_timestamp": datetime.now(timezone.utc).isoformat(),
                "total_sections_analyzed": len(all_sections),
                "selected_sections": len(selected_sections),
                "processing_stats": self.processing_stats
            },
            "extracted_sections": extracted_sections,
            "subsection_analysis": subsection_analysis
        }

        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)

        logger.info(f"Generated output: {output_path}")
        logger.info(f"Selected {len(extracted_sections)} sections from {len(all_sections)} total")

        return output_data

    def set_processing_stats(self, stats: Dict[str, Any]):
        """Set processing statistics for metadata"""
        self.processing_stats = stats

    def validate_output(self, output_data: Dict[str, Any]) -> bool:
        """Basic validation of output structure"""
        required_keys = ['metadata', 'extracted_sections', 'subsection_analysis']

        if not all(key in output_data for key in required_keys):
            logger.error("Missing required keys in output")
            return False

        metadata = output_data['metadata']
        required_metadata = ['input_documents', 'persona', 'job_to_be_done', 'processing_timestamp']

        if not all(key in metadata for key in required_metadata):
            logger.error("Missing required metadata keys")
            return False

        extracted = output_data['extracted_sections']
        analysis = output_data['subsection_analysis']

        if len(extracted) != len(analysis):
            logger.error("Mismatch between extracted sections and analysis")
            return False

        logger.info("Output validation passed")
        return True
