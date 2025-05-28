"""
FEVER dataset converter for NewAIBench.

FEVER (Fact Extraction and VERification) is a dataset for fact-checking
against Wikipedia articles. Claims need to be verified as SUPPORTED,
REFUTED, or NOT ENOUGH INFO based on evidence from Wikipedia.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Set

from .base_converter import BaseDatasetConverter, ConversionConfig

logger = logging.getLogger(__name__)


class FEVERConverter(BaseDatasetConverter):
    """
    Converter for FEVER dataset.
    
    FEVER contains claims that need to be fact-checked against Wikipedia.
    Each example includes:
    - A claim to be verified
    - Evidence sentences from Wikipedia articles
    - A label (SUPPORTED, REFUTED, NOT ENOUGH INFO)
    
    The converter handles both the original FEVER format and processed versions.
    """
    
    def __init__(self, config: ConversionConfig):
        super().__init__(config)
        self.dataset_name = "fever"
        
    def detect_format(self) -> bool:
        """
        Detect if the input directory contains FEVER data.
        
        Returns:
            bool: True if FEVER format is detected
        """
        input_path = Path(self.config.input_path)
        
        # Check for standard FEVER files
        fever_patterns = [
            "**/fever*.jsonl",
            "**/train*.jsonl",
            "**/dev*.jsonl",
            "**/test*.jsonl",
            "**/shared_task*.jsonl",
            "**/wiki-pages*.jsonl"
        ]
        
        for pattern in fever_patterns:
            if list(input_path.glob(pattern)):
                logger.info(f"Detected FEVER format with pattern: {pattern}")
                return True
                
        # Check for FEVER specific structure
        if self._check_fever_structure(input_path):
            return True
            
        return False
    
    def _check_fever_structure(self, path: Path) -> bool:
        """Check if files contain FEVER structure."""
        for file_path in path.rglob("*.jsonl"):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    # Check first few lines for FEVER structure
                    for i, line in enumerate(f):
                        if i >= 3:  # Check first 3 lines
                            break
                        try:
                            data = json.loads(line.strip())
                            # Look for FEVER specific fields
                            if self._is_fever_format(data):
                                return True
                        except json.JSONDecodeError:
                            continue
            except Exception:
                continue
        return False
    
    def _is_fever_format(self, data: Dict) -> bool:
        """Check if data structure matches FEVER format."""
        # FEVER claim format indicators
        claim_fields = {
            'claim', 'label', 'evidence', 'id', 'verifiable'
        }
        
        # FEVER wiki format indicators
        wiki_fields = {
            'id', 'text', 'lines'
        }
        
        data_fields = set(data.keys())
        
        # Check for claim format
        if len(claim_fields.intersection(data_fields)) >= 3:
            return True
            
        # Check for wiki pages format
        if len(wiki_fields.intersection(data_fields)) >= 2:
            # Additional check for 'lines' being a list of lists
            if 'lines' in data and isinstance(data['lines'], list):
                if len(data['lines']) > 0 and isinstance(data['lines'][0], list):
                    return True
                    
        return False
    
    def convert_corpus(self) -> Path:
        """
        Convert FEVER Wikipedia articles to NewAIBench corpus format.
        
        Returns:
            Path: Path to the converted corpus file
        """
        logger.info("Converting FEVER corpus...")
        
        corpus_file = self.config.output_path / "corpus.jsonl"
        input_path = Path(self.config.input_path)
        
        seen_docs = set()
        doc_count = 0
        
        with open(corpus_file, 'w', encoding='utf-8') as out_f:
            
            # Process all JSONL files
            for file_path in input_path.rglob("*.jsonl"):
                logger.info(f"Processing file: {file_path}")
                
                with open(file_path, 'r', encoding='utf-8') as in_f:
                    for line_num, line in enumerate(in_f, 1):
                        try:
                            line = line.strip()
                            if not line:
                                continue
                                
                            data = json.loads(line)
                            
                            # Extract documents based on format
                            docs = self._extract_documents(data, file_path.stem, line_num)
                            
                            for doc_id, doc_content in docs:
                                if doc_id not in seen_docs:
                                    seen_docs.add(doc_id)
                                    doc_count += 1
                                    
                                    corpus_entry = {
                                        "_id": doc_id,
                                        "title": doc_content.get("title", ""),
                                        "text": doc_content["text"],
                                        "metadata": {
                                            "source": "fever",
                                            "file": file_path.name,
                                            **doc_content.get("metadata", {})
                                        }
                                    }
                                    
                                    out_f.write(json.dumps(corpus_entry) + '\n')
                                    
                        except json.JSONDecodeError as e:
                            logger.warning(f"Invalid JSON on line {line_num} in {file_path}: {e}")
                            continue
                        except Exception as e:
                            logger.warning(f"Error processing line {line_num} in {file_path}: {e}")
                            continue
        
        logger.info(f"Converted {doc_count} documents to corpus")
        self._update_stats("corpus_documents", doc_count)
        
        return corpus_file
    
    def _extract_documents(self, data: Dict, file_name: str, line_num: int) -> List[Tuple[str, Dict]]:
        """Extract document(s) from a FEVER entry."""
        documents = []
        
        # Handle FEVER wiki pages format
        if 'id' in data and 'text' in data and 'lines' in data:
            doc_id = f"fever_{data['id']}"
            title = data.get('text', f"Document_{data['id']}")
            
            # Process lines into text
            lines = data.get('lines', [])
            text_parts = []
            
            for line_entry in lines:
                if isinstance(line_entry, list) and len(line_entry) >= 2:
                    # line_entry is [line_number, sentence_text]
                    sentence = line_entry[1] if len(line_entry) > 1 else str(line_entry[0])
                    if sentence.strip():
                        text_parts.append(sentence.strip())
                elif isinstance(line_entry, str):
                    if line_entry.strip():
                        text_parts.append(line_entry.strip())
            
            full_text = ' '.join(text_parts)
            
            doc_content = {
                "title": title,
                "text": full_text,
                "metadata": {
                    "original_id": data['id'],
                    "line_count": len(lines)
                }
            }
            
            documents.append((doc_id, doc_content))
            
        # Handle FEVER claim format (extract evidence documents)
        elif 'evidence' in data and 'claim' in data:
            evidence_docs = self._extract_evidence_documents(data, file_name, line_num)
            documents.extend(evidence_docs)
        
        # Handle simple text format
        elif 'text' in data:
            doc_id = f"fever_{file_name}_{line_num}"
            if 'id' in data:
                doc_id = f"fever_{data['id']}"
            
            doc_content = {
                "title": data.get('title', ''),
                "text": data['text'],
                "metadata": {}
            }
            documents.append((doc_id, doc_content))
        
        return documents
    
    def _extract_evidence_documents(self, claim_data: Dict, file_name: str, line_num: int) -> List[Tuple[str, Dict]]:
        """Extract evidence documents from FEVER claim data."""
        documents = []
        
        evidence = claim_data.get('evidence', [])
        
        for evidence_set in evidence:
            if isinstance(evidence_set, list):
                for evidence_item in evidence_set:
                    if isinstance(evidence_item, list) and len(evidence_item) >= 3:
                        # evidence_item format: [annotation_id, evidence_id, wikipedia_url, line_number]
                        wiki_url = evidence_item[2] if len(evidence_item) > 2 else None
                        line_number = evidence_item[3] if len(evidence_item) > 3 else None
                        
                        if wiki_url:
                            # Extract title from Wikipedia URL
                            title = self._extract_title_from_url(wiki_url)
                            doc_id = f"fever_evidence_{self._normalize_title(title)}"
                            
                            # Note: In practice, you'd need the actual Wikipedia text
                            # This is a placeholder showing the structure
                            doc_content = {
                                "title": title,
                                "text": f"Evidence from {title} (line {line_number})",
                                "metadata": {
                                    "wikipedia_url": wiki_url,
                                    "line_number": line_number,
                                    "evidence_type": "supporting_evidence"
                                }
                            }
                            
                            documents.append((doc_id, doc_content))
        
        return documents
    
    def _extract_title_from_url(self, url: str) -> str:
        """Extract title from Wikipedia URL."""
        if '/wiki/' in url:
            title = url.split('/wiki/')[-1]
            # Replace URL encoding
            title = title.replace('_', ' ')
            return title
        return url
    
    def _normalize_title(self, title: str) -> str:
        """Normalize title for use as document ID."""
        import re
        normalized = re.sub(r'[^\w\s-]', '', title)
        normalized = re.sub(r'\s+', '_', normalized)
        return normalized.lower()
    
    def convert_queries(self) -> Path:
        """
        Convert FEVER claims to NewAIBench queries format.
        
        Returns:
            Path: Path to the converted queries file
        """
        logger.info("Converting FEVER queries...")
        
        queries_file = self.config.output_path / "queries.jsonl"
        input_path = Path(self.config.input_path)
        
        query_count = 0
        
        with open(queries_file, 'w', encoding='utf-8') as out_f:
            
            for file_path in input_path.rglob("*.jsonl"):
                logger.info(f"Processing queries from: {file_path}")
                
                with open(file_path, 'r', encoding='utf-8') as in_f:
                    for line_num, line in enumerate(in_f, 1):
                        try:
                            line = line.strip()
                            if not line:
                                continue
                                
                            data = json.loads(line)
                            
                            # Extract query (only from claim data, not wiki pages)
                            query_entry = self._extract_query(data, file_path.stem, line_num)
                            
                            if query_entry:
                                out_f.write(json.dumps(query_entry) + '\n')
                                query_count += 1
                                
                        except json.JSONDecodeError as e:
                            logger.warning(f"Invalid JSON on line {line_num} in {file_path}: {e}")
                            continue
                        except Exception as e:
                            logger.warning(f"Error processing line {line_num} in {file_path}: {e}")
                            continue
        
        logger.info(f"Converted {query_count} queries")
        self._update_stats("queries", query_count)
        
        return queries_file
    
    def _extract_query(self, data: Dict, file_name: str, line_num: int) -> Optional[Dict]:
        """Extract query from a FEVER entry."""
        
        # Only process claim data, not wiki pages
        if 'claim' in data:
            query_id = str(data.get('id', f"fever_{file_name}_{line_num}"))
            
            return {
                "_id": query_id,
                "text": data['claim'],
                "metadata": {
                    "source": "fever",
                    "file": file_name,
                    "label": data.get('label', 'unknown'),
                    "verifiable": data.get('verifiable', 'unknown')
                }
            }
        
        return None
    
    def convert_qrels(self) -> Path:
        """
        Convert FEVER evidence to NewAIBench qrels format.
        
        Returns:
            Path: Path to the converted qrels file
        """
        logger.info("Converting FEVER qrels...")
        
        qrels_file = self.config.output_path / "qrels.jsonl"
        input_path = Path(self.config.input_path)
        
        qrel_count = 0
        
        with open(qrels_file, 'w', encoding='utf-8') as out_f:
            
            for file_path in input_path.rglob("*.jsonl"):
                logger.info(f"Processing qrels from: {file_path}")
                
                with open(file_path, 'r', encoding='utf-8') as in_f:
                    for line_num, line in enumerate(in_f, 1):
                        try:
                            line = line.strip()
                            if not line:
                                continue
                                
                            data = json.loads(line)
                            
                            # Extract qrels (only from claim data)
                            qrel_entries = self._extract_qrels(data, file_path.stem, line_num)
                            
                            for qrel_entry in qrel_entries:
                                out_f.write(json.dumps(qrel_entry) + '\n')
                                qrel_count += 1
                                
                        except json.JSONDecodeError as e:
                            logger.warning(f"Invalid JSON on line {line_num} in {file_path}: {e}")
                            continue
                        except Exception as e:
                            logger.warning(f"Error processing line {line_num} in {file_path}: {e}")
                            continue
        
        logger.info(f"Converted {qrel_count} qrels")
        self._update_stats("qrels", qrel_count)
        
        return qrels_file
    
    def _extract_qrels(self, data: Dict, file_name: str, line_num: int) -> List[Dict]:
        """Extract qrels from a FEVER entry."""
        qrel_entries = []
        
        # Only process claim data
        if 'claim' not in data:
            return qrel_entries
            
        query_id = str(data.get('id', f"fever_{file_name}_{line_num}"))
        label = data.get('label', 'NOT ENOUGH INFO')
        evidence = data.get('evidence', [])
        
        # Process evidence
        for evidence_set in evidence:
            if isinstance(evidence_set, list):
                for evidence_item in evidence_set:
                    if isinstance(evidence_item, list) and len(evidence_item) >= 3:
                        wiki_url = evidence_item[2] if len(evidence_item) > 2 else None
                        
                        if wiki_url:
                            title = self._extract_title_from_url(wiki_url)
                            doc_id = f"fever_evidence_{self._normalize_title(title)}"
                            
                            # Determine relevance based on label
                            if label == "SUPPORTS":
                                relevance = 2  # High relevance
                            elif label == "REFUTES":
                                relevance = 1  # Medium relevance (relevant but contradictory)
                            else:  # NOT ENOUGH INFO
                                relevance = 0  # Not relevant
                            
                            qrel_entries.append({
                                "query_id": query_id,
                                "doc_id": doc_id,
                                "relevance": relevance,
                                "metadata": {
                                    "fever_label": label,
                                    "evidence_type": "wikipedia_sentence",
                                    "wikipedia_url": wiki_url,
                                    "verifiable": data.get('verifiable', 'unknown')
                                }
                            })
        
        # If no evidence found but we have a claim, create a default entry
        if not qrel_entries and 'claim' in data:
            # Create a general document reference
            doc_id = f"fever_{file_name}_{line_num}"
            relevance = 1 if label in ["SUPPORTS", "REFUTES"] else 0
            
            qrel_entries.append({
                "query_id": query_id,
                "doc_id": doc_id,
                "relevance": relevance,
                "metadata": {
                    "fever_label": label,
                    "evidence_type": "general",
                    "verifiable": data.get('verifiable', 'unknown')
                }
            })
        
        return qrel_entries
    
    def get_metadata(self) -> Dict[str, Any]:
        """Get metadata information for the converted dataset."""
        base_metadata = super().get_metadata()
        
        base_metadata.update({
            "dataset_name": "fever",
            "description": "FEVER: Fact Extraction and VERification dataset for fact-checking claims against Wikipedia",
            "original_format": "JSONL with claims, evidence, and verification labels",
            "claim_types": ["factual", "numerical", "temporal", "geographical"],
            "labels": ["SUPPORTS", "REFUTES", "NOT ENOUGH INFO"],
            "language": "English",
            "domain": "General knowledge (Wikipedia)",
            "task_type": "fact_verification",
            "evidence_source": "wikipedia",
            "splits": ["train", "dev", "test"]
        })
        
        return base_metadata