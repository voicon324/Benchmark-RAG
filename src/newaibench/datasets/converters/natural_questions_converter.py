"""
Natural Questions dataset converter for NewAIBench.

This converter handles the Natural Questions dataset, which consists of real questions
from Google search and Wikipedia articles as context. The dataset includes both
short and long answers extracted from Wikipedia passages.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Set

from .base_converter import BaseDatasetConverter, ConversionConfig

logger = logging.getLogger(__name__)


class NaturalQuestionsConverter(BaseDatasetConverter):
    """
    Converter for Natural Questions dataset.
    
    Natural Questions contains real questions from Google search paired with
    Wikipedia articles. Each example has:
    - A question from real users
    - A Wikipedia article as context
    - Annotations for short and long answers
    
    The converter supports both the original JSONL format and simplified formats.
    """
    
    def __init__(self, config: ConversionConfig):
        super().__init__(config)
        self.dataset_name = "natural_questions"
        
    def detect_format(self) -> bool:
        """
        Detect if the input directory contains Natural Questions data.
        
        Returns:
            bool: True if Natural Questions format is detected
        """
        input_path = Path(self.config.input_path)
        
        # Check for standard Natural Questions files
        nq_patterns = [
            "**/nq-*.jsonl",
            "**/natural_questions*.jsonl", 
            "**/simplified*.jsonl",
            "**/train.jsonl",
            "**/dev.jsonl",
            "**/test.jsonl"
        ]
        
        for pattern in nq_patterns:
            if list(input_path.glob(pattern)):
                logger.info(f"Detected Natural Questions format with pattern: {pattern}")
                return True
                
        # Check for Natural Questions specific structure
        if self._check_nq_structure(input_path):
            return True
            
        return False
    
    def _check_nq_structure(self, path: Path) -> bool:
        """Check if files contain Natural Questions structure."""
        for file_path in path.rglob("*.jsonl"):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    # Check first few lines for NQ structure
                    for i, line in enumerate(f):
                        if i >= 3:  # Check first 3 lines
                            break
                        try:
                            data = json.loads(line.strip())
                            # Look for Natural Questions specific fields
                            if self._is_nq_format(data):
                                return True
                        except json.JSONDecodeError:
                            continue
            except Exception:
                continue
        return False
    
    def _is_nq_format(self, data: Dict) -> bool:
        """Check if data structure matches Natural Questions format."""
        # Original NQ format indicators
        nq_fields = {
            'question_text', 'document_text', 'annotations',
            'document_tokens', 'question_tokens'
        }
        
        # Simplified NQ format indicators  
        simplified_fields = {
            'question', 'title', 'context', 'answer'
        }
        
        data_fields = set(data.keys())
        
        # Check for original format
        if len(nq_fields.intersection(data_fields)) >= 3:
            return True
            
        # Check for simplified format
        if len(simplified_fields.intersection(data_fields)) >= 3:
            return True
            
        return False
    
    def convert_corpus(self) -> Path:
        """
        Convert Natural Questions documents to NewAIBench corpus format.
        
        Returns:
            Path: Path to the converted corpus file
        """
        logger.info("Converting Natural Questions corpus...")
        
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
                            
                            # Extract document based on format
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
                                            "source": "natural_questions",
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
        """Extract document(s) from a Natural Questions entry."""
        documents = []
        
        # Handle original NQ format
        if 'document_text' in data and 'document_title' in data:
            doc_id = f"nq_{file_name}_{line_num}"
            
            # Use example_id if available
            if 'example_id' in data:
                doc_id = f"nq_{data['example_id']}"
            
            doc_content = {
                "title": data.get('document_title', ''),
                "text": data['document_text'],
                "metadata": {
                    "document_url": data.get('document_url', ''),
                    "long_answer_candidates": data.get('long_answer_candidates', [])
                }
            }
            documents.append((doc_id, doc_content))
            
        # Handle simplified format
        elif 'context' in data:
            doc_id = f"nq_{file_name}_{line_num}"
            
            # Use id if available
            if 'id' in data:
                doc_id = f"nq_{data['id']}"
            
            doc_content = {
                "title": data.get('title', ''),
                "text": data['context'],
                "metadata": {}
            }
            documents.append((doc_id, doc_content))
            
        # Handle other formats
        elif 'text' in data or 'passage' in data:
            doc_id = f"nq_{file_name}_{line_num}"
            
            if 'id' in data:
                doc_id = f"nq_{data['id']}"
                
            text = data.get('text', data.get('passage', ''))
            doc_content = {
                "title": data.get('title', ''),
                "text": text,
                "metadata": {}
            }
            documents.append((doc_id, doc_content))
        
        return documents
    
    def convert_queries(self) -> Path:
        """
        Convert Natural Questions questions to NewAIBench queries format.
        
        Returns:
            Path: Path to the converted queries file
        """
        logger.info("Converting Natural Questions queries...")
        
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
                            
                            # Extract query
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
        """Extract query from a Natural Questions entry."""
        
        # Handle original NQ format
        if 'question_text' in data:
            query_id = f"nq_{file_name}_{line_num}"
            
            if 'example_id' in data:
                query_id = f"nq_{data['example_id']}"
                
            return {
                "_id": query_id,
                "text": data['question_text'],
                "metadata": {
                    "source": "natural_questions",
                    "file": file_name
                }
            }
            
        # Handle simplified format
        elif 'question' in data:
            query_id = f"nq_{file_name}_{line_num}"
            
            if 'id' in data:
                query_id = f"nq_{data['id']}"
                
            return {
                "_id": query_id,
                "text": data['question'],
                "metadata": {
                    "source": "natural_questions",
                    "file": file_name
                }
            }
        
        return None
    
    def convert_qrels(self) -> Path:
        """
        Convert Natural Questions annotations to NewAIBench qrels format.
        
        Returns:
            Path: Path to the converted qrels file
        """
        logger.info("Converting Natural Questions qrels...")
        
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
                            
                            # Extract qrels
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
        """Extract qrels from a Natural Questions entry."""
        qrel_entries = []
        
        # Generate IDs
        query_id = f"nq_{file_name}_{line_num}"
        doc_id = f"nq_{file_name}_{line_num}"
        
        if 'example_id' in data:
            query_id = f"nq_{data['example_id']}"
            doc_id = f"nq_{data['example_id']}"
        elif 'id' in data:
            query_id = f"nq_{data['id']}"
            doc_id = f"nq_{data['id']}"
        
        # Handle original NQ format with annotations
        if 'annotations' in data:
            for annotation in data['annotations']:
                # Check for short answer
                if annotation.get('short_answers'):
                    relevance = 2  # High relevance for short answers
                elif annotation.get('long_answer', {}).get('candidate_index', -1) >= 0:
                    relevance = 1  # Medium relevance for long answers
                else:
                    relevance = 0  # No answer
                
                qrel_entries.append({
                    "query_id": query_id,
                    "doc_id": doc_id,
                    "relevance": relevance,
                    "metadata": {
                        "annotation_type": "natural_questions",
                        "has_short_answer": bool(annotation.get('short_answers')),
                        "has_long_answer": annotation.get('long_answer', {}).get('candidate_index', -1) >= 0
                    }
                })
        
        # Handle simplified format
        elif 'answer' in data:
            # If answer exists, assume relevant
            relevance = 1 if data['answer'].strip() else 0
            
            qrel_entries.append({
                "query_id": query_id,
                "doc_id": doc_id,
                "relevance": relevance,
                "metadata": {
                    "annotation_type": "simplified",
                    "answer": data['answer']
                }
            })
        
        # Default: assume document is relevant to its question
        else:
            qrel_entries.append({
                "query_id": query_id,
                "doc_id": doc_id,
                "relevance": 1,
                "metadata": {
                    "annotation_type": "default"
                }
            })
        
        return qrel_entries
    
    def get_metadata(self) -> Dict[str, Any]:
        """Get metadata information for the converted dataset."""
        base_metadata = super().get_metadata()
        
        base_metadata.update({
            "dataset_name": "natural_questions",
            "description": "Natural Questions: Real questions from Google search with Wikipedia context",
            "original_format": "JSONL with questions, documents, and annotations",
            "question_types": ["factoid", "entity", "description"],
            "answer_types": ["short_answer", "long_answer", "no_answer"],
            "language": "English",
            "domain": "General knowledge (Wikipedia)",
            "task_type": "question_answering",
            "splits": ["train", "dev", "test"]
        })
        
        return base_metadata