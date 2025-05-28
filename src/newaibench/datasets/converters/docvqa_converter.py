"""
DocVQA dataset converter for NewAIBench.

DocVQA (Document Visual Question Answering) is a dataset containing
document images with question-answer pairs that can be adapted for
document retrieval tasks.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Set

from .base_converter import BaseDatasetConverter, ConversionConfig

logger = logging.getLogger(__name__)


class DocVQAConverter(BaseDatasetConverter):
    """
    Converter for DocVQA dataset to document retrieval format.
    
    DocVQA contains document images with question-answer pairs.
    For retrieval adaptation:
    - Questions become queries
    - Document images become corpus items
    - Relevance is inferred from question-document associations
    """
    
    def __init__(self, config: ConversionConfig):
        super().__init__(config)
        self.dataset_name = "docvqa"
        
    def detect_format(self) -> bool:
        """
        Detect if the input directory contains DocVQA data.
        
        Returns:
            bool: True if DocVQA format is detected
        """
        input_path = Path(self.config.input_path)
        
        # Check for DocVQA specific files
        docvqa_patterns = [
            "**/train_v*.json",
            "**/val_v*.json",
            "**/test_v*.json",
            "**/documents.json",
            "**/annotations.json"
        ]
        
        for pattern in docvqa_patterns:
            if list(input_path.glob(pattern)):
                logger.info(f"Detected DocVQA format with pattern: {pattern}")
                return True
                
        # Check for DocVQA specific structure in JSON files
        for json_file in input_path.rglob("*.json"):
            if self._check_docvqa_structure(json_file):
                return True
                
        return False
    
    def _check_docvqa_structure(self, json_file: Path) -> bool:
        """Check if JSON file contains DocVQA structure."""
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # DocVQA data structure indicators
            if isinstance(data, dict):
                if "data" in data:
                    data = data["data"]
                
                # Check for typical DocVQA fields
                if isinstance(data, list) and len(data) > 0:
                    sample = data[0]
                    docvqa_fields = {"questionId", "question", "docId", "image", "answers"}
                    if len(docvqa_fields.intersection(set(sample.keys()))) >= 3:
                        return True
                        
            elif isinstance(data, list) and len(data) > 0:
                sample = data[0]
                docvqa_fields = {"questionId", "question", "docId", "image", "answers"}
                if len(docvqa_fields.intersection(set(sample.keys()))) >= 3:
                    return True
                    
        except Exception:
            pass
            
        return False
    
    def convert_corpus(self) -> Path:
        """
        Convert DocVQA documents to NewAIBench corpus format.
        
        Returns:
            Path: Path to the converted corpus file
        """
        logger.info("Converting DocVQA corpus...")
        
        corpus_file = self.config.output_path / "corpus.jsonl"
        input_path = Path(self.config.input_path)
        
        seen_docs = set()
        doc_count = 0
        
        # Collect all document information
        documents = {}
        ocr_data = {}
        
        # Load OCR data if available
        ocr_file = input_path / "ocr_results.json"
        if ocr_file.exists():
            logger.info("Loading OCR data...")
            with open(ocr_file, 'r', encoding='utf-8') as f:
                ocr_data = json.load(f)
        
        with open(corpus_file, 'w', encoding='utf-8') as out_f:
            
            # Process all JSON files
            for json_file in input_path.rglob("*.json"):
                if json_file.name in ["ocr_results.json", "documents.json"]:
                    continue
                    
                logger.info(f"Processing file: {json_file}")
                
                try:
                    with open(json_file, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    
                    # Handle different data structures
                    if isinstance(data, dict) and "data" in data:
                        data = data["data"]
                    
                    if not isinstance(data, list):
                        continue
                    
                    # Extract documents from questions
                    for item in data:
                        doc_info = self._extract_document_info(item, ocr_data)
                        if doc_info and doc_info["doc_id"] not in seen_docs:
                            documents[doc_info["doc_id"]] = doc_info
                            seen_docs.add(doc_info["doc_id"])
                            
                except Exception as e:
                    logger.warning(f"Error processing {json_file}: {e}")
                    continue
            
            # Write corpus entries
            for doc_id, doc_info in documents.items():
                corpus_entry = {
                    "_id": doc_id,
                    "title": doc_info.get("title", ""),
                    "text": doc_info.get("ocr_text", ""),
                    "image_path": doc_info.get("image_path", ""),
                    "metadata": {
                        "source": "docvqa",
                        "document_type": doc_info.get("document_type", "unknown"),
                        "page_count": doc_info.get("page_count", 1),
                        **doc_info.get("metadata", {})
                    }
                }
                
                out_f.write(json.dumps(corpus_entry, ensure_ascii=False) + '\n')
                doc_count += 1
        
        logger.info(f"Converted {doc_count} documents to corpus")
        self._update_stats("corpus_documents", doc_count)
        
        return corpus_file
    
    def _extract_document_info(self, item: Dict, ocr_data: Dict) -> Optional[Dict]:
        """Extract document information from DocVQA item."""
        try:
            doc_id = item.get("docId") or item.get("doc_id") or item.get("document_id")
            if not doc_id:
                return None
            
            # Get image path
            image_path = item.get("image", "")
            if not image_path:
                image_path = f"documents/{doc_id}.png"  # Default pattern
            
            # Get OCR text
            ocr_text = ""
            if str(doc_id) in ocr_data:
                # OCR data might be in different formats
                ocr_info = ocr_data[str(doc_id)]
                if isinstance(ocr_info, str):
                    ocr_text = ocr_info
                elif isinstance(ocr_info, dict):
                    ocr_text = ocr_info.get("text", "") or ocr_info.get("ocr_text", "")
                elif isinstance(ocr_info, list):
                    # List of OCR results
                    text_parts = []
                    for ocr_item in ocr_info:
                        if isinstance(ocr_item, dict):
                            text_parts.append(ocr_item.get("text", ""))
                        elif isinstance(ocr_item, str):
                            text_parts.append(ocr_item)
                    ocr_text = " ".join(text_parts)
            
            # Extract title from image path or use document ID
            title = Path(image_path).stem if image_path else f"Document {doc_id}"
            
            return {
                "doc_id": str(doc_id),
                "title": title,
                "image_path": image_path,
                "ocr_text": ocr_text.strip(),
                "document_type": item.get("document_type", "document"),
                "metadata": {
                    "original_docId": doc_id
                }
            }
            
        except Exception as e:
            logger.warning(f"Error extracting document info: {e}")
            return None
    
    def convert_queries(self) -> Path:
        """
        Convert DocVQA questions to NewAIBench queries format.
        
        Returns:
            Path: Path to the converted queries file
        """
        logger.info("Converting DocVQA queries...")
        
        queries_file = self.config.output_path / "queries.jsonl"
        input_path = Path(self.config.input_path)
        
        query_count = 0
        
        with open(queries_file, 'w', encoding='utf-8') as out_f:
            
            for json_file in input_path.rglob("*.json"):
                if json_file.name in ["ocr_results.json", "documents.json"]:
                    continue
                    
                logger.info(f"Processing queries from: {json_file}")
                
                try:
                    with open(json_file, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    
                    # Handle different data structures
                    if isinstance(data, dict) and "data" in data:
                        data = data["data"]
                    
                    if not isinstance(data, list):
                        continue
                    
                    # Extract queries from questions
                    for item in data:
                        query_entry = self._extract_query(item)
                        if query_entry:
                            out_f.write(json.dumps(query_entry, ensure_ascii=False) + '\n')
                            query_count += 1
                            
                except Exception as e:
                    logger.warning(f"Error processing queries from {json_file}: {e}")
                    continue
        
        logger.info(f"Converted {query_count} queries")
        self._update_stats("queries", query_count)
        
        return queries_file
    
    def _extract_query(self, item: Dict) -> Optional[Dict]:
        """Extract query from a DocVQA item."""
        try:
            question_id = item.get("questionId") or item.get("question_id") or item.get("qid")
            question = item.get("question")
            
            if not question_id or not question:
                return None
            
            return {
                "_id": str(question_id),
                "text": question.strip(),
                "metadata": {
                    "source": "docvqa",
                    "original_questionId": question_id,
                    "docId": item.get("docId") or item.get("doc_id"),
                    "answers": item.get("answers", [])
                }
            }
            
        except Exception as e:
            logger.warning(f"Error extracting query: {e}")
            return None
    
    def convert_qrels(self) -> Path:
        """
        Convert DocVQA question-document associations to NewAIBench qrels format.
        
        Returns:
            Path: Path to the converted qrels file
        """
        logger.info("Converting DocVQA qrels...")
        
        qrels_file = self.config.output_path / "qrels.jsonl"
        input_path = Path(self.config.input_path)
        
        qrel_count = 0
        
        with open(qrels_file, 'w', encoding='utf-8') as out_f:
            
            for json_file in input_path.rglob("*.json"):
                if json_file.name in ["ocr_results.json", "documents.json"]:
                    continue
                    
                logger.info(f"Processing qrels from: {json_file}")
                
                try:
                    with open(json_file, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    
                    # Handle different data structures
                    if isinstance(data, dict) and "data" in data:
                        data = data["data"]
                    
                    if not isinstance(data, list):
                        continue
                    
                    # Extract qrels from question-document pairs
                    for item in data:
                        qrel_entry = self._extract_qrel(item)
                        if qrel_entry:
                            out_f.write(json.dumps(qrel_entry, ensure_ascii=False) + '\n')
                            qrel_count += 1
                            
                except Exception as e:
                    logger.warning(f"Error processing qrels from {json_file}: {e}")
                    continue
        
        logger.info(f"Converted {qrel_count} qrels")
        self._update_stats("qrels", qrel_count)
        
        return qrels_file
    
    def _extract_qrel(self, item: Dict) -> Optional[Dict]:
        """Extract qrel from a DocVQA item."""
        try:
            question_id = item.get("questionId") or item.get("question_id") or item.get("qid")
            doc_id = item.get("docId") or item.get("doc_id") or item.get("document_id")
            
            if not question_id or not doc_id:
                return None
            
            # Determine relevance based on answer availability and quality
            answers = item.get("answers", [])
            
            # High relevance if there are valid answers
            if answers and len(answers) > 0:
                # Check if answers are meaningful (not "unanswerable", etc.)
                valid_answers = [
                    ans for ans in answers 
                    if isinstance(ans, str) and ans.lower() not in ["unanswerable", "no", "unknown", ""]
                ]
                relevance = 2 if valid_answers else 1
            else:
                relevance = 1  # Medium relevance - question exists but no clear answer
            
            return {
                "query_id": str(question_id),
                "doc_id": str(doc_id),
                "relevance": relevance,
                "metadata": {
                    "source": "docvqa",
                    "answers": answers,
                    "question": item.get("question", "")
                }
            }
            
        except Exception as e:
            logger.warning(f"Error extracting qrel: {e}")
            return None
    
    def get_metadata(self) -> Dict[str, Any]:
        """Get metadata information for the converted dataset."""
        base_metadata = super().get_metadata()
        
        base_metadata.update({
            "dataset_name": "docvqa",
            "description": "DocVQA: Document Visual Question Answering dataset adapted for document retrieval",
            "original_format": "JSON with question-answer pairs and document images",
            "task_type": "document_visual_question_answering",
            "domain": "Business documents, forms, reports",
            "language": "English",
            "document_types": ["forms", "reports", "tables", "invoices", "letters"],
            "adaptation_note": "Questions converted to queries, document associations to qrels",
            "relevance_scale": "0-2 (0: not relevant, 1: relevant, 2: highly relevant)",
            "splits": ["train", "val", "test"]
        })
        
        return base_metadata
