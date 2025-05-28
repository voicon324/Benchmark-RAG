"""
HotpotQA dataset converter for NewAIBench.

HotpotQA is a multi-hop question answering dataset that requires reasoning
over multiple supporting documents to answer complex questions.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Set

from .base_converter import BaseDatasetConverter, ConversionConfig

logger = logging.getLogger(__name__)


class HotpotQAConverter(BaseDatasetConverter):
    """
    Converter for HotpotQA dataset.
    
    HotpotQA contains questions that require multi-hop reasoning over
    multiple Wikipedia paragraphs. Each example includes:
    - A complex question requiring multi-hop reasoning
    - Supporting paragraphs from Wikipedia
    - Answer annotations with supporting facts
    
    The converter handles both fullwiki and distractor settings.
    """
    
    def __init__(self, config: ConversionConfig):
        super().__init__(config)
        self.dataset_name = "hotpotqa"
        
    def detect_format(self) -> bool:
        """
        Detect if the input directory contains HotpotQA data.
        
        Returns:
            bool: True if HotpotQA format is detected
        """
        input_path = Path(self.config.input_path)
        
        # Check for standard HotpotQA files
        hotpot_patterns = [
            "**/hotpot_*.json",
            "**/train*.json",
            "**/dev*.json", 
            "**/test*.json",
            "**/fullwiki*.json",
            "**/distractor*.json"
        ]
        
        for pattern in hotpot_patterns:
            if list(input_path.glob(pattern)):
                logger.info(f"Detected HotpotQA format with pattern: {pattern}")
                return True
                
        # Check for HotpotQA specific structure
        if self._check_hotpot_structure(input_path):
            return True
            
        return False
    
    def _check_hotpot_structure(self, path: Path) -> bool:
        """Check if files contain HotpotQA structure."""
        for file_path in path.rglob("*.json"):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    
                    # Check if it's a list of HotpotQA examples
                    if isinstance(data, list) and len(data) > 0:
                        sample = data[0]
                        if self._is_hotpot_format(sample):
                            return True
                            
            except Exception:
                continue
        return False
    
    def _is_hotpot_format(self, data: Dict) -> bool:
        """Check if data structure matches HotpotQA format."""
        # HotpotQA format indicators
        hotpot_fields = {
            'question', 'answer', 'supporting_facts', 'context',
            '_id', 'type', 'level'
        }
        
        data_fields = set(data.keys())
        
        # Check for HotpotQA format (need at least 4 core fields)
        if len(hotpot_fields.intersection(data_fields)) >= 4:
            return True
            
        # Check for context structure (list of [title, sentences])
        if 'context' in data and isinstance(data['context'], list):
            if len(data['context']) > 0 and isinstance(data['context'][0], list):
                if len(data['context'][0]) >= 2:
                    return True
                    
        return False
    
    def convert_corpus(self) -> Path:
        """
        Convert HotpotQA context documents to NewAIBench corpus format.
        
        Returns:
            Path: Path to the converted corpus file
        """
        logger.info("Converting HotpotQA corpus...")
        
        corpus_file = self.config.output_path / "corpus.jsonl"
        input_path = Path(self.config.input_path)
        
        seen_docs = set()
        doc_count = 0
        
        with open(corpus_file, 'w', encoding='utf-8') as out_f:
            
            # Process all JSON files
            for file_path in input_path.rglob("*.json"):
                logger.info(f"Processing file: {file_path}")
                
                try:
                    with open(file_path, 'r', encoding='utf-8') as in_f:
                        data = json.load(in_f)
                        
                    # Handle list of examples
                    if isinstance(data, list):
                        examples = data
                    else:
                        examples = [data]
                        
                    for example in examples:
                        # Extract documents from context
                        docs = self._extract_documents_from_context(example)
                        
                        for doc_id, doc_content in docs:
                            if doc_id not in seen_docs:
                                seen_docs.add(doc_id)
                                doc_count += 1
                                
                                corpus_entry = {
                                    "_id": doc_id,
                                    "title": doc_content["title"],
                                    "text": doc_content["text"],
                                    "metadata": {
                                        "source": "hotpotqa",
                                        "file": file_path.name,
                                        **doc_content.get("metadata", {})
                                    }
                                }
                                
                                out_f.write(json.dumps(corpus_entry) + '\n')
                                
                except Exception as e:
                    logger.warning(f"Error processing file {file_path}: {e}")
                    continue
        
        logger.info(f"Converted {doc_count} documents to corpus")
        self._update_stats("corpus_documents", doc_count)
        
        return corpus_file
    
    def _extract_documents_from_context(self, example: Dict) -> List[Tuple[str, Dict]]:
        """Extract documents from HotpotQA context."""
        documents = []
        
        if 'context' not in example:
            return documents
            
        context = example['context']
        
        for i, context_item in enumerate(context):
            if not isinstance(context_item, list) or len(context_item) < 2:
                continue
                
            title = context_item[0]
            sentences = context_item[1]
            
            # Create document ID from title
            doc_id = f"hotpot_{self._normalize_title(title)}"
            
            # Join sentences into text
            if isinstance(sentences, list):
                text = ' '.join(sentences)
            else:
                text = str(sentences)
            
            doc_content = {
                "title": title,
                "text": text,
                "metadata": {
                    "sentence_count": len(sentences) if isinstance(sentences, list) else 1,
                    "context_index": i
                }
            }
            
            documents.append((doc_id, doc_content))
        
        return documents
    
    def _normalize_title(self, title: str) -> str:
        """Normalize title for use as document ID."""
        # Remove special characters and replace spaces with underscores
        import re
        normalized = re.sub(r'[^\w\s-]', '', title)
        normalized = re.sub(r'\s+', '_', normalized)
        return normalized.lower()
    
    def convert_queries(self) -> Path:
        """
        Convert HotpotQA questions to NewAIBench queries format.
        
        Returns:
            Path: Path to the converted queries file
        """
        logger.info("Converting HotpotQA queries...")
        
        queries_file = self.config.output_path / "queries.jsonl"
        input_path = Path(self.config.input_path)
        
        query_count = 0
        
        with open(queries_file, 'w', encoding='utf-8') as out_f:
            
            for file_path in input_path.rglob("*.json"):
                logger.info(f"Processing queries from: {file_path}")
                
                try:
                    with open(file_path, 'r', encoding='utf-8') as in_f:
                        data = json.load(in_f)
                        
                    # Handle list of examples
                    if isinstance(data, list):
                        examples = data
                    else:
                        examples = [data]
                        
                    for example in examples:
                        query_entry = self._extract_query(example, file_path.stem)
                        
                        if query_entry:
                            out_f.write(json.dumps(query_entry) + '\n')
                            query_count += 1
                            
                except Exception as e:
                    logger.warning(f"Error processing file {file_path}: {e}")
                    continue
        
        logger.info(f"Converted {query_count} queries")
        self._update_stats("queries", query_count)
        
        return queries_file
    
    def _extract_query(self, example: Dict, file_name: str) -> Optional[Dict]:
        """Extract query from a HotpotQA example."""
        
        if 'question' not in example:
            return None
            
        # Use provided ID or generate one
        query_id = example.get('_id', f"hotpot_{file_name}_{hash(example['question']) % 1000000}")
        
        return {
            "_id": str(query_id),
            "text": example['question'],
            "metadata": {
                "source": "hotpotqa",
                "file": file_name,
                "question_type": example.get('type', 'unknown'),
                "difficulty_level": example.get('level', 'unknown'),
                "answer": example.get('answer', '')
            }
        }
    
    def convert_qrels(self) -> Path:
        """
        Convert HotpotQA supporting facts to NewAIBench qrels format.
        
        Returns:
            Path: Path to the converted qrels file
        """
        logger.info("Converting HotpotQA qrels...")
        
        qrels_file = self.config.output_path / "qrels.jsonl"
        input_path = Path(self.config.input_path)
        
        qrel_count = 0
        
        with open(qrels_file, 'w', encoding='utf-8') as out_f:
            
            for file_path in input_path.rglob("*.json"):
                logger.info(f"Processing qrels from: {file_path}")
                
                try:
                    with open(file_path, 'r', encoding='utf-8') as in_f:
                        data = json.load(in_f)
                        
                    # Handle list of examples
                    if isinstance(data, list):
                        examples = data
                    else:
                        examples = [data]
                        
                    for example in examples:
                        qrel_entries = self._extract_qrels(example, file_path.stem)
                        
                        for qrel_entry in qrel_entries:
                            out_f.write(json.dumps(qrel_entry) + '\n')
                            qrel_count += 1
                            
                except Exception as e:
                    logger.warning(f"Error processing file {file_path}: {e}")
                    continue
        
        logger.info(f"Converted {qrel_count} qrels")
        self._update_stats("qrels", qrel_count)
        
        return qrels_file
    
    def _extract_qrels(self, example: Dict, file_name: str) -> List[Dict]:
        """Extract qrels from a HotpotQA example."""
        qrel_entries = []
        
        if 'question' not in example or 'context' not in example:
            return qrel_entries
            
        # Get query ID
        query_id = example.get('_id', f"hotpot_{file_name}_{hash(example['question']) % 1000000}")
        query_id = str(query_id)
        
        # Get supporting facts
        supporting_facts = example.get('supporting_facts', [])
        supporting_titles = set()
        
        for fact in supporting_facts:
            if isinstance(fact, list) and len(fact) >= 1:
                supporting_titles.add(fact[0])  # fact[0] is the title
        
        # Create qrels for all documents in context
        context = example.get('context', [])
        
        for context_item in context:
            if not isinstance(context_item, list) or len(context_item) < 2:
                continue
                
            title = context_item[0]
            doc_id = f"hotpot_{self._normalize_title(title)}"
            
            # Determine relevance based on supporting facts
            if title in supporting_titles:
                relevance = 2  # High relevance (supporting fact)
            else:
                relevance = 0  # Not relevant (distractor)
            
            qrel_entries.append({
                "query_id": query_id,
                "doc_id": doc_id,
                "relevance": relevance,
                "metadata": {
                    "is_supporting": title in supporting_titles,
                    "title": title,
                    "question_type": example.get('type', 'unknown'),
                    "difficulty_level": example.get('level', 'unknown')
                }
            })
        
        return qrel_entries
    
    def get_metadata(self) -> Dict[str, Any]:
        """Get metadata information for the converted dataset."""
        base_metadata = super().get_metadata()
        
        base_metadata.update({
            "dataset_name": "hotpotqa",
            "description": "HotpotQA: Multi-hop question answering dataset requiring reasoning over multiple documents",
            "original_format": "JSON with questions, context paragraphs, and supporting facts",
            "question_types": ["bridge", "comparison"],
            "reasoning_type": "multi_hop",
            "language": "English",
            "domain": "General knowledge (Wikipedia)",
            "task_type": "multi_hop_qa",
            "settings": ["fullwiki", "distractor"],
            "splits": ["train", "dev", "test"]
        })
        
        return base_metadata