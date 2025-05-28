"""
BEIR dataset converter for NewAIBench.

This module converts BEIR benchmark datasets to NewAIBench format,
supporting all BEIR datasets with their standardized JSONL format.
"""

import json
import csv
import logging
from pathlib import Path
from typing import Dict, List, Optional, Union, Any, Iterator, Tuple

from .base_converter import BaseDatasetConverter, ConversionConfig, DatasetRegistry

logger = logging.getLogger(__name__)


class BEIRConverter(BaseDatasetConverter):
    """
    Converter for BEIR benchmark datasets.
    
    BEIR datasets follow a standardized format:
    - corpus.jsonl: {"_id": str, "title": str, "text": str}
    - queries.jsonl: {"_id": str, "text": str}  
    - qrels/{split}.tsv: qid \t Q0 \t did \t rel
    
    Supported datasets include:
    - NFCorpus, FiQA, SciFact, TREC-COVID
    - HotpotQA, FEVER, Climate-FEVER
    - And all other BEIR benchmark datasets
    
    Output:
    - corpus.jsonl: Documents in NewAIBench format
    - queries.jsonl: Queries in NewAIBench format
    - qrels.txt: Relevance judgments in TREC format
    """
    
    def __init__(self, config: ConversionConfig):
        """Initialize BEIR converter."""
        super().__init__(config)
        
        # BEIR specific configuration
        self.split = getattr(config, 'split', 'test')  # Default to test split
        
        # Validate BEIR structure
        self._validate_beir_structure()
    
    def _validate_beir_structure(self) -> None:
        """Validate that source follows BEIR structure."""
        required_files = [
            'corpus.jsonl',
            'queries.jsonl'
        ]
        
        for filename in required_files:
            file_path = self.source_path / filename
            if not file_path.exists():
                raise FileNotFoundError(f"Required BEIR file not found: {filename}")
        
        # Check for qrels directory
        qrels_dir = self.source_path / 'qrels'
        if qrels_dir.exists():
            qrels_file = qrels_dir / f'{self.split}.tsv'
            if not qrels_file.exists():
                logger.warning(f"QRels file not found for split '{self.split}': {qrels_file}")
        else:
            logger.warning("QRels directory not found")
    
    def convert_corpus(self) -> Path:
        """Convert BEIR corpus to NewAIBench format."""
        corpus_file = self.source_path / 'corpus.jsonl'
        output_file = self.output_path / 'corpus'
        
        logger.info(f"Converting BEIR corpus from: {corpus_file}")
        
        docs_iter = self._convert_corpus_entries()
        self._write_jsonl(docs_iter, output_file)
        
        output_path = output_file.with_suffix('.jsonl.gz' if self.config.compress_output else '.jsonl')
        logger.info(f"Corpus conversion completed: {output_path}")
        return output_path
    
    def _convert_corpus_entries(self) -> Iterator[Dict[str, Any]]:
        """Convert BEIR corpus entries."""
        corpus_file = self.source_path / 'corpus.jsonl'
        
        with open(corpus_file, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                
                self.stats['docs_processed'] += 1
                
                try:
                    data = json.loads(line)
                    
                    # BEIR format: {"_id": str, "title": str, "text": str}
                    doc_id = str(data.get('_id', ''))
                    title = str(data.get('title', ''))
                    text = str(data.get('text', ''))
                    
                    if not doc_id:
                        logger.warning(f"Missing doc_id at line {line_num + 1}")
                        self.stats['docs_skipped'] += 1
                        continue
                    
                    # Create document in NewAIBench format
                    doc = {
                        'doc_id': doc_id,
                        'text': self._normalize_text(text),
                        'metadata': {
                            'source': 'beir',
                            'dataset': self.config.dataset_name,
                            'original_id': doc_id
                        }
                    }
                    
                    # Add title if available
                    if title:
                        doc['title'] = self._normalize_text(title)
                    
                    # Preserve additional metadata if requested
                    if self.config.preserve_metadata:
                        for key, value in data.items():
                            if key not in ['_id', 'title', 'text']:
                                doc['metadata'][f'beir_{key}'] = value
                    
                    # Validate and yield
                    if self._validate_document(doc):
                        self.stats['docs_valid'] += 1
                        yield doc
                    else:
                        self.stats['docs_skipped'] += 1
                        logger.debug(f"Skipped invalid document: {doc_id}")
                
                except json.JSONDecodeError as e:
                    logger.warning(f"Invalid JSON at line {line_num + 1}: {e}")
                    self.stats['docs_skipped'] += 1
                
                # Progress logging
                if line_num % 10000 == 0 and line_num > 0:
                    logger.debug(f"Processed {line_num} corpus entries")
    
    def convert_queries(self) -> Path:
        """Convert BEIR queries to NewAIBench format."""
        queries_file = self.source_path / 'queries.jsonl'
        output_file = self.output_path / 'queries'
        
        logger.info(f"Converting BEIR queries from: {queries_file}")
        
        queries_iter = self._convert_query_entries()
        self._write_jsonl(queries_iter, output_file)
        
        output_path = output_file.with_suffix('.jsonl.gz' if self.config.compress_output else '.jsonl')
        logger.info(f"Queries conversion completed: {output_path}")
        return output_path
    
    def _convert_query_entries(self) -> Iterator[Dict[str, Any]]:
        """Convert BEIR query entries."""
        queries_file = self.source_path / 'queries.jsonl'
        
        with open(queries_file, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                
                self.stats['queries_processed'] += 1
                
                try:
                    data = json.loads(line)
                    
                    # BEIR format: {"_id": str, "text": str}
                    query_id = str(data.get('_id', ''))
                    text = str(data.get('text', ''))
                    
                    if not query_id:
                        logger.warning(f"Missing query_id at line {line_num + 1}")
                        self.stats['queries_skipped'] += 1
                        continue
                    
                    # Create query in NewAIBench format
                    query = {
                        'query_id': query_id,
                        'text': self._normalize_text(text),
                        'metadata': {
                            'source': 'beir',
                            'dataset': self.config.dataset_name,
                            'split': self.split,
                            'original_id': query_id
                        }
                    }
                    
                    # Preserve additional metadata if requested
                    if self.config.preserve_metadata:
                        for key, value in data.items():
                            if key not in ['_id', 'text']:
                                query['metadata'][f'beir_{key}'] = value
                    
                    # Validate and yield
                    if self._validate_query(query):
                        self.stats['queries_valid'] += 1
                        yield query
                    else:
                        self.stats['queries_skipped'] += 1
                        logger.debug(f"Skipped invalid query: {query_id}")
                
                except json.JSONDecodeError as e:
                    logger.warning(f"Invalid JSON at line {line_num + 1}: {e}")
                    self.stats['queries_skipped'] += 1
                
                # Progress logging
                if line_num % 1000 == 0 and line_num > 0:
                    logger.debug(f"Processed {line_num} query entries")
    
    def convert_qrels(self) -> Path:
        """Convert BEIR qrels to NewAIBench format."""
        # Look for qrels file
        qrels_dir = self.source_path / 'qrels'
        qrels_file = qrels_dir / f'{self.split}.tsv'
        
        output_file = self.output_path / 'qrels.txt'
        
        if not qrels_file.exists():
            logger.warning(f"QRels file not found: {qrels_file}")
            logger.info("Creating empty qrels.txt")
            output_file.touch()
            return output_file
        
        logger.info(f"Converting BEIR qrels from: {qrels_file}")
        
        qrels_data = list(self._parse_qrels(qrels_file))
        self._write_qrels(qrels_data, output_file)
        
        logger.info(f"QRels conversion completed: {output_file}")
        return output_file
    
    def _parse_qrels(self, qrels_file: Path) -> Iterator[Tuple[str, str, int]]:
        """Parse BEIR qrels file."""
        with open(qrels_file, 'r', encoding='utf-8') as f:
            reader = csv.reader(f, delimiter='\t')
            
            for row_num, row in enumerate(reader):
                if len(row) < 4:
                    continue
                
                try:
                    # BEIR qrels format: qid \t Q0 \t did \t rel
                    query_id = str(row[0]).strip()
                    doc_id = str(row[2]).strip()
                    relevance = int(row[3])
                    
                    self.stats['qrels_processed'] += 1
                    yield (query_id, doc_id, relevance)
                
                except (ValueError, IndexError) as e:
                    logger.warning(f"Invalid qrels line {row_num + 1} in {qrels_file}: {e}")
    
    def get_dataset_info(self) -> Dict[str, Any]:
        """Get additional dataset information specific to BEIR."""
        info = super()._create_dataset_info()
        
        # Add BEIR-specific metadata
        beir_info = {
            'beir_split': self.split,
            'beir_format_version': '1.0',
            'supported_splits': self._get_available_splits()
        }
        
        info.update(beir_info)
        return info
    
    def _get_available_splits(self) -> List[str]:
        """Get list of available splits in qrels directory."""
        qrels_dir = self.source_path / 'qrels'
        if not qrels_dir.exists():
            return []
        
        splits = []
        for qrels_file in qrels_dir.glob('*.tsv'):
            split_name = qrels_file.stem
            splits.append(split_name)
        
        return sorted(splits)


class BEIRMultiSplitConverter(BEIRConverter):
    """
    Extended BEIR converter that can process multiple splits at once.
    
    This converter creates separate query/qrels files for each split
    while sharing the same corpus.
    """
    
    def __init__(self, config: ConversionConfig):
        """Initialize multi-split BEIR converter."""
        super().__init__(config)
        self.splits = getattr(config, 'splits', ['test'])
        if isinstance(self.splits, str):
            self.splits = [self.splits]
    
    def convert_all(self) -> Dict[str, Path]:
        """Convert all components for multiple splits."""
        results = {}
        
        # Convert corpus once (shared across splits)
        corpus_path = self.convert_corpus()
        results['corpus'] = corpus_path
        
        # Convert queries and qrels for each split
        for split in self.splits:
            self.split = split
            
            # Convert queries for this split
            queries_path = self.convert_queries()
            results[f'queries_{split}'] = queries_path
            
            # Convert qrels for this split
            qrels_path = self.convert_qrels()
            results[f'qrels_{split}'] = qrels_path
        
        # Create dataset info
        info_path = self._create_dataset_info()
        results['info'] = info_path
        
        return results


# Register the converters
DatasetRegistry.register('beir', BEIRConverter)
DatasetRegistry.register('beir_multi', BEIRMultiSplitConverter)
