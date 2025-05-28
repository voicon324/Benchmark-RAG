"""
TREC dataset converter for NewAIBench.

This module converts TREC Deep Learning Track datasets and other TREC format
datasets to NewAIBench format, supporting various TREC data formats.
"""

import json
import csv
import logging
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Optional, Union, Any, Iterator, Tuple
import re

from .base_converter import BaseDatasetConverter, ConversionConfig, DatasetRegistry

logger = logging.getLogger(__name__)


class TRECConverter(BaseDatasetConverter):
    """
    Converter for TREC format datasets.
    
    Supports conversion from:
    - TREC Deep Learning Track (2019, 2020, 2021)
    - Classic TREC collections (TREC-6, TREC-7, etc.)
    - TREC-COVID dataset
    - Custom TREC format datasets
    
    Input formats:
    - Documents: XML, TXT, or JSONL
    - Queries: XML or TXT format
    - QRels: Standard TREC qrels format
    
    Output:
    - corpus.jsonl: Documents in NewAIBench format
    - queries.jsonl: Queries in NewAIBench format
    - qrels.txt: Relevance judgments in TREC format (preserved)
    """
    
    def __init__(self, config: ConversionConfig):
        """Initialize TREC converter."""
        super().__init__(config)
        
        # TREC specific configuration
        self.year = getattr(config, 'year', None)  # 2019, 2020, 2021 for DL track
        self.track = getattr(config, 'track', 'dl')  # 'dl', 'covid', 'robust', etc.
        self.doc_format = getattr(config, 'doc_format', 'auto')  # 'xml', 'txt', 'jsonl', 'auto'
        
        # File detection
        self.documents_file = None
        self.queries_file = None  
        self.qrels_file = None
        
        self._detect_trec_files()
    
    def _detect_trec_files(self) -> None:
        """Detect TREC format files in source directory."""
        source_files = list(self.source_path.glob('*'))
        
        # Look for document files
        doc_patterns = [
            '*collection*', '*corpus*', '*documents*', '*docs*',
            '*msmarco*', '*.xml', '*.txt'
        ]
        for pattern in doc_patterns:
            matches = list(self.source_path.glob(pattern))
            if matches:
                # Prefer larger files (likely to be the main collection)
                self.documents_file = max(matches, key=lambda x: x.stat().st_size)
                break
        
        # Look for query files
        query_patterns = [
            '*queries*', '*topics*', '*questions*', 
            f'*{self.year}*' if self.year else '*'
        ]
        for pattern in query_patterns:
            matches = list(self.source_path.glob(pattern))
            query_files = [f for f in matches if 'queries' in f.name.lower() or 'topics' in f.name.lower()]
            if query_files:
                self.queries_file = query_files[0]
                break
        
        # Look for qrels files
        qrels_patterns = ['*qrels*', '*relevance*', '*judgments*']
        for pattern in qrels_patterns:
            matches = list(self.source_path.glob(pattern))
            if matches:
                self.qrels_file = matches[0]
                break
        
        logger.info(f"TREC files detected:")
        logger.info(f"  Documents: {self.documents_file}")
        logger.info(f"  Queries: {self.queries_file}")
        logger.info(f"  QRels: {self.qrels_file}")
    
    def convert_corpus(self) -> Path:
        """Convert TREC corpus to NewAIBench format."""
        if not self.documents_file:
            raise FileNotFoundError("TREC documents file not found")
        
        output_file = self.output_path / 'corpus'
        
        logger.info(f"Converting TREC corpus from: {self.documents_file}")
        
        # Detect document format
        if self.doc_format == 'auto':
            self.doc_format = self._detect_document_format()
        
        # Convert based on format
        if self.doc_format == 'xml':
            docs_iter = self._convert_xml_documents()
        elif self.doc_format == 'jsonl':
            docs_iter = self._convert_jsonl_documents()
        elif self.doc_format == 'txt':
            docs_iter = self._convert_txt_documents()
        else:
            raise ValueError(f"Unsupported document format: {self.doc_format}")
        
        # Write output
        self._write_jsonl(docs_iter, output_file)
        
        output_path = output_file.with_suffix('.jsonl.gz' if self.config.compress_output else '.jsonl')
        logger.info(f"Corpus conversion completed: {output_path}")
        return output_path
    
    def _detect_document_format(self) -> str:
        """Auto-detect document format."""
        if self.documents_file.suffix == '.xml':
            return 'xml'
        elif self.documents_file.suffix == '.jsonl':
            return 'jsonl'
        elif self.documents_file.suffix in ['.txt', '.tsv']:
            return 'txt'
        else:
            # Try to detect by content
            with open(self.documents_file, 'r', encoding='utf-8') as f:
                first_line = f.readline().strip()
                if first_line.startswith('<'):
                    return 'xml'
                elif first_line.startswith('{'):
                    return 'jsonl'
                else:
                    return 'txt'
    
    def _convert_xml_documents(self) -> Iterator[Dict[str, Any]]:
        """Convert XML document format (TREC collections)."""
        logger.info("Processing XML document format")
        
        try:
            # Parse XML file
            tree = ET.parse(self.documents_file)
            root = tree.getroot()
            
            # Handle different XML structures
            docs = self._extract_docs_from_xml(root)
            
            for doc_element in docs:
                self.stats['docs_processed'] += 1
                
                doc_data = self._parse_xml_document(doc_element)
                if doc_data:
                    if self._validate_document(doc_data):
                        self.stats['docs_valid'] += 1
                        yield doc_data
                    else:
                        self.stats['docs_skipped'] += 1
        
        except ET.ParseError as e:
            logger.error(f"XML parsing error: {e}")
            raise
    
    def _extract_docs_from_xml(self, root) -> List:
        """Extract document elements from XML root."""
        # Try common TREC XML structures
        patterns = ['DOC', 'document', 'doc', 'DOCUMENT']
        
        for pattern in patterns:
            docs = root.findall(f'.//{pattern}')
            if docs:
                logger.info(f"Found {len(docs)} documents with pattern: {pattern}")
                return docs
        
        # Fallback: treat root children as documents
        return list(root)
    
    def _parse_xml_document(self, doc_element) -> Optional[Dict[str, Any]]:
        """Parse individual XML document element."""
        try:
            # Extract doc ID
            doc_id = None
            for id_field in ['DOCNO', 'docno', 'id', 'ID']:
                id_elem = doc_element.find(id_field)
                if id_elem is not None:
                    doc_id = id_elem.text.strip()
                    break
            
            if not doc_id:
                # Use element attributes as fallback
                doc_id = doc_element.get('id', doc_element.get('docno', ''))
            
            if not doc_id:
                logger.warning("Document without ID found")
                return None
            
            # Extract title
            title = ""
            for title_field in ['TITLE', 'title', 'headline', 'HEADLINE']:
                title_elem = doc_element.find(title_field)
                if title_elem is not None:
                    title = title_elem.text or ""
                    break
            
            # Extract text content
            text_content = ""
            for text_field in ['TEXT', 'text', 'content', 'CONTENT', 'body', 'BODY']:
                text_elem = doc_element.find(text_field)
                if text_elem is not None:
                    text_content = text_elem.text or ""
                    break
            
            # If no specific text field, use all text content
            if not text_content:
                text_content = ET.tostring(doc_element, method='text', encoding='unicode')
            
            # Create document
            doc = {
                'doc_id': doc_id,
                'text': self._normalize_text(text_content),
                'metadata': {
                    'source': 'trec',
                    'track': self.track,
                    'format': 'xml'
                }
            }
            
            if title:
                doc['title'] = self._normalize_text(title)
            
            if self.year:
                doc['metadata']['year'] = self.year
            
            return doc
        
        except Exception as e:
            logger.warning(f"Error parsing XML document: {e}")
            return None
    
    def _convert_jsonl_documents(self) -> Iterator[Dict[str, Any]]:
        """Convert JSONL document format."""
        logger.info("Processing JSONL document format")
        
        with open(self.documents_file, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                
                self.stats['docs_processed'] += 1
                
                try:
                    data = json.loads(line)
                    
                    # Extract fields (flexible field names)
                    doc_id = str(data.get('id', data.get('doc_id', data.get('docid', ''))))
                    text = str(data.get('text', data.get('content', data.get('contents', ''))))
                    title = str(data.get('title', ''))
                    
                    if not doc_id:
                        logger.warning(f"Missing doc_id at line {line_num + 1}")
                        self.stats['docs_skipped'] += 1
                        continue
                    
                    # Create document
                    doc = {
                        'doc_id': doc_id,
                        'text': self._normalize_text(text),
                        'metadata': {
                            'source': 'trec',
                            'track': self.track,
                            'format': 'jsonl'
                        }
                    }
                    
                    if title:
                        doc['title'] = self._normalize_text(title)
                    
                    if self.year:
                        doc['metadata']['year'] = self.year
                    
                    # Preserve additional metadata
                    if self.config.preserve_metadata:
                        for key, value in data.items():
                            if key not in ['id', 'doc_id', 'docid', 'text', 'content', 'contents', 'title']:
                                doc['metadata'][key] = value
                    
                    # Validate and yield
                    if self._validate_document(doc):
                        self.stats['docs_valid'] += 1
                        yield doc
                    else:
                        self.stats['docs_skipped'] += 1
                
                except json.JSONDecodeError as e:
                    logger.warning(f"Invalid JSON at line {line_num + 1}: {e}")
                    self.stats['docs_skipped'] += 1
                
                if line_num % 10000 == 0:
                    logger.debug(f"Processed {line_num} documents")
    
    def _convert_txt_documents(self) -> Iterator[Dict[str, Any]]:
        """Convert plain text or TSV document format."""
        logger.info("Processing TXT/TSV document format")
        
        with open(self.documents_file, 'r', encoding='utf-8') as f:
            # Try to detect delimiter
            first_line = f.readline()
            f.seek(0)
            
            delimiter = '\t' if '\t' in first_line else None
            
            if delimiter:
                reader = csv.reader(f, delimiter=delimiter)
                for row_num, row in enumerate(reader):
                    self.stats['docs_processed'] += 1
                    
                    if len(row) < 2:
                        self.stats['docs_skipped'] += 1
                        continue
                    
                    doc_id = str(row[0]).strip()
                    text = str(row[1]).strip()
                    title = str(row[2]).strip() if len(row) > 2 else ""
                    
                    # Create document
                    doc = {
                        'doc_id': doc_id,
                        'text': self._normalize_text(text),
                        'metadata': {
                            'source': 'trec',
                            'track': self.track,
                            'format': 'tsv'
                        }
                    }
                    
                    if title:
                        doc['title'] = self._normalize_text(title)
                    
                    if self.year:
                        doc['metadata']['year'] = self.year
                    
                    # Validate and yield
                    if self._validate_document(doc):
                        self.stats['docs_valid'] += 1
                        yield doc
                    else:
                        self.stats['docs_skipped'] += 1
            else:
                # Treat as single document
                content = f.read()
                doc = {
                    'doc_id': self.documents_file.stem,
                    'text': self._normalize_text(content),
                    'metadata': {
                        'source': 'trec',
                        'track': self.track,
                        'format': 'txt'
                    }
                }
                
                if self.year:
                    doc['metadata']['year'] = self.year
                
                if self._validate_document(doc):
                    self.stats['docs_valid'] += 1
                    yield doc
                else:
                    self.stats['docs_skipped'] += 1
    
    def convert_queries(self) -> Path:
        """Convert TREC queries to NewAIBench format."""
        if not self.queries_file:
            raise FileNotFoundError("TREC queries file not found")
        
        output_file = self.output_path / 'queries'
        
        logger.info(f"Converting TREC queries from: {self.queries_file}")
        
        # Detect query format
        if self.queries_file.suffix == '.xml':
            queries_iter = self._convert_xml_queries()
        else:
            queries_iter = self._convert_txt_queries()
        
        # Write output
        self._write_jsonl(queries_iter, output_file)
        
        output_path = output_file.with_suffix('.jsonl.gz' if self.config.compress_output else '.jsonl')
        logger.info(f"Queries conversion completed: {output_path}")
        return output_path
    
    def _convert_xml_queries(self) -> Iterator[Dict[str, Any]]:
        """Convert XML query format (TREC topics)."""
        logger.info("Processing XML queries format")
        
        try:
            tree = ET.parse(self.queries_file)
            root = tree.getroot()
            
            # Look for topic/query elements
            topics = root.findall('.//topic') or root.findall('.//TOP') or root.findall('.//query')
            
            for topic in topics:
                self.stats['queries_processed'] += 1
                
                query_data = self._parse_xml_query(topic)
                if query_data:
                    if self._validate_query(query_data):
                        self.stats['queries_valid'] += 1
                        yield query_data
                    else:
                        self.stats['queries_skipped'] += 1
        
        except ET.ParseError as e:
            logger.error(f"XML parsing error: {e}")
            raise
    
    def _parse_xml_query(self, topic_element) -> Optional[Dict[str, Any]]:
        """Parse individual XML query/topic element."""
        try:
            # Extract query ID
            query_id = topic_element.get('number') or topic_element.get('id')
            if not query_id:
                num_elem = topic_element.find('num') or topic_element.find('number')
                if num_elem is not None:
                    query_id = num_elem.text.strip()
            
            if not query_id:
                return None
            
            # Clean query ID (remove prefixes like "Number: ")
            query_id = re.sub(r'^(Number|Topic):\s*', '', query_id).strip()
            
            # Extract query text (try different fields)
            query_text = ""
            for text_field in ['title', 'TITLE', 'text', 'TEXT', 'query', 'QUERY']:
                text_elem = topic_element.find(text_field)
                if text_elem is not None and text_elem.text:
                    query_text = text_elem.text.strip()
                    break
            
            # If no specific field, try description
            if not query_text:
                desc_elem = topic_element.find('desc') or topic_element.find('description')
                if desc_elem is not None and desc_elem.text:
                    query_text = desc_elem.text.strip()
            
            if not query_text:
                return None
            
            # Create query
            query = {
                'query_id': str(query_id),
                'text': self._normalize_text(query_text),
                'metadata': {
                    'source': 'trec',
                    'track': self.track,
                    'format': 'xml'
                }
            }
            
            if self.year:
                query['metadata']['year'] = self.year
            
            return query
        
        except Exception as e:
            logger.warning(f"Error parsing XML query: {e}")
            return None
    
    def _convert_txt_queries(self) -> Iterator[Dict[str, Any]]:
        """Convert plain text or TSV query format."""
        logger.info("Processing TXT queries format")
        
        with open(self.queries_file, 'r', encoding='utf-8') as f:
            # Try to detect format
            first_line = f.readline()
            f.seek(0)
            
            if '\t' in first_line:
                # TSV format: qid \t query
                reader = csv.reader(f, delimiter='\t')
                for row in reader:
                    if len(row) < 2:
                        continue
                    
                    self.stats['queries_processed'] += 1
                    
                    query_id = str(row[0]).strip()
                    text = str(row[1]).strip()
                    
                    query = {
                        'query_id': query_id,
                        'text': self._normalize_text(text),
                        'metadata': {
                            'source': 'trec',
                            'track': self.track,
                            'format': 'tsv'
                        }
                    }
                    
                    if self.year:
                        query['metadata']['year'] = self.year
                    
                    if self._validate_query(query):
                        self.stats['queries_valid'] += 1
                        yield query
                    else:
                        self.stats['queries_skipped'] += 1
    
    def convert_qrels(self) -> Path:
        """Convert TREC qrels to NewAIBench format."""
        output_file = self.output_path / 'qrels.txt'
        
        if not self.qrels_file:
            logger.warning("No qrels file found, creating empty qrels.txt")
            output_file.touch()
            return output_file
        
        logger.info(f"Converting TREC qrels from: {self.qrels_file}")
        
        # TREC qrels are already in the correct format, just copy and validate
        qrels_data = list(self._parse_qrels())
        self._write_qrels(qrels_data, output_file)
        
        logger.info(f"QRels conversion completed: {output_file}")
        return output_file
    
    def _parse_qrels(self) -> Iterator[Tuple[str, str, int]]:
        """Parse TREC qrels file."""
        with open(self.qrels_file, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f):
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                
                parts = line.split()
                if len(parts) < 4:
                    continue
                
                try:
                    # TREC qrels format: qid iteration docid relevance
                    query_id = parts[0]
                    doc_id = parts[2]
                    relevance = int(parts[3])
                    
                    self.stats['qrels_processed'] += 1
                    yield (query_id, doc_id, relevance)
                
                except (ValueError, IndexError) as e:
                    logger.warning(f"Invalid qrels line {line_num + 1}: {e}")


# Register the converter
DatasetRegistry.register('trec', TRECConverter)
