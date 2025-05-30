"""
Sparse retrieval models implementation for NewAIBench framework.

This module implements traditional sparse retrieval models like BM25,
which use term-based matching and statistical scoring functions.
"""

import re
import logging
from typing import Dict, List, Any, Optional, Union
from collections import defaultdict
import numpy as np

try:
    from rank_bm25 import BM25Okapi
except ImportError:
    raise ImportError(
        "rank_bm25 is required for sparse models. "
        "Install it with: pip install rank-bm25"
    )

from .base import BaseRetrievalModel, ModelType

logger = logging.getLogger(__name__)


class BM25Model(BaseRetrievalModel):
    """
    Okapi BM25 sparse retrieval model implementation.
    
    This implementation uses the rank_bm25 library to provide fast BM25 scoring
    for text retrieval tasks. It supports:
    
    - Configurable BM25 parameters (k1, b)
    - Automatic corpus indexing and caching
    - Batch query processing
    - Text preprocessing and tokenization
    - Top-k retrieval with score normalization
    
    Example:
        >>> config = {
        ...     "name": "bm25_baseline",
        ...     "type": "sparse",
        ...     "parameters": {
        ...         "k1": 1.6,
        ...         "b": 0.75,
        ...         "tokenizer": "simple"
        ...     }
        ... }
        >>> model = BM25Model(config)
        >>> model.load_model()
        >>> model.index_corpus(corpus)
        >>> results = model.predict(queries, corpus, top_k=10)
    """
    
    def __init__(self, model_config: Dict[str, Any], **kwargs):
        """
        Initialize BM25 model with configuration.
        
        Args:
            model_config: Configuration dictionary containing:
                - k1 (float): BM25 k1 parameter (default: 1.6)
                - b (float): BM25 b parameter (default: 0.75) 
                - tokenizer (str): Tokenization method ('simple', 'regex') (default: 'simple')
                - min_token_length (int): Minimum token length (default: 2)
                - lowercase (bool): Convert to lowercase (default: True)
                - remove_stopwords (bool): Remove stopwords (default: False)
                - stopwords (List[str]): Custom stopwords list
            **kwargs: Additional arguments
        """
        super().__init__(model_config, **kwargs)
        
        # BM25 specific parameters from config
        params = self.config.parameters
        self.k1 = params.get('k1', 1.6)
        self.b = params.get('b', 0.75)
        self.tokenizer_type = params.get('tokenizer', 'simple')
        self.min_token_length = params.get('min_token_length', 2)
        self.lowercase = params.get('lowercase', True)
        self.remove_stopwords = params.get('remove_stopwords', False)
        self.custom_stopwords = set(params.get('stopwords', []))
        
        # Default English stopwords (simple set)
        self.default_stopwords = {
            'a', 'an', 'and', 'are', 'as', 'at', 'be', 'by', 'for', 'from',
            'has', 'he', 'in', 'is', 'it', 'its', 'of', 'on', 'that', 'the',
            'to', 'was', 'were', 'will', 'with'
        }
        
        # Initialize model components
        self.bm25_model = None
        self.doc_ids = []  # Maintain document ID order
        self.tokenized_corpus = []
        self._corpus_cache = {}  # Cache for avoiding re-indexing same corpus
        
        logger.info(f"Initialized BM25Model with k1={self.k1}, b={self.b}")
    
    def _tokenize_text(self, text: str) -> List[str]:
        """
        Tokenize text according to configured tokenization method.
        
        Args:
            text: Input text to tokenize
            
        Returns:
            List of tokens
        """
        if not text or not isinstance(text, str):
            return []
        
        # Lowercase if configured
        if self.lowercase:
            text = text.lower()
        
        # Tokenization
        if self.tokenizer_type == 'simple':
            # Simple whitespace + punctuation tokenization
            tokens = re.findall(r'\b\w+\b', text)
        elif self.tokenizer_type == 'regex':
            # More sophisticated regex tokenization
            tokens = re.findall(r'\b[a-zA-Z]+\b', text)
        else:
            # Fallback to simple
            tokens = re.findall(r'\b\w+\b', text)
        
        # Filter by minimum length
        tokens = [t for t in tokens if len(t) >= self.min_token_length]
        
        # Remove stopwords if configured
        if self.remove_stopwords:
            stopwords = self.default_stopwords | self.custom_stopwords
            tokens = [t for t in tokens if t not in stopwords]
        
        return tokens
    
    def load_model(self) -> None:
        """
        Load/initialize the BM25 model.
        
        For BM25, this mainly validates configuration and sets the loaded flag.
        The actual BM25 model is created during corpus indexing.
        """
        # Validate BM25 parameters
        if not (0 < self.k1 <= 10):
            raise ValueError(f"k1 parameter must be in (0, 10], got {self.k1}")
        if not (0 <= self.b <= 1):
            raise ValueError(f"b parameter must be in [0, 1], got {self.b}")
        
        self.is_loaded = True
        logger.info(f"BM25Model {self.name} loaded successfully")
    
    def index_corpus(self, corpus: Dict[str, Dict[str, Any]], **kwargs) -> None:
        """
        Index the corpus for BM25 retrieval.
        
        This method processes the corpus, tokenizes all documents, and builds
        the BM25 index for fast retrieval.
        
        Args:
            corpus: Dictionary of documents with format:
                {'doc_id': {'text': 'document content', ...}}
            **kwargs: Additional parameters:
                - force_reindex (bool): Force rebuilding even if cached
                - show_progress (bool): Show indexing progress
        """
        if not self.is_loaded:
            raise RuntimeError("Model must be loaded before indexing")
        
        force_reindex = kwargs.get('force_reindex', False)
        show_progress = kwargs.get('show_progress', True)
        
        # Generate corpus hash for caching
        corpus_hash = hash(frozenset(corpus.keys()))
        
        # Check if we can reuse existing index
        if (not force_reindex and 
            self.bm25_model is not None and 
            corpus_hash in self._corpus_cache):
            logger.info("Reusing existing BM25 index")
            return
        
        logger.info(f"Indexing corpus with {len(corpus)} documents")
        
        # Reset state
        self.doc_ids = []
        self.tokenized_corpus = []
        
        # Process each document
        for doc_id, doc_data in corpus.items():
            # Extract text content
            text_content = doc_data.get('text', '')
            if not text_content:
                # Try alternative text fields
                text_content = doc_data.get('title', '') + ' ' + doc_data.get('ocr_text', '')
            
            # Tokenize document
            tokens = self._tokenize_text(text_content)
            
            # Store document ID and tokens
            self.doc_ids.append(doc_id)
            self.tokenized_corpus.append(tokens)
            
            if show_progress and len(self.doc_ids) % 1000 == 0:
                logger.info(f"Processed {len(self.doc_ids)} documents")
        
        # Build BM25 index
        logger.info("Building BM25 index...")
        self.bm25_model = BM25Okapi(self.tokenized_corpus, k1=self.k1, b=self.b)
        
        # Cache the corpus
        self._corpus_cache[corpus_hash] = True
        self._corpus_indexed = True
        
        # Log statistics
        total_tokens = sum(len(tokens) for tokens in self.tokenized_corpus)
        avg_doc_length = total_tokens / len(self.tokenized_corpus) if self.tokenized_corpus else 0
        
        logger.info(f"BM25 indexing completed:")
        logger.info(f"  - Documents: {len(self.doc_ids)}")
        logger.info(f"  - Total tokens: {total_tokens}")
        logger.info(f"  - Average document length: {avg_doc_length:.2f} tokens")
        logger.info(f"  - Vocabulary size: {len(self.bm25_model.idf) if self.bm25_model else 0}")
    
    def predict(self, 
                queries: List[Dict[str, str]], 
                corpus: Dict[str, Dict[str, Any]], 
                top_k: int = 1000,
                **kwargs) -> Dict[str, Dict[str, float]]:
        """
        Perform BM25 retrieval for given queries.
        
        Args:
            queries: List of query dictionaries with 'query_id' and 'text'
            corpus: Document corpus (must match indexed corpus)
            top_k: Number of top documents to return per query
            **kwargs: Additional parameters:
                - normalize_scores (bool): Normalize scores to [0,1] (default: False)
                - min_score (float): Minimum score threshold (default: 0.0)
                - batch_size (int): Query batch size (default: 1)
                
        Returns:
            Dictionary mapping query_id to {doc_id: score}
        """
        # Call parent validation
        super().predict(queries, corpus, top_k, **kwargs)
        
        # Check if corpus is indexed
        if not self._corpus_indexed or self.bm25_model is None:
            logger.info("Corpus not indexed, indexing now...")
            self.index_corpus(corpus, **kwargs)
        
        # Verify corpus consistency
        if set(self.doc_ids) != set(corpus.keys()):
            logger.warning("Corpus has changed since indexing, re-indexing...")
            self.index_corpus(corpus, force_reindex=True, **kwargs)
        
        # Extract parameters
        normalize_scores = kwargs.get('normalize_scores', False)
        min_score = kwargs.get('min_score', 0.0)
        
        results = {}
        
        # Process each query
        for query_data in queries:
            logger.info(f"Processing query: {query_data.get('query_id', 'unknown')}")
            query_id = query_data['query_id']
            query_text = query_data.get('text', '')
            
            if not query_text:
                logger.warning(f"Empty query text for query_id: {query_id}")
                results[query_id] = {}
                continue
            
            # Tokenize query
            query_tokens = self._tokenize_text(query_text)
            
            if not query_tokens:
                logger.warning(f"No valid tokens for query_id: {query_id}")
                results[query_id] = {}
                continue
            
            # Get BM25 scores for all documents
            doc_scores = self.bm25_model.get_scores(query_tokens)
            
            # Create (score, doc_id) pairs and sort by score
            scored_docs = [(score, doc_id) for score, doc_id in zip(doc_scores, self.doc_ids)]
            scored_docs.sort(key=lambda x: x[0], reverse=True)
            
            # Apply minimum score threshold
            if min_score > 0.0:
                scored_docs = [(score, doc_id) for score, doc_id in scored_docs if score >= min_score]
            
            # Take top_k results
            top_docs = scored_docs[:top_k]
            
            # Normalize scores if requested
            if normalize_scores and top_docs:
                max_score = top_docs[0][0]
                if max_score > 0:
                    top_docs = [(score / max_score, doc_id) for score, doc_id in top_docs]
            
            # Format results
            query_results = {doc_id: float(score) for score, doc_id in top_docs}
            results[query_id] = query_results
            
            logger.debug(f"Query {query_id}: retrieved {len(query_results)} documents "
                        f"(max_score: {top_docs[0][0] if top_docs else 0:.4f})")
        
        logger.info(f"BM25 prediction completed for {len(queries)} queries")
        return results
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the current model state.
        
        Returns:
            Dictionary with model information
        """
        # For sparse models like BM25, parameter count is not applicable
        # since they don't have neural network parameters
        return {
            'model_name': self.name,
            'model_type': 'BM25',
            'parameter_count': None,  # N/A for sparse models
            'parameters': {
                'k1': self.k1,
                'b': self.b,
                'tokenizer': self.tokenizer_type,
                'lowercase': self.lowercase,
                'remove_stopwords': self.remove_stopwords
            },
            'corpus_indexed': self._corpus_indexed,
            'num_documents': len(self.doc_ids) if self.doc_ids else 0,
            'vocabulary_size': len(self.bm25_model.idf) if self.bm25_model else 0,
            'is_loaded': self.is_loaded
        }