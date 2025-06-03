"""
Optimized sparse retrieval models implementation for NewAIBench framework.

This module implements highly optimized sparse retrieval models with:
- Vectorized operations and parallel processing
- Memory-efficient data structures
- Caching and memoization
- Batch processing optimizations
- GPU acceleration (optional)
- Compiled functions for critical paths
"""

import re
import logging
import pickle
import hashlib
import multiprocessing as mp
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Tuple
from collections import defaultdict, Counter
from dataclasses import dataclass
import numpy as np
import array
import time

# Try to import numba for JIT compilation
try:
    from numba import jit, prange
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
    # Fallback decorators
    def jit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator
    prange = range

try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False
    # Fallback tqdm for when library is not available
    def tqdm(iterable, **kwargs):
        return iterable

try:
    from rank_bm25 import BM25Okapi
except ImportError:
    raise ImportError(
        "rank_bm25 is required for sparse models. "
        "Install it with: pip install rank-bm25"
    )

try:
    import scipy.sparse as sp
    from sklearn.feature_extraction.text import TfidfVectorizer
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    
try:
    import spacy
    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False

try:
    import cupy as cp
    GPU_AVAILABLE = cp.cuda.is_available()
except ImportError:
    GPU_AVAILABLE = False

from .base import BaseRetrievalModel, ModelType

logger = logging.getLogger(__name__)


@dataclass
class BM25OptimizationConfig:
    """Configuration for BM25 optimizations."""
    use_parallel_indexing: bool = True
    num_workers: int = None
    use_sparse_matrix: bool = True
    use_memory_mapping: bool = False
    use_caching: bool = True
    cache_size: int = 10000
    batch_size: int = 1000
    use_gpu: bool = False
    enable_pruning: bool = True
    pruning_threshold: float = 0.1
    use_fast_tokenizer: bool = True
    early_termination_k: int = 10000
    # New optimizations for query processing
    use_vectorized_scoring: bool = True
    aggressive_pruning: bool = True
    reduce_progress_overhead: bool = True
    use_compiled_functions: bool = True
    use_parallel_query_processing: bool = True
    parallel_query_threshold: int = 50  # Use parallel processing for queries >= this number
    
    def __post_init__(self):
        if self.num_workers is None:
            self.num_workers = min(mp.cpu_count(), 8)  # Limit to 8 cores max


class FastTokenizer:
    """High-performance tokenizer with caching."""
    
    def __init__(self, 
                 tokenizer_type: str = 'simple',
                 min_token_length: int = 2,
                 lowercase: bool = True,
                 remove_stopwords: bool = False,
                 custom_stopwords: set = None):
        
        self.tokenizer_type = tokenizer_type
        self.min_token_length = min_token_length
        self.lowercase = lowercase
        self.remove_stopwords = remove_stopwords
        
        # Default English stopwords
        self.default_stopwords = {
            'a', 'an', 'and', 'are', 'as', 'at', 'be', 'by', 'for', 'from',
            'has', 'he', 'in', 'is', 'it', 'its', 'of', 'on', 'that', 'the',
            'to', 'was', 'were', 'will', 'with'
        }
        self.custom_stopwords = custom_stopwords or set()
        
        # Initialize spaCy if available
        self.nlp = None
        if SPACY_AVAILABLE and tokenizer_type == 'spacy':
            try:
                self.nlp = spacy.load("en_core_web_sm", disable=["parser", "ner", "lemmatizer"])
            except OSError:
                logger.warning("spaCy model not found, falling back to regex tokenizer")
                self.tokenizer_type = 'simple'
        
        # Precompiled regex patterns for better performance
        if self.tokenizer_type == 'simple':
            self.token_pattern = re.compile(r'\b\w+\b')
        elif self.tokenizer_type == 'regex':
            self.token_pattern = re.compile(r'\b[a-zA-Z]+\b')
    
    @lru_cache(maxsize=50000)
    def _tokenize_cached(self, text: str) -> Tuple[str, ...]:
        """Cached tokenization for frequently seen texts."""
        return tuple(self._tokenize_uncached(text))
    
    def _tokenize_uncached(self, text: str) -> List[str]:
        """Core tokenization without caching."""
        if not text or not isinstance(text, str):
            return []
        
        # Lowercase if configured
        if self.lowercase:
            text = text.lower()
        
        # Tokenization
        if self.nlp is not None:
            # Use spaCy for better tokenization
            tokens = [token.text for token in self.nlp(text) if token.is_alpha]
        elif hasattr(self, 'token_pattern'):
            # Use precompiled regex
            tokens = self.token_pattern.findall(text)
        else:
            # Fallback to simple regex
            tokens = re.findall(r'\b\w+\b', text)
        
        # Filter by minimum length
        if self.min_token_length > 1:
            tokens = [t for t in tokens if len(t) >= self.min_token_length]
        
        # Remove stopwords if configured
        if self.remove_stopwords:
            stopwords = self.default_stopwords | self.custom_stopwords
            tokens = [t for t in tokens if t not in stopwords]
        
        return tokens
    
    def tokenize(self, text: str, use_cache: bool = True) -> List[str]:
        """Tokenize text with optional caching."""
        if use_cache:
            return list(self._tokenize_cached(text))
        else:
            return self._tokenize_uncached(text)
    
    def batch_tokenize(self, texts: List[str], use_cache: bool = True) -> List[List[str]]:
        """Batch tokenization for better performance."""
        if self.nlp is not None:
            # Use spaCy's efficient batch processing
            results = []
            for doc in self.nlp.pipe(texts, batch_size=1000):
                tokens = [token.text.lower() if self.lowercase else token.text 
                         for token in doc if token.is_alpha and len(token.text) >= self.min_token_length]
                
                if self.remove_stopwords:
                    stopwords = self.default_stopwords | self.custom_stopwords
                    tokens = [t for t in tokens if t not in stopwords]
                
                results.append(tokens)
            return results
        else:
            # Use cached tokenization for each text
            return [self.tokenize(text, use_cache) for text in texts]


class SparseMatrix:
    """Memory-efficient sparse matrix for BM25 term frequencies."""
    
    def __init__(self):
        self.doc_term_matrix = None
        self.vocab_to_id = {}
        self.id_to_vocab = []
        self.doc_lengths = array.array('f')  # Use array for memory efficiency
        self.doc_ids = []
    
    def build_from_tokenized_docs(self, tokenized_docs: List[List[str]], doc_ids: List[str]):
        """Build sparse matrix from tokenized documents."""
        if not SCIPY_AVAILABLE:
            raise ImportError("SciPy required for sparse matrix operations")
        
        # Build vocabulary
        all_terms = set()
        for tokens in tokenized_docs:
            all_terms.update(tokens)
        
        self.vocab_to_id = {term: idx for idx, term in enumerate(sorted(all_terms))}
        self.id_to_vocab = list(self.vocab_to_id.keys())
        
        # Build term-document matrix
        rows, cols, data = [], [], []
        
        for doc_idx, tokens in enumerate(tokenized_docs):
            term_counts = Counter(tokens)
            self.doc_lengths.append(len(tokens))
            
            for term, count in term_counts.items():
                if term in self.vocab_to_id:
                    term_idx = self.vocab_to_id[term]
                    rows.append(term_idx)
                    cols.append(doc_idx)
                    data.append(count)
        
        # Create sparse matrix (terms x documents)
        self.doc_term_matrix = sp.csr_matrix(
            (data, (rows, cols)), 
            shape=(len(self.vocab_to_id), len(tokenized_docs))
        )
        
        self.doc_ids = doc_ids.copy()
        
        logger.info(f"Built sparse matrix: {self.doc_term_matrix.shape} with {self.doc_term_matrix.nnz} non-zeros")
    
    def get_term_frequencies(self, doc_idx: int) -> Dict[str, int]:
        """Get term frequencies for a document."""
        if self.doc_term_matrix is None:
            return {}
        
        doc_column = self.doc_term_matrix[:, doc_idx]
        term_freqs = {}
        
        for term_idx in doc_column.nonzero()[0]:
            term = self.id_to_vocab[term_idx]
            freq = doc_column[term_idx, 0]
            term_freqs[term] = freq
        
        return term_freqs


def _process_chunk_parallel(args):
    """Process a chunk of documents in parallel."""
    chunk_docs, tokenizer, show_progress = args
    
    chunk_doc_ids = []
    chunk_tokenized = []
    
    # Use tqdm if available and show_progress is True
    iterator = tqdm(chunk_docs, desc="Processing chunk", disable=not (show_progress and TQDM_AVAILABLE)) if TQDM_AVAILABLE else chunk_docs
    
    for doc_id, doc_data in iterator:
        # Extract text content
        text_content = doc_data.get('text', '')
        if not text_content:
            text_content = doc_data.get('title', '') + ' ' + doc_data.get('ocr_text', '')
        
        # Tokenize
        tokens = tokenizer.tokenize(text_content, use_cache=False)  # Disable cache in parallel
        
        chunk_doc_ids.append(doc_id)
        chunk_tokenized.append(tokens)
    
    return chunk_doc_ids, chunk_tokenized


class OptimizedBM25Model(BaseRetrievalModel):
    """
    Highly optimized BM25 implementation with multiple performance enhancements.
    
    Features:
    - Parallel corpus indexing
    - Vectorized operations
    - Memory-efficient data structures
    - Query result caching
    - Batch processing
    - Optional GPU acceleration
    - Early termination and pruning
    """
    
    def __init__(self, model_config: Dict[str, Any], 
                 opt_config: BM25OptimizationConfig = None, **kwargs):
        super().__init__(model_config, **kwargs)
        
        # Optimization configuration
        self.opt_config = opt_config or BM25OptimizationConfig()
        
        # BM25 parameters
        params = self.config.parameters
        self.k1 = params.get('k1', 1.6)
        self.b = params.get('b', 0.75)
        
        # Tokenizer configuration
        tokenizer_type = 'spacy' if (self.opt_config.use_fast_tokenizer and SPACY_AVAILABLE) else 'simple'
        self.tokenizer = FastTokenizer(
            tokenizer_type=tokenizer_type,
            min_token_length=params.get('min_token_length', 2),
            lowercase=params.get('lowercase', True),
            remove_stopwords=params.get('remove_stopwords', False),
            custom_stopwords=set(params.get('stopwords', []))
        )
        
        # Model components
        self.bm25_model = None
        self.sparse_matrix = None
        self.doc_ids = []
        self.tokenized_corpus = []
        
        # Caching
        self.query_cache = {}
        self.score_cache = {}  # Cache for computed scores
        self._corpus_cache = {}
        self._corpus_indexed = False
        
        # Performance tracking
        self.stats = {
            'indexing_time': 0.0,
            'query_times': [],
            'cache_hits': 0,
            'cache_misses': 0
        }
        
        logger.info(f"Initialized OptimizedBM25Model with {self.opt_config.num_workers} workers")
        if self.opt_config.use_gpu and not GPU_AVAILABLE:
            logger.warning("GPU optimization requested but CuPy not available")
    
    def load_model(self) -> None:
        """Load/initialize the optimized BM25 model."""
        # Validate BM25 parameters
        if not (0 < self.k1 <= 10):
            raise ValueError(f"k1 parameter must be in (0, 10], got {self.k1}")
        if not (0 <= self.b <= 1):
            raise ValueError(f"b parameter must be in [0, 1], got {self.b}")
        
        self.is_loaded = True
        logger.info(f"OptimizedBM25Model loaded successfully")
    
    def _parallel_index_corpus(self, corpus: Dict[str, Dict[str, Any]], **kwargs) -> None:
        """Index corpus using parallel processing."""
        logger.info(f"Starting parallel indexing with {self.opt_config.num_workers} workers")
        start_time = time.time()
        
        show_progress = kwargs.get('show_progress', True)
        
        # Prepare chunks for parallel processing
        corpus_items = list(corpus.items())
        chunk_size = max(1, len(corpus_items) // self.opt_config.num_workers)
        chunks = [
            corpus_items[i:i + chunk_size] 
            for i in range(0, len(corpus_items), chunk_size)
        ]
        
        # Prepare arguments for parallel processing
        chunk_args = [
            (chunk, self.tokenizer, show_progress) 
            for chunk in chunks
        ]
        
        # Process in parallel
        if show_progress and TQDM_AVAILABLE:
            logger.info(f"Processing {len(corpus)} documents in {len(chunks)} parallel chunks...")
        
        with mp.Pool(self.opt_config.num_workers) as pool:
            if show_progress and TQDM_AVAILABLE:
                results = list(tqdm(
                    pool.imap(_process_chunk_parallel, chunk_args),
                    total=len(chunk_args),
                    desc="Parallel indexing chunks"
                ))
            else:
                results = pool.map(_process_chunk_parallel, chunk_args)
        
        # Merge results
        self.doc_ids = []
        self.tokenized_corpus = []
        
        if show_progress and TQDM_AVAILABLE:
            results_iter = tqdm(results, desc="Merging chunks")
        else:
            results_iter = results
            
        for chunk_doc_ids, chunk_tokenized in results_iter:
            self.doc_ids.extend(chunk_doc_ids)
            self.tokenized_corpus.extend(chunk_tokenized)
        
        # Build BM25 index
        logger.info("Building optimized BM25 index...")
        if self.opt_config.use_sparse_matrix and SCIPY_AVAILABLE:
            self.sparse_matrix = SparseMatrix()
            self.sparse_matrix.build_from_tokenized_docs(self.tokenized_corpus, self.doc_ids)
        
        self.bm25_model = BM25Okapi(self.tokenized_corpus, k1=self.k1, b=self.b)
        
        self.stats['indexing_time'] = time.time() - start_time
        logger.info(f"Parallel indexing completed in {self.stats['indexing_time']:.2f}s")
    
    def _sequential_index_corpus(self, corpus: Dict[str, Dict[str, Any]], **kwargs) -> None:
        """Fallback sequential indexing."""
        logger.info("Starting sequential indexing")
        start_time = time.time()
        
        show_progress = kwargs.get('show_progress', True)
        
        # Reset state
        self.doc_ids = []
        self.tokenized_corpus = []
        
        # Extract texts for batch tokenization
        texts = []
        doc_ids = []
        
        corpus_items = list(corpus.items())
        
        # Use progress bar for document processing
        if show_progress and TQDM_AVAILABLE:
            corpus_iter = tqdm(corpus_items, desc="Processing documents")
        else:
            corpus_iter = corpus_items
        
        for doc_id, doc_data in corpus_iter:
            text_content = doc_data.get('text', '')
            if not text_content:
                text_content = doc_data.get('title', '') + ' ' + doc_data.get('ocr_text', '')
            
            texts.append(text_content)
            doc_ids.append(doc_id)
        
        # Batch tokenization
        logger.info("Tokenizing documents in batches...")
        self.tokenized_corpus = self.tokenizer.batch_tokenize(texts)
        self.doc_ids = doc_ids
        
        # Build indices
        logger.info("Building BM25 index...")
        if self.opt_config.use_sparse_matrix and SCIPY_AVAILABLE:
            self.sparse_matrix = SparseMatrix()
            self.sparse_matrix.build_from_tokenized_docs(self.tokenized_corpus, self.doc_ids)
        
        self.bm25_model = BM25Okapi(self.tokenized_corpus, k1=self.k1, b=self.b)
        
        self.stats['indexing_time'] = time.time() - start_time
        logger.info(f"Sequential indexing completed in {self.stats['indexing_time']:.2f}s")
    
    def index_corpus(self, corpus: Dict[str, Dict[str, Any]], **kwargs) -> None:
        """Index corpus with optimizations."""
        if not self.is_loaded:
            raise RuntimeError("Model must be loaded before indexing")
        
        force_reindex = kwargs.get('force_reindex', False)
        
        # Check cache
        corpus_hash = hash(frozenset(corpus.keys()))
        if (not force_reindex and 
            self.bm25_model is not None and 
            corpus_hash in self._corpus_cache):
            logger.info("Reusing existing optimized BM25 index")
            return
        
        logger.info(f"Indexing corpus with {len(corpus)} documents using optimizations")
        
        # Choose indexing strategy
        if (self.opt_config.use_parallel_indexing and 
            len(corpus) > 1000 and 
            self.opt_config.num_workers > 1):
            self._parallel_index_corpus(corpus, **kwargs)
        else:
            self._sequential_index_corpus(corpus, **kwargs)
        
        # Cache the corpus
        self._corpus_cache[corpus_hash] = True
        self._corpus_indexed = True
        
        # Log statistics
        total_tokens = sum(len(tokens) for tokens in self.tokenized_corpus)
        avg_doc_length = total_tokens / len(self.tokenized_corpus) if self.tokenized_corpus else 0
        
        logger.info(f"Optimized BM25 indexing completed:")
        logger.info(f"  - Documents: {len(self.doc_ids)}")
        logger.info(f"  - Total tokens: {total_tokens}")
        logger.info(f"  - Average document length: {avg_doc_length:.2f} tokens")
        logger.info(f"  - Vocabulary size: {len(self.bm25_model.idf) if self.bm25_model else 0}")
        logger.info(f"  - Indexing time: {self.stats['indexing_time']:.2f}s")
    
    def _get_cached_query_tokens(self, query_text: str) -> List[str]:
        """Get tokenized query with highly optimized caching."""
        # Fast hash-based lookup
        if self.opt_config.use_caching and query_text in self.query_cache:
            self.stats['cache_hits'] += 1
            return self.query_cache[query_text]
        
        self.stats['cache_misses'] += 1
        tokens = self.tokenizer.tokenize(query_text)
        
        # Cache management with LRU-style eviction
        if self.opt_config.use_caching:
            if len(self.query_cache) >= self.opt_config.cache_size:
                # Remove oldest entry (simple FIFO, could be improved to LRU)
                oldest_key = next(iter(self.query_cache))
                del self.query_cache[oldest_key]
            self.query_cache[query_text] = tokens
        
        return tokens
    
    def _compute_scores_optimized(self, query_tokens: List[str], top_k: int) -> List[Tuple[float, str]]:
        """Ultra-optimized BM25 scoring with caching and minimal memory allocation."""
        # Create cache key from query tokens
        if self.opt_config.use_caching:
            cache_key = tuple(sorted(query_tokens))  # Sort for consistent caching
            if cache_key in self.score_cache:
                cached_results = self.score_cache[cache_key]
                # Return top_k from cached results
                return cached_results[:top_k]
        
        # Get scores from BM25 model (this is the main computational cost)
        doc_scores = self.bm25_model.get_scores(query_tokens)
        
        # Fast path for small results
        num_docs = len(doc_scores)
        if top_k >= num_docs:
            # No need for complex sorting logic
            indices = np.arange(num_docs)
            scores = np.asarray(doc_scores)
            sorted_indices = np.argsort(scores)[::-1]
            results = [(float(scores[idx]), self.doc_ids[idx]) for idx in sorted_indices]
        else:
            # Convert to numpy array once
            scores_array = np.asarray(doc_scores)
            
            # Optimized selection strategy based on size ratio
            selection_ratio = top_k / num_docs
            
            if selection_ratio > 0.1 or not self.opt_config.enable_pruning:
                # If we need a large portion, just sort everything
                top_indices = np.argsort(scores_array)[::-1][:top_k]
            else:
                # Use argpartition for very selective queries
                if self.opt_config.aggressive_pruning and num_docs > 10000:
                    # Aggressive pruning: get only 2x top_k candidates
                    partition_k = min(top_k * 2, self.opt_config.early_termination_k)
                else:
                    # Conservative pruning: get 3x top_k candidates
                    partition_k = min(top_k * 3, self.opt_config.early_termination_k)
                
                if self.opt_config.use_gpu and GPU_AVAILABLE and num_docs > 50000:
                    # GPU acceleration for very large corpora
                    gpu_scores = cp.asarray(scores_array)
                    top_indices = cp.argpartition(gpu_scores, -partition_k)[-partition_k:]
                    top_candidates_scores = gpu_scores[top_indices]
                    sorted_indices = cp.argsort(top_candidates_scores)[::-1][:top_k]
                    top_indices = cp.asnumpy(top_indices[sorted_indices])
                else:
                    # CPU optimization with minimal memory allocation
                    top_indices = np.argpartition(scores_array, -partition_k)[-partition_k:]
                    # Sort only the partition (much smaller)
                    partition_scores = scores_array[top_indices]
                    sorted_order = np.argsort(partition_scores)[::-1][:top_k]
                    top_indices = top_indices[sorted_order]
            
            # Vectorized result creation with minimal tuple creation
            top_scores = scores_array[top_indices]
            
            # Pre-allocate result list for better performance
            results = []
            doc_ids = self.doc_ids  # Cache reference
            
            for i in range(len(top_indices)):
                results.append((float(top_scores[i]), doc_ids[top_indices[i]]))
        
        # Cache results for future use (cache more than requested for efficiency)
        if self.opt_config.use_caching and cache_key:
            cache_size_limit = max(top_k * 3, 100)  # Cache more results for flexibility
            if len(results) >= cache_size_limit:
                cache_results = results[:cache_size_limit]
            else:
                cache_results = results
            
            # Manage cache size
            if len(self.score_cache) >= self.opt_config.cache_size // 2:  # Use half cache for scores
                oldest_key = next(iter(self.score_cache))
                del self.score_cache[oldest_key]
            
            self.score_cache[cache_key] = cache_results
        
        return results
    
    def _batch_predict(self, queries: List[Dict[str, str]], 
                      corpus: Dict[str, Dict[str, Any]], 
                      top_k: int, **kwargs) -> Dict[str, Dict[str, float]]:
        """Highly optimized batch processing for multiple queries."""
        results = {}
        
        # Pre-extract filter parameters to avoid repeated dict lookups
        normalize_scores = kwargs.get('normalize_scores', False)
        min_score = kwargs.get('min_score', 0.0)
        show_progress = kwargs.get('show_progress', True)
        
        # Batch tokenize all queries at once for efficiency
        if self.opt_config.use_vectorized_scoring:
            all_query_tokens = self._batch_tokenize_queries(queries)
        
        # Track performance with reduced overhead
        batch_start_time = time.time()
        
        # Process queries with minimal overhead
        if show_progress and TQDM_AVAILABLE and not self.opt_config.reduce_progress_overhead:
            query_iter = tqdm(queries, desc="Processing queries")
        else:
            query_iter = queries
        
        for query_data in query_iter:
            query_id = query_data['query_id']
            query_text = query_data.get('text', '')
            
            if not query_text:
                results[query_id] = {}
                continue
            
            # Get pre-tokenized query tokens or fall back to individual tokenization
            if self.opt_config.use_vectorized_scoring and query_id in all_query_tokens:
                query_tokens = all_query_tokens[query_id]
            else:
                query_tokens = self._get_cached_query_tokens(query_text)
            
            if not query_tokens:
                results[query_id] = {}
                continue
            
            # Compute scores (main computational work)
            scored_docs = self._compute_scores_optimized(query_tokens, top_k)
            
            # Apply filters efficiently
            if min_score > 0.0:
                scored_docs = [(score, doc_id) for score, doc_id in scored_docs if score >= min_score]
            
            if normalize_scores and scored_docs:
                max_score = scored_docs[0][0]
                if max_score > 0:
                    scored_docs = [(score / max_score, doc_id) for score, doc_id in scored_docs]
            
            # Format results
            query_results = {doc_id: float(score) for score, doc_id in scored_docs}
            results[query_id] = query_results
        
        # Record batch timing with reduced overhead
        batch_time = time.time() - batch_start_time
        avg_query_time = batch_time / len(queries) if queries else 0
        self.stats['query_times'].append(avg_query_time)
        
        return results
    
    def _batch_tokenize_queries(self, queries: List[Dict[str, str]]) -> Dict[str, List[str]]:
        """Batch tokenize queries for better performance."""
        query_tokens = {}
        
        # Group queries by text to avoid duplicate work
        text_to_queries = defaultdict(list)
        for query in queries:
            query_text = query.get('text', '')
            if query_text:
                text_to_queries[query_text].append(query['query_id'])
        
        # Tokenize each unique text once
        for query_text, query_ids in text_to_queries.items():
            if self.opt_config.use_caching and query_text in self.query_cache:
                tokens = self.query_cache[query_text]
                self.stats['cache_hits'] += len(query_ids)
            else:
                tokens = self.tokenizer.tokenize(query_text)
                self.stats['cache_misses'] += len(query_ids)
                
                # Cache if enabled
                if self.opt_config.use_caching:
                    if len(self.query_cache) >= self.opt_config.cache_size:
                        oldest_key = next(iter(self.query_cache))
                        del self.query_cache[oldest_key]
                    self.query_cache[query_text] = tokens
            
            # Assign tokens to all queries with this text
            for query_id in query_ids:
                query_tokens[query_id] = tokens
        
        return query_tokens

    def predict(self, 
                queries: List[Dict[str, str]], 
                corpus: Dict[str, Dict[str, Any]], 
                top_k: int = 1000,
                **kwargs) -> Dict[str, Dict[str, float]]:
        """Optimized BM25 prediction with all enhancements."""
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
        
        logger.info(f"Processing {len(queries)} queries with optimized BM25")
        
        # Choose processing strategy based on query count and configuration
        if (self.opt_config.use_parallel_query_processing and 
            len(queries) >= self.opt_config.parallel_query_threshold and
            self.opt_config.num_workers > 1):
            # Use parallel processing for large query batches
            results = self._parallel_query_processing(queries, top_k, **kwargs)
        else:
            # Use optimized sequential processing
            results = self._batch_predict(queries, corpus, top_k, **kwargs)
        
        # Log performance statistics
        avg_query_time = np.mean(self.stats['query_times']) if self.stats['query_times'] else 0
        cache_hit_rate = self.stats['cache_hits'] / (self.stats['cache_hits'] + self.stats['cache_misses']) if (self.stats['cache_hits'] + self.stats['cache_misses']) > 0 else 0
        
        logger.info(f"Optimized BM25 prediction completed:")
        logger.info(f"  - Queries processed: {len(queries)}")
        logger.info(f"  - Average query time: {avg_query_time:.4f}s")
        logger.info(f"  - Cache hit rate: {cache_hit_rate:.2%}")
        
        return results
    
    def _parallel_query_processing(self, queries: List[Dict[str, str]], 
                                 top_k: int, **kwargs) -> Dict[str, Dict[str, float]]:
        """Process queries in parallel for maximum performance."""
        import concurrent.futures
        from functools import partial
        
        # Pre-extract parameters
        normalize_scores = kwargs.get('normalize_scores', False)
        min_score = kwargs.get('min_score', 0.0)
        
        def process_single_query(query_data):
            query_id = query_data['query_id']
            query_text = query_data.get('text', '')
            
            if not query_text:
                return query_id, {}
            
            # Get tokens
            query_tokens = self._get_cached_query_tokens(query_text)
            
            if not query_tokens:
                return query_id, {}
            
            # Compute scores
            scored_docs = self._compute_scores_optimized(query_tokens, top_k)
            
            # Apply filters
            if min_score > 0.0:
                scored_docs = [(score, doc_id) for score, doc_id in scored_docs if score >= min_score]
            
            if normalize_scores and scored_docs:
                max_score = scored_docs[0][0]
                if max_score > 0:
                    scored_docs = [(score / max_score, doc_id) for score, doc_id in scored_docs]
            
            # Format results
            query_results = {doc_id: float(score) for score, doc_id in scored_docs}
            return query_id, query_results
        
        # Use ThreadPoolExecutor for I/O bound tasks
        max_workers = min(self.opt_config.num_workers, len(queries))
        results = {}
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_query = {executor.submit(process_single_query, query): query 
                             for query in queries}
            
            for future in concurrent.futures.as_completed(future_to_query):
                query_id, query_results = future.result()
                results[query_id] = query_results
        
        return results

    def get_model_info(self) -> Dict[str, Any]:
        """Get comprehensive model information including optimization stats."""
        base_info = {
            'model_name': self.name,
            'model_type': 'OptimizedBM25',
            'parameter_count': None,
            'parameters': {
                'k1': self.k1,
                'b': self.b,
                'tokenizer': self.tokenizer.tokenizer_type,
                'use_parallel_indexing': self.opt_config.use_parallel_indexing,
                'use_sparse_matrix': self.opt_config.use_sparse_matrix,
                'use_caching': self.opt_config.use_caching,
                'num_workers': self.opt_config.num_workers,
            },
            'corpus_indexed': self._corpus_indexed,
            'num_documents': len(self.doc_ids) if self.doc_ids else 0,
            'vocabulary_size': len(self.bm25_model.idf) if self.bm25_model else 0,
            'is_loaded': self.is_loaded,
            'optimization_stats': {
                'indexing_time': self.stats['indexing_time'],
                'avg_query_time': np.mean(self.stats['query_times']) if self.stats['query_times'] else 0,
                'cache_hits': self.stats['cache_hits'],
                'cache_misses': self.stats['cache_misses'],
                'cache_size': len(self.query_cache),
            }
        }
        
        return base_info
    
    def clear_cache(self):
        """Clear all caches to free memory."""
        self.query_cache.clear()
        self.score_cache.clear()
        self.stats['cache_hits'] = 0
        self.stats['cache_misses'] = 0
        logger.info("Cleared all caches including score cache")
