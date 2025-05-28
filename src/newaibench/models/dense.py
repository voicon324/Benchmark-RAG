"""
Dense retrieval models implementation for NewAIBench framework.

This module implements dense retrieval models that use neural networks to generate
embeddings for queries and documents, then perform similarity-based retrieval.
Supports both single-encoder and dual-encoder architectures with options for
brute-force and ANN (Approximate Nearest Neighbor) search.
"""

import logging
import numpy as np
import torch
from typing import Dict, List, Any, Optional, Union, Tuple
from pathlib import Path
import pickle
import os
from abc import ABC, abstractmethod

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    raise ImportError(
        "sentence-transformers is required for dense models. "
        "Install it with: pip install sentence-transformers"
    )

try:
    from transformers import AutoTokenizer, AutoModel
except ImportError:
    raise ImportError(
        "transformers is required for dense models. "
        "Install it with: pip install transformers"
    )

# Optional ANN libraries
try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False

try:
    import hnswlib
    HNSWLIB_AVAILABLE = True
except ImportError:
    HNSWLIB_AVAILABLE = False

from .base import BaseRetrievalModel, ModelType
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger(__name__)


class DenseTextRetriever(BaseRetrievalModel):
    """
    Base class for dense text retrieval models.
    
    This class provides a unified interface for dense retrieval using neural embeddings.
    It supports various architectures (single-encoder, dual-encoder) and indexing
    strategies (brute-force, FAISS, HNSWLIB).
    
    Key Features:
    - Configurable encoder models (Sentence-BERT, DPR, custom transformers)
    - Multiple similarity search strategies (brute-force vs ANN)
    - Embedding caching for faster subsequent retrievals
    - Batch processing for efficient GPU utilization
    - Flexible text preprocessing and normalization
    
    Example:
        >>> config = {
        ...     "name": "sbert_retriever",
        ...     "type": "dense",
        ...     "model_name_or_path": "all-MiniLM-L6-v2",
        ...     "parameters": {
        ...         "use_ann_index": True,
        ...         "ann_backend": "faiss",
        ...         "normalize_embeddings": True
        ...     }
        ... }
        >>> model = DenseTextRetriever(config)
        >>> model.load_model()
        >>> model.index_corpus(corpus)
        >>> results = model.predict(queries, corpus, top_k=10)
    """

    def __init__(self, model_config: Dict[str, Any], **kwargs):
        """
        Initialize dense text retriever.
        
        Args:
            model_config: Configuration dictionary containing:
                - model_name_or_path (str): Path to pre-trained model
                - model_type (str): Type of model architecture 
                    ('single_encoder', 'dual_encoder', 'dpr')
                - use_ann_index (bool): Whether to use ANN indexing (default: False)
                - ann_backend (str): ANN backend ('faiss', 'hnswlib') (default: 'faiss')
                - normalize_embeddings (bool): Normalize embeddings (default: True)
                - embedding_dim (int): Expected embedding dimension (optional)
                - max_seq_length (int): Maximum sequence length (default: 512)
                - pooling_strategy (str): Pooling strategy for transformers models
                    ('cls', 'mean', 'max') (default: 'mean')
            **kwargs: Additional arguments
        """
        super().__init__(model_config, **kwargs)
        
        # Model configuration
        params = self.config.parameters
        self.model_name_or_path = model_config.get('model_name_or_path', 'all-MiniLM-L6-v2')
        self.model_architecture = params.get('model_architecture', params.get('model_type', 'single_encoder'))
        self.use_ann_index = params.get('use_ann_index', False)
        self.ann_backend = params.get('ann_backend', 'faiss')
        self.normalize_embeddings = params.get('normalize_embeddings', True)
        self.embedding_dim = params.get('embedding_dim', None)
        self.max_seq_length = params.get('max_seq_length', 512)
        self.pooling_strategy = params.get('pooling_strategy', 'mean')
        
        # ANN index parameters
        self.faiss_index_factory = params.get('faiss_index_factory_string', 'Flat')
        self.faiss_nprobe = params.get('faiss_nprobe', None)  # Auto-configure if None
        self.faiss_use_gpu = params.get('faiss_use_gpu', False)
        self.hnsw_m = params.get('m_parameter_hnsw', 16)
        self.hnsw_ef_construction = params.get('ef_construction_hnsw', 200)
        self.hnsw_ef_search = params.get('ef_search_hnsw', 50)
        
        # Model components (initialized in load_model)
        self.encoder_model = None
        self.query_encoder = None  # For dual-encoder models
        self.doc_encoder = None    # For dual-encoder models
        self.tokenizer = None
        
        # Embedding storage and indexing
        self.doc_embeddings = {}
        self.doc_ids_list = []  # Ordered list for ANN index mapping
        self.ann_index = None
        self.embedding_cache_path = None
        self.faiss_index_cache_path = None
        
        # Validate ANN backend availability
        if self.use_ann_index:
            if self.ann_backend == 'faiss' and not FAISS_AVAILABLE:
                raise ImportError("FAISS not available. Install with: pip install faiss-cpu")
            elif self.ann_backend == 'hnswlib' and not HNSWLIB_AVAILABLE:
                raise ImportError("HNSWLIB not available. Install with: pip install hnswlib")
        
        logger.info(f"Initialized DenseTextRetriever with model: {self.model_name_or_path}")
        logger.info(f"Architecture: {self.model_architecture}, ANN: {self.use_ann_index}")
    
    def load_model(self) -> None:
        """
        Load the dense retrieval model.
        
        This method loads the appropriate encoder model(s) based on the configuration.
        Supports sentence-transformers models, Hugging Face transformers, and 
        specialized architectures like DPR.
        """
        try:
            if self.model_architecture == 'sentence_transformer':
                # Use sentence-transformers library
                self.encoder_model = SentenceTransformer(
                    self.model_name_or_path,
                    device=self.config.device
                )
                
                # Set max sequence length
                self.encoder_model.max_seq_length = self.max_seq_length
                
                # Get embedding dimension
                if self.embedding_dim is None:
                    self.embedding_dim = self.encoder_model.get_sentence_embedding_dimension()
                
                logger.info(f"Loaded Sentence-BERT model: {self.model_name_or_path}")
                
            elif self.model_architecture == 'dpr':
                # DPR dual-encoder model
                from transformers import DPRQuestionEncoder, DPRContextEncoder
                
                if 'question' in self.model_name_or_path.lower():
                    self.query_encoder = DPRQuestionEncoder.from_pretrained(
                        self.model_name_or_path
                    ).to(self.config.device)
                    # Need corresponding context encoder
                    ctx_model_path = self.model_name_or_path.replace('question', 'ctx')
                    self.doc_encoder = DPRContextEncoder.from_pretrained(
                        ctx_model_path
                    ).to(self.config.device)
                else:
                    # Assume context encoder, find corresponding question encoder
                    self.doc_encoder = DPRContextEncoder.from_pretrained(
                        self.model_name_or_path
                    ).to(self.config.device)
                    query_model_path = self.model_name_or_path.replace('ctx', 'question')
                    self.query_encoder = DPRQuestionEncoder.from_pretrained(
                        query_model_path
                    ).to(self.config.device)
                
                # Load tokenizer
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_name_or_path)
                
                # Get embedding dimension from one of the encoders
                if self.embedding_dim is None:
                    self.embedding_dim = self.query_encoder.config.hidden_size
                
                logger.info(f"Loaded DPR dual-encoder model: {self.model_name_or_path}")
                
            elif self.model_architecture == 'transformers':
                # Generic transformers model
                self.encoder_model = AutoModel.from_pretrained(
                    self.model_name_or_path
                ).to(self.config.device)
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_name_or_path)
                
                # Get embedding dimension
                if self.embedding_dim is None:
                    self.embedding_dim = self.encoder_model.config.hidden_size
                
                logger.info(f"Loaded Transformers model: {self.model_name_or_path}")
                
            else:
                # Default to sentence-transformers
                self.encoder_model = SentenceTransformer(
                    self.model_name_or_path,
                    device=self.config.device
                )
                self.encoder_model.max_seq_length = self.max_seq_length
                
                # Update architecture to reflect that we're using sentence-transformers
                if self.model_architecture in ['single_encoder', 'dual_encoder']:
                    self.model_architecture = 'sentence_transformer'
                
                if self.embedding_dim is None:
                    self.embedding_dim = self.encoder_model.get_sentence_embedding_dimension()
                
                logger.info(f"Loaded default Sentence-BERT model: {self.model_name_or_path}")
            
            # Set up embedding cache path
            if self.config.cache_dir:
                cache_dir = Path(self.config.cache_dir)
                cache_dir.mkdir(parents=True, exist_ok=True)
                model_safe_name = self.model_name_or_path.replace('/', '_')
                self.embedding_cache_path = cache_dir / f"{model_safe_name}_embeddings.pkl"
                # Also set up FAISS index cache path
                self.faiss_index_cache_path = cache_dir / f"{model_safe_name}_faiss_{self.faiss_index_factory.replace(',', '_').replace(':', '_')}.index"
            
            self.is_loaded = True
            logger.info(f"Model loaded successfully. Embedding dimension: {self.embedding_dim}")
            
        except Exception as e:
            logger.error(f"Failed to load model {self.model_name_or_path}: {str(e)}")
            raise
    
    def _encode_texts_transformers(self, texts: List[str], is_query: bool = False) -> np.ndarray:
        """
        Encode texts using raw transformers model.
        
        Args:
            texts: List of texts to encode
            is_query: Whether these are queries (for dual-encoder models)
            
        Returns:
            Numpy array of embeddings
        """
        if self.model_architecture == 'dpr':
            # Use appropriate encoder for DPR
            encoder = self.query_encoder if is_query else self.doc_encoder
        else:
            encoder = self.encoder_model
        
        embeddings = []
        
        # Process in batches
        for i in range(0, len(texts), self.config.batch_size):
            batch_texts = texts[i:i + self.config.batch_size]
            
            # Tokenize
            inputs = self.tokenizer(
                batch_texts,
                truncation=True,
                padding=True,
                max_length=self.max_seq_length,
                return_tensors='pt'
            ).to(self.config.device)
            
            # Encode
            with torch.no_grad():
                outputs = encoder(**inputs)
                
                # Apply pooling strategy
                if self.pooling_strategy == 'cls':
                    batch_embeddings = outputs.last_hidden_state[:, 0, :]
                elif self.pooling_strategy == 'mean':
                    # Mean pooling with attention mask
                    attention_mask = inputs['attention_mask']
                    mask_expanded = attention_mask.unsqueeze(-1).expand(outputs.last_hidden_state.size()).float()
                    sum_embeddings = torch.sum(outputs.last_hidden_state * mask_expanded, 1)
                    sum_mask = torch.clamp(mask_expanded.sum(1), min=1e-9)
                    batch_embeddings = sum_embeddings / sum_mask
                elif self.pooling_strategy == 'max':
                    batch_embeddings = torch.max(outputs.last_hidden_state, dim=1)[0]
                else:
                    # Default to CLS
                    batch_embeddings = outputs.last_hidden_state[:, 0, :]
                
                embeddings.append(batch_embeddings.cpu().numpy())
        
        return np.vstack(embeddings)
    
    def encode_texts(self, 
                    texts: List[str], 
                    is_query: bool = False,
                    show_progress: bool = False) -> np.ndarray:
        """
        Encode a list of texts into embeddings.
        
        Args:
            texts: List of texts to encode
            is_query: Whether these are queries (affects encoder choice for dual-encoder)
            show_progress: Whether to show progress bar
            
        Returns:
            Numpy array of embeddings with shape (len(texts), embedding_dim)
        """
        if not texts:
            return np.empty((0, self.embedding_dim))
        
        logger.debug(f"Using model architecture: {self.model_architecture}")
        
        if self.model_architecture == 'sentence_transformer':
            # Use sentence-transformers
            logger.debug(f"Using sentence-transformer architecture, encoder_model: {self.encoder_model}")
            if self.encoder_model is None:
                raise RuntimeError("encoder_model is None - model was not loaded properly")
            embeddings = self.encoder_model.encode(
                texts,
                batch_size=self.config.batch_size,
                show_progress_bar=show_progress,
                convert_to_numpy=True
            )
        else:
            # Use transformers model
            embeddings = self._encode_texts_transformers(texts, is_query=is_query)
        
        # Normalize embeddings if requested
        if self.normalize_embeddings:
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            norms = np.where(norms == 0, 1, norms)  # Avoid division by zero
            embeddings = embeddings / norms
        
        return embeddings
    
    def encode_queries(self, 
                      queries: List[Dict[str, str]], 
                      **kwargs) -> Dict[str, np.ndarray]:
        """
        Encode queries into embeddings.
        
        Args:
            queries: List of query dictionaries with 'query_id' and 'text'
            **kwargs: Additional parameters
            
        Returns:
            Dictionary mapping query_id to embedding array
        """
        show_progress = kwargs.get('show_progress', False)
        
        # Extract query texts and IDs
        query_texts = []
        query_ids = []
        
        for query in queries:
            query_id = query['query_id']
            query_text = query.get('text', '')
            
            if not query_text:
                logger.warning(f"Empty query text for query_id: {query_id}")
                query_text = ""  # Use empty string for consistency
            
            query_texts.append(query_text)
            query_ids.append(query_id)
        
        # Encode all queries
        embeddings = self.encode_texts(query_texts, is_query=True, show_progress=show_progress)
        
        # Create mapping
        return {query_id: emb for query_id, emb in zip(query_ids, embeddings)}
    
    def encode_documents(self, 
                        documents: Dict[str, Dict[str, Any]], 
                        **kwargs) -> Dict[str, np.ndarray]:
        """
        Encode documents into embeddings.
        
        Args:
            documents: Dictionary of documents with doc_id -> document data
            **kwargs: Additional parameters
            
        Returns:
            Dictionary mapping doc_id to embedding array
        """
        show_progress = kwargs.get('show_progress', True)
        
        # Extract document texts and IDs
        doc_texts = []
        doc_ids = []
        
        for doc_id, doc_data in documents.items():
            # Extract text content with fallbacks
            text_content = doc_data.get('text', '')
            
            # Add title if available
            if 'title' in doc_data and doc_data['title']:
                title = doc_data['title']
                text_content = f"{title}. {text_content}" if text_content else title
            
            # Add OCR text if available and main text is empty
            if not text_content and 'ocr_text' in doc_data:
                text_content = doc_data.get('ocr_text', '')
            
            # Use empty string if no text found
            if not text_content:
                logger.warning(f"No text content found for document {doc_id}")
                text_content = ""
            
            doc_texts.append(text_content)
            doc_ids.append(doc_id)
        
        # Encode all documents
        embeddings = self.encode_texts(doc_texts, is_query=False, show_progress=show_progress)
        
        # Create mapping
        return {doc_id: emb for doc_id, emb in zip(doc_ids, embeddings)}
    
    def _create_faiss_index(self, embeddings: np.ndarray) -> Any:
        """
        Create FAISS index from embeddings with comprehensive support for different index types.
        
        Args:
            embeddings: Document embeddings array (N x D)
            
        Returns:
            Configured and trained FAISS index
        """
        if not FAISS_AVAILABLE:
            raise ImportError("FAISS not available")
        
        n_docs = len(embeddings)
        embeddings_f32 = embeddings.astype(np.float32)
        
        logger.info(f"Creating FAISS index for {n_docs} documents with factory string: {self.faiss_index_factory}")
        
        # Choose distance metric based on embedding normalization
        distance_metric = "IP" if self.normalize_embeddings else "L2"
        
        # Handle different index types with intelligent defaults
        if self.faiss_index_factory in ['Flat', 'IndexFlat']:
            # Exact search - best for small corpora (<10K)
            if self.normalize_embeddings:
                index = faiss.IndexFlatIP(self.embedding_dim)
                logger.info("Using IndexFlatIP for exact cosine similarity search")
            else:
                index = faiss.IndexFlatL2(self.embedding_dim)
                logger.info("Using IndexFlatL2 for exact L2 distance search")
                
        elif self.faiss_index_factory.startswith('IVF') or 'IVF' in self.faiss_index_factory:
            # IVF-based indices for medium to large corpora
            try:
                # Calculate optimal nlist (number of Voronoi cells) for later use
                nlist = min(int(np.sqrt(n_docs)), n_docs // 10)
                nlist = max(nlist, 1)  # Ensure at least 1 cluster
                nlist = min(nlist, 65536)  # FAISS limitation
                
                # Parse IVF configuration or use intelligent defaults
                if 'IVF' in self.faiss_index_factory and ',' in self.faiss_index_factory:
                    # Parse complex factory string like "IVF1024,Flat" or "IVF4096,PQ32"
                    index = faiss.index_factory(self.embedding_dim, self.faiss_index_factory, 
                                              faiss.METRIC_INNER_PRODUCT if self.normalize_embeddings else faiss.METRIC_L2)
                    logger.info(f"Created IVF index using factory string: {self.faiss_index_factory}")
                else:
                    # Create quantizer (first level index)
                    if self.normalize_embeddings:
                        quantizer = faiss.IndexFlatIP(self.embedding_dim)
                    else:
                        quantizer = faiss.IndexFlatL2(self.embedding_dim)
                    
                    # Create IVF index
                    if 'PQ' in self.faiss_index_factory:
                        # IVF with Product Quantization for memory efficiency
                        # Extract PQ parameters or use defaults
                        m_pq = 8  # Number of subquantizers (should divide embedding_dim)
                        while self.embedding_dim % m_pq != 0 and m_pq > 4:
                            m_pq -= 1
                        nbits = 8  # Bits per subquantizer
                        
                        index = faiss.IndexIVFPQ(quantizer, self.embedding_dim, nlist, m_pq, nbits)
                        logger.info(f"Using IndexIVFPQ with nlist={nlist}, m={m_pq}, nbits={nbits}")
                    else:
                        # Standard IVF with flat quantization
                        index = faiss.IndexIVFFlat(quantizer, self.embedding_dim, nlist)
                        logger.info(f"Using IndexIVFFlat with nlist={nlist}")
                
                # Configure IVF search parameters if it's an IVF-type index
                if hasattr(index, 'nprobe'):
                    # nprobe: number of clusters to search (trade-off between speed and recall)
                    nprobe = min(max(1, nlist // 10), 50)  # Search 10% of clusters, max 50
                    index.nprobe = nprobe
                    logger.info(f"Set nprobe={nprobe} for IVF search")
                
            except Exception as e:
                logger.warning(f"Failed to create IVF index: {e}, falling back to IndexFlat")
                index = faiss.IndexFlatIP(self.embedding_dim) if self.normalize_embeddings else faiss.IndexFlatL2(self.embedding_dim)
                
        elif 'HNSW' in self.faiss_index_factory:
            # Hierarchical Navigable Small World for fast approximate search
            try:
                if ',' in self.faiss_index_factory:
                    # Parse factory string like "HNSW32,Flat"
                    index = faiss.index_factory(self.embedding_dim, self.faiss_index_factory,
                                              faiss.METRIC_INNER_PRODUCT if self.normalize_embeddings else faiss.METRIC_L2)
                else:
                    # Create HNSW with default parameters
                    M = 32  # Number of connections per element (higher = better quality, more memory)
                    index = faiss.IndexHNSWFlat(self.embedding_dim, M)
                    if not self.normalize_embeddings:
                        index.metric_type = faiss.METRIC_L2
                    
                logger.info(f"Using HNSW index with M={getattr(index, 'hnsw', {}).get('M', 32)}")
                
            except Exception as e:
                logger.warning(f"Failed to create HNSW index: {e}, falling back to IndexFlat")
                index = faiss.IndexFlatIP(self.embedding_dim) if self.normalize_embeddings else faiss.IndexFlatL2(self.embedding_dim)
                
        else:
            # Try to use factory string directly
            try:
                metric = faiss.METRIC_INNER_PRODUCT if self.normalize_embeddings else faiss.METRIC_L2
                index = faiss.index_factory(self.embedding_dim, self.faiss_index_factory, metric)
                logger.info(f"Created index using factory string: {self.faiss_index_factory}")
            except Exception as e:
                logger.warning(f"Failed to create index with factory string '{self.faiss_index_factory}': {e}")
                logger.warning("Falling back to IndexFlatIP")
                index = faiss.IndexFlatIP(self.embedding_dim)
        
        # Train the index if required
        if not index.is_trained:
            logger.info(f"Training FAISS index with {n_docs} vectors...")
            # For large datasets, we can train on a subset for efficiency
            train_size = min(n_docs, 100000)  # Limit training set size
            if train_size < n_docs:
                train_indices = np.random.choice(n_docs, train_size, replace=False)
                train_embeddings = embeddings_f32[train_indices]
                logger.info(f"Training on subset of {train_size} vectors")
            else:
                train_embeddings = embeddings_f32
            
            index.train(train_embeddings)
            logger.info("FAISS index training completed")
        
        # Add all vectors to the index
        logger.info(f"Adding {n_docs} vectors to FAISS index...")
        index.add(embeddings_f32)
        
        # GPU support if requested and available
        if (self.faiss_use_gpu or self.config.device == 'cuda') and FAISS_AVAILABLE:
            try:
                if hasattr(faiss, 'index_cpu_to_gpu'):
                    res = faiss.StandardGpuResources()
                    index = faiss.index_cpu_to_gpu(res, 0, index)
                    logger.info("Moved FAISS index to GPU")
            except Exception as e:
                logger.warning(f"Failed to move FAISS index to GPU: {e}")
        
        logger.info(f"FAISS index created successfully with {index.ntotal} vectors")
        return index
    
    def _create_hnswlib_index(self, embeddings: np.ndarray) -> Any:
        """Create HNSWLIB index from embeddings."""
        if not HNSWLIB_AVAILABLE:
            raise ImportError("HNSWLIB not available")
        
        # Create index
        index = hnswlib.Index(space='cosine', dim=self.embedding_dim)
        index.init_index(
            max_elements=len(embeddings),
            M=self.hnsw_m,
            ef_construction=self.hnsw_ef_construction
        )
        
        # Set search parameters
        index.set_ef(self.hnsw_ef_search)
        
        # Add vectors
        index.add_items(embeddings, list(range(len(embeddings))))
        
        return index
    
    def index_corpus(self, corpus: Dict[str, Dict[str, Any]], **kwargs) -> None:
        """
        Index the corpus for dense retrieval.
        
        This method encodes all documents and optionally builds an ANN index
        for fast similarity search.
        
        Args:
            corpus: Dictionary of documents to index
            **kwargs: Additional parameters:
                - force_rebuild (bool): Force rebuilding index
                - cache_embeddings (bool): Cache embeddings to disk
                - load_cached_embeddings (bool): Load cached embeddings if available
        """
        if not self.is_loaded:
            raise RuntimeError("Model must be loaded before indexing")
        
        force_rebuild = kwargs.get('force_rebuild', False)
        cache_embeddings = kwargs.get('cache_embeddings', True)
        load_cached = kwargs.get('load_cached_embeddings', True)
        
        # Try to load cached embeddings
        if (load_cached and self.embedding_cache_path and 
            self.embedding_cache_path.exists() and not force_rebuild):
            try:
                with open(self.embedding_cache_path, 'rb') as f:
                    cache_data = pickle.load(f)
                    
                if (cache_data['doc_ids'] == list(corpus.keys()) and 
                    cache_data['model_name'] == self.model_name_or_path):
                    
                    self.doc_embeddings = cache_data['embeddings']
                    self.doc_ids_list = cache_data['doc_ids']
                    
                    logger.info(f"Loaded cached embeddings for {len(self.doc_embeddings)} documents")
                    
                    # Build ANN index if needed
                    if self.use_ann_index:
                        embeddings_array = np.array([self.doc_embeddings[doc_id] for doc_id in self.doc_ids_list])
                        self._build_ann_index(embeddings_array)
                    
                    self._corpus_indexed = True
                    return
            except Exception as e:
                logger.warning(f"Failed to load cached embeddings: {e}")
        
        logger.info(f"Encoding corpus with {len(corpus)} documents")
        
        # Encode all documents
        self.doc_embeddings = self.encode_documents(corpus, show_progress=True)
        self.doc_ids_list = list(corpus.keys())
        
        # Cache embeddings if requested
        if cache_embeddings and self.embedding_cache_path:
            try:
                cache_data = {
                    'embeddings': self.doc_embeddings,
                    'doc_ids': self.doc_ids_list,
                    'model_name': self.model_name_or_path,
                    'embedding_dim': self.embedding_dim
                }
                with open(self.embedding_cache_path, 'wb') as f:
                    pickle.dump(cache_data, f)
                logger.info(f"Cached embeddings to {self.embedding_cache_path}")
            except Exception as e:
                logger.warning(f"Failed to cache embeddings: {e}")
        
        # Build ANN index if requested
        if self.use_ann_index:
            embeddings_array = np.array([self.doc_embeddings[doc_id] for doc_id in self.doc_ids_list])
            self._build_ann_index(embeddings_array)
        
        self._corpus_indexed = True
        logger.info(f"Corpus indexing completed. Embeddings: {len(self.doc_embeddings)}, ANN: {self.use_ann_index}")
    
    def _build_ann_index(self, embeddings: np.ndarray) -> None:
        """
        Build ANN index from embeddings array with caching support.
        
        Args:
            embeddings: Document embeddings array
        """
        # Try to load cached FAISS index first
        if (self.ann_backend == 'faiss' and self.faiss_index_cache_path and 
            self.faiss_index_cache_path.exists()):
            try:
                logger.info(f"Loading cached FAISS index from {self.faiss_index_cache_path}")
                if FAISS_AVAILABLE:
                    self.ann_index = faiss.read_index(str(self.faiss_index_cache_path))
                    
                    # Move to GPU if requested
                    if self.config.device == 'cuda' and FAISS_AVAILABLE:
                        try:
                            if hasattr(faiss, 'index_cpu_to_gpu'):
                                res = faiss.StandardGpuResources()
                                self.ann_index = faiss.index_cpu_to_gpu(res, 0, self.ann_index)
                                logger.info("Moved cached FAISS index to GPU")
                        except Exception as e:
                            logger.warning(f"Failed to move cached FAISS index to GPU: {e}")
                    
                    logger.info(f"Successfully loaded cached FAISS index with {self.ann_index.ntotal} vectors")
                    return
            except Exception as e:
                logger.warning(f"Failed to load cached FAISS index: {e}, rebuilding...")
        
        logger.info(f"Building {self.ann_backend} index with {len(embeddings)} vectors")
        
        if self.ann_backend == 'faiss':
            self.ann_index = self._create_faiss_index(embeddings)
            
            # Save FAISS index to cache
            if self.faiss_index_cache_path:
                try:
                    # Save CPU version of index for caching
                    index_to_save = self.ann_index
                    if hasattr(self.ann_index, 'device') and self.ann_index.device >= 0:
                        # If index is on GPU, move to CPU for saving
                        index_to_save = faiss.index_gpu_to_cpu(self.ann_index)
                    
                    faiss.write_index(index_to_save, str(self.faiss_index_cache_path))
                    logger.info(f"Saved FAISS index to cache: {self.faiss_index_cache_path}")
                except Exception as e:
                    logger.warning(f"Failed to save FAISS index to cache: {e}")
                    
        elif self.ann_backend == 'hnswlib':
            self.ann_index = self._create_hnswlib_index(embeddings)
            
            # HNSWLIB doesn't have built-in persistence, but we could implement it
            # For now, just build fresh each time
        else:
            raise ValueError(f"Unsupported ANN backend: {self.ann_backend}")
        
        logger.info(f"{self.ann_backend} index built successfully")
    
    def _search_ann(self, query_embeddings: np.ndarray, top_k: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Search using ANN index with optimized performance and proper distance handling.
        
        Args:
            query_embeddings: Query embeddings array (N_queries x D)
            top_k: Number of top results to return
            
        Returns:
            Tuple of (scores, indices) arrays where:
            - scores: similarity scores (higher = more similar)
            - indices: document indices in the corpus
        """
        query_embeddings_f32 = query_embeddings.astype(np.float32)
        
        if self.ann_backend == 'faiss':
            if hasattr(self.ann_index, 'nprobe'):
                # Configure search-time parameters for IVF indices
                original_nprobe = self.ann_index.nprobe
                # Use configured nprobe or auto-calculate
                if self.faiss_nprobe is not None:
                    self.ann_index.nprobe = self.faiss_nprobe
                else:
                    # Dynamically adjust nprobe based on desired quality vs speed trade-off
                    # Higher nprobe = better recall but slower search
                    self.ann_index.nprobe = min(original_nprobe * 2, 100)
            
            # Perform FAISS search
            distances, indices = self.ann_index.search(query_embeddings_f32, top_k)
            
            # Convert distances to scores based on the index type and metric
            if hasattr(self.ann_index, 'metric_type'):
                if self.ann_index.metric_type == faiss.METRIC_INNER_PRODUCT:
                    # For inner product (cosine similarity with normalized vectors)
                    scores = distances  # distances are already similarity scores
                elif self.ann_index.metric_type == faiss.METRIC_L2:
                    # For L2 distance, convert to similarity scores
                    # Using exponential decay: similarity = exp(-distance)
                    scores = np.exp(-distances)
                else:
                    # Default: assume higher distance means lower similarity
                    scores = -distances
            elif isinstance(self.ann_index, type(faiss.IndexFlatIP(1))):
                # IndexFlatIP returns inner products (similarity scores)
                scores = distances
            else:
                # For other indices, assume distances need to be converted
                # Check if distances are negative (indicating they might be similarities)
                if np.all(distances <= 0):
                    scores = -distances  # Convert back to positive similarities
                else:
                    # Convert L2 distances to similarities
                    scores = 1.0 / (1.0 + distances)
            
            # Restore original nprobe if it was modified
            if hasattr(self.ann_index, 'nprobe') and 'original_nprobe' in locals():
                self.ann_index.nprobe = original_nprobe
                
        elif self.ann_backend == 'hnswlib':
            # HNSWLIB returns (indices, distances) for single query
            if len(query_embeddings_f32) == 1:
                indices, distances = self.ann_index.knn_query(query_embeddings_f32[0], k=top_k)
                # Convert cosine distance to similarity: similarity = 1 - distance
                scores = 1.0 - distances
                indices = indices.reshape(1, -1)
                scores = scores.reshape(1, -1)
            else:
                # Batch query processing
                all_indices = []
                all_scores = []
                for query_emb in query_embeddings_f32:
                    idx, dist = self.ann_index.knn_query(query_emb, k=top_k)
                    all_indices.append(idx)
                    all_scores.append(1.0 - dist)  # Convert distance to similarity
                indices = np.array(all_indices)
                scores = np.array(all_scores)
        else:
            raise ValueError(f"Unsupported ANN backend: {self.ann_backend}")
        
        # Ensure scores are properly bounded (similarities should be between 0 and 1 for cosine)
        if self.normalize_embeddings:
            scores = np.clip(scores, 0.0, 1.0)
        
        return scores, indices
    
    def _search_brute_force(self, query_embeddings: np.ndarray, top_k: int) -> List[List[Tuple[str, float]]]:
        """
        Brute-force similarity search.
        
        Args:
            query_embeddings: Query embeddings array
            top_k: Number of top results to return
            
        Returns:
            List of results for each query, where each result is a list of (doc_id, score) tuples
        """
        # Prepare document embeddings array
        doc_embeddings_array = np.array([self.doc_embeddings[doc_id] for doc_id in self.doc_ids_list])
        
        results = []
        
        for query_emb in query_embeddings:
            # Compute cosine similarities
            query_emb = query_emb.reshape(1, -1)
            similarities = cosine_similarity(query_emb, doc_embeddings_array)[0]
            
            # Get top-k indices
            top_indices = np.argsort(similarities)[::-1][:top_k]
            
            # Create results list
            query_results = []
            for idx in top_indices:
                doc_id = self.doc_ids_list[idx]
                score = float(similarities[idx])
                query_results.append((doc_id, score))
            
            results.append(query_results)
        
        return results
    
    def predict(self, 
                queries: List[Dict[str, str]], 
                corpus: Dict[str, Dict[str, Any]], 
                top_k: int = 1000,
                **kwargs) -> Dict[str, Dict[str, float]]:
        """
        Perform dense retrieval for given queries.
        
        Args:
            queries: List of query dictionaries with 'query_id' and 'text'
            corpus: Document corpus
            top_k: Number of top documents to return per query
            **kwargs: Additional parameters:
                - min_score (float): Minimum similarity score threshold
                
        Returns:
            Dictionary mapping query_id to {doc_id: score}
        """
        # Call parent validation
        super().predict(queries, corpus, top_k, **kwargs)
        
        # Ensure corpus is indexed
        if not self._corpus_indexed:
            logger.info("Corpus not indexed, indexing now...")
            self.index_corpus(corpus, **kwargs)
        
        # Check if corpus has changed
        if set(self.doc_ids_list) != set(corpus.keys()):
            logger.warning("Corpus has changed since indexing, re-indexing...")
            self.index_corpus(corpus, force_rebuild=True, **kwargs)
        
        # Encode queries
        query_embeddings_dict = self.encode_queries(queries)
        
        # Prepare embeddings array for search
        query_ids = [q['query_id'] for q in queries]
        query_embeddings = np.array([query_embeddings_dict[qid] for qid in query_ids])
        
        # Perform search
        if self.use_ann_index and self.ann_index is not None:
            logger.debug(f"Performing ANN search with {self.ann_backend}")
            scores, indices = self._search_ann(query_embeddings, top_k)
            
            # Convert to results format
            results = {}
            min_score = kwargs.get('min_score', 0.0)
            
            for i, query_id in enumerate(query_ids):
                query_results = {}
                
                for j in range(len(indices[i])):
                    if indices[i][j] == -1:  # Invalid index
                        break
                    
                    doc_idx = indices[i][j]
                    score = float(scores[i][j])
                    
                    if score >= min_score:
                        doc_id = self.doc_ids_list[doc_idx]
                        query_results[doc_id] = score
                
                results[query_id] = query_results
        else:
            logger.debug("Performing brute-force search")
            search_results = self._search_brute_force(query_embeddings, top_k)
            
            # Convert to results format
            results = {}
            min_score = kwargs.get('min_score', 0.0)
            
            for query_id, query_results_list in zip(query_ids, search_results):
                query_results = {
                    doc_id: score for doc_id, score in query_results_list
                    if score >= min_score
                }
                results[query_id] = query_results
        
        logger.info(f"Dense retrieval completed for {len(queries)} queries")
        return results
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the current model state.
        
        Returns:
            Dictionary with model information
        """
        return {
            'model_name': self.name,
            'model_type': 'Dense',
            'model_path': self.model_name_or_path,
            'architecture': self.model_architecture,
            'embedding_dim': self.embedding_dim,
            'parameters': {
                'use_ann_index': self.use_ann_index,
                'ann_backend': self.ann_backend if self.use_ann_index else None,
                'normalize_embeddings': self.normalize_embeddings,
                'max_seq_length': self.max_seq_length,
                'pooling_strategy': self.pooling_strategy
            },
            'corpus_indexed': self._corpus_indexed,
            'num_documents': len(self.doc_embeddings) if self.doc_embeddings else 0,
            'is_loaded': self.is_loaded,
            'device': self.config.device,
            'batch_size': self.config.batch_size
        }


# Convenience classes for specific architectures

class SentenceBERTModel(DenseTextRetriever):
    """
    Sentence-BERT model implementation.
    
    This is a convenience class that pre-configures DenseTextRetriever
    for Sentence-BERT models from the sentence-transformers library.
    """
    
    def __init__(self, model_config: Dict[str, Any], **kwargs):
        # Set default architecture
        if 'parameters' not in model_config:
            model_config['parameters'] = {}
        model_config['parameters']['model_type'] = 'sentence_transformer'
        
        super().__init__(model_config, **kwargs)


class DPRModel(DenseTextRetriever):
    """
    Dense Passage Retrieval (DPR) model implementation.
    
    This class handles DPR dual-encoder models with separate
    encoders for questions and passages.
    """
    
    def __init__(self, model_config: Dict[str, Any], **kwargs):
        # Set default architecture
        if 'parameters' not in model_config:
            model_config['parameters'] = {}
        model_config['parameters']['model_type'] = 'dpr'
        
        super().__init__(model_config, **kwargs)


class TransformersModel(DenseTextRetriever):
    """
    Generic Transformers model implementation.
    
    This class handles any Hugging Face transformers model
    for dense retrieval with configurable pooling strategies.
    """
    
    def __init__(self, model_config: Dict[str, Any], **kwargs):
        # Set default architecture
        if 'parameters' not in model_config:
            model_config['parameters'] = {}
        model_config['parameters']['model_type'] = 'transformers'
        
        super().__init__(model_config, **kwargs)
