"""
OpenAI Embedding-based retrieval model for NewAIBench framework.

This module provides a retrieval model that uses pre-computed OpenAI embeddings
stored in files. It loads embeddings for corpus and queries, then performs
cosine similarity search for retrieval.
"""

import logging
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import json
import os

# Import base classes
from .base import BaseRetrievalModel, ModelType

logger = logging.getLogger(__name__)

class OpenAIEmbeddingRetriever(BaseRetrievalModel):
    """
    Retrieval model using pre-computed OpenAI embeddings.
    
    This model loads pre-computed embeddings from OpenAI API that were
    saved as .npy files, and performs similarity search for retrieval.
    
    Features:
    - Uses pre-computed OpenAI embeddings (text-embedding-3-small/large)
    - Cosine similarity search
    - Efficient numpy-based computation
    - Support for different embedding models
    - Automatic detection of embedding dimensions
    
    Example:
        >>> config = {
        ...     "name": "openai_retriever",
        ...     "type": "openai_embedding",
        ...     "parameters": {
        ...         "embeddings_dir": "../embeddings",
        ...         "embedding_model": "text-embedding-3-small",
        ...         "dataset_name": "tydiqa_goldp_vietnamese"
        ...     }
        ... }
        >>> model = OpenAIEmbeddingRetriever(config)
        >>> model.load_model()
        >>> results = model.predict(queries, corpus, top_k=10)
    """
    
    def __init__(self, model_config: Dict[str, Any], **kwargs):
        """Initialize OpenAI embedding retriever."""
        super().__init__(model_config, **kwargs)
        
        # Extract parameters
        params = self.config.parameters
        
        # Required parameters
        self.embeddings_dir = Path(params.get('embeddings_dir', '../embeddings'))
        self.embedding_model = params.get('embedding_model', 'text-embedding-3-small')
        self.dataset_name = params.get('dataset_name')
        
        # Optional parameters
        self.normalize_embeddings = params.get('normalize_embeddings', True)
        self.cache_embeddings = params.get('cache_embeddings', True)
        
        # Build embedding paths (will be updated if dataset_name is set later)
        self._update_embedding_paths()
        
        # Storage for loaded embeddings
        self.corpus_embeddings = {}
        self.query_embeddings = {}
        self.embedding_dim = None
        
        # Model state
        self.is_loaded = False
        self._corpus_indexed = False
        
        logger.info(f"Initialized OpenAI Embedding Retriever")
        logger.info(f"  Model: {self.embedding_model}")
        logger.info(f"  Dataset: {self.dataset_name}")
        if self.dataset_name:
            logger.info(f"  Embeddings path: {self.model_embedding_dir}")
    
    def _update_embedding_paths(self):
        """Update embedding paths based on current dataset_name."""
        if self.dataset_name:
            self.model_embedding_dir = self.embeddings_dir / self.embedding_model / self.dataset_name
            self.corpus_embedding_dir = self.model_embedding_dir / "corpus"
            self.queries_embedding_dir = self.model_embedding_dir / "queries"
        else:
            self.model_embedding_dir = None
            self.corpus_embedding_dir = None
            self.queries_embedding_dir = None
    
    def set_dataset_name(self, dataset_name: str):
        """Set dataset name and update embedding paths.
        
        This allows the same model instance to be used with different datasets.
        
        Args:
            dataset_name: Name of the dataset
        """
        self.dataset_name = dataset_name
        self._update_embedding_paths()
        
        # Reset loaded state since paths changed
        self.is_loaded = False
        self._corpus_indexed = False
        self.corpus_embeddings = {}
        self.query_embeddings = {}
        
        logger.info(f"Dataset name set to: {dataset_name}")
        logger.info(f"Updated embeddings path: {self.model_embedding_dir}")
    
    def _check_embeddings_exist(self) -> Tuple[bool, str]:
        """Check if embedding files exist and return status."""
        if not self.model_embedding_dir.exists():
            return False, f"Embedding directory not found: {self.model_embedding_dir}"
        
        if not self.corpus_embedding_dir.exists():
            return False, f"Corpus embeddings not found: {self.corpus_embedding_dir}"
        
        if not self.queries_embedding_dir.exists():
            return False, f"Query embeddings not found: {self.queries_embedding_dir}"
        
        # Check if there are actual embedding files
        corpus_files = list(self.corpus_embedding_dir.glob("*.npy"))
        query_files = list(self.queries_embedding_dir.glob("*.npy"))
        
        if not corpus_files:
            return False, f"No corpus embedding files found in {self.corpus_embedding_dir}"
        
        if not query_files:
            return False, f"No query embedding files found in {self.queries_embedding_dir}"
        
        return True, f"Found {len(corpus_files)} corpus and {len(query_files)} query embeddings"
    
    def load_model(self) -> None:
        """Load the model (check if embeddings exist)."""
        logger.info("Loading OpenAI embedding model...")
        
        # Check if dataset_name is set
        if not self.dataset_name:
            raise ValueError(
                "dataset_name must be set before loading model. "
                "Either provide it in parameters or call set_dataset_name() first."
            )
        
        # Check if embeddings exist
        exists, message = self._check_embeddings_exist()
        if not exists:
            raise FileNotFoundError(
                f"Embeddings not found for model '{self.embedding_model}' "
                f"and dataset '{self.dataset_name}'. {message}\\n"
                f"Please run embedding generation first:\\n"
                f"cd embedding_tools && python embed_dataset.py "
                f"--dataset {self.dataset_name} --model {self.embedding_model}"
            )
        
        logger.info(f"✅ {message}")
        
        # Load metadata if available
        metadata_file = self.model_embedding_dir / "embedding_metadata.json"
        if metadata_file.exists():
            with open(metadata_file, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
                logger.info(f"Embedding metadata: {metadata}")
        
        self.is_loaded = True
        logger.info("OpenAI embedding model loaded successfully")
    
    def _load_embedding(self, file_path: Path) -> np.ndarray:
        """Load a single embedding from file."""
        embedding = np.load(file_path)
        
        # Set embedding dimension if not set
        if self.embedding_dim is None:
            self.embedding_dim = embedding.shape[0]
            logger.info(f"Detected embedding dimension: {self.embedding_dim}")
        
        # Normalize if requested
        if self.normalize_embeddings:
            embedding = embedding / np.linalg.norm(embedding)
        
        return embedding
    
    def _load_corpus_embeddings(self, corpus: Dict[str, Dict[str, Any]]) -> Dict[str, np.ndarray]:
        """Load corpus embeddings for given documents."""
        logger.info("Loading corpus embeddings...")
        
        embeddings = {}
        missing_docs = []
        
        for doc_id in corpus.keys():
            embedding_file = self.corpus_embedding_dir / f"{doc_id}.npy"
            
            if embedding_file.exists():
                embeddings[doc_id] = self._load_embedding(embedding_file)
            else:
                missing_docs.append(doc_id)
        
        if missing_docs:
            logger.warning(f"Missing embeddings for {len(missing_docs)} documents: {missing_docs[:5]}...")
        
        logger.info(f"Loaded embeddings for {len(embeddings)}/{len(corpus)} corpus documents")
        return embeddings
    
    def _load_query_embeddings(self, queries: List[Dict[str, str]]) -> Dict[str, np.ndarray]:
        """Load query embeddings for given queries."""
        logger.info("Loading query embeddings...")
        
        embeddings = {}
        missing_queries = []
        
        for query in queries:
            query_id = query['query_id']
            embedding_file = self.queries_embedding_dir / f"{query_id}.npy"
            
            if embedding_file.exists():
                embeddings[query_id] = self._load_embedding(embedding_file)
            else:
                missing_queries.append(query_id)
        
        if missing_queries:
            logger.warning(f"Missing embeddings for {len(missing_queries)} queries: {missing_queries[:5]}...")
        
        logger.info(f"Loaded embeddings for {len(embeddings)}/{len(queries)} queries")
        return embeddings
    
    def _compute_similarities(self, 
                            query_embeddings: Dict[str, np.ndarray],
                            corpus_embeddings: Dict[str, np.ndarray]) -> Dict[str, Dict[str, float]]:
        """Compute cosine similarities between queries and documents."""
        logger.info("Computing similarities...")
        
        results = {}
        
        for query_id, query_emb in query_embeddings.items():
            doc_scores = {}
            
            for doc_id, doc_emb in corpus_embeddings.items():
                # Cosine similarity (embeddings already normalized if requested)
                if self.normalize_embeddings:
                    similarity = float(np.dot(query_emb, doc_emb))
                else:
                    similarity = float(np.dot(query_emb, doc_emb) / 
                                     (np.linalg.norm(query_emb) * np.linalg.norm(doc_emb)))
                doc_scores[doc_id] = similarity
            
            results[query_id] = doc_scores
        
        return results
    
    def index_corpus(self, corpus: Dict[str, Dict[str, Any]], **kwargs) -> None:
        """Index corpus by loading embeddings."""
        if not self.is_loaded:
            raise RuntimeError("Model must be loaded before indexing corpus")
        
        logger.info(f"Indexing corpus with {len(corpus)} documents...")
        
        # Load corpus embeddings
        self.corpus_embeddings = self._load_corpus_embeddings(corpus)
        
        if not self.corpus_embeddings:
            raise RuntimeError("No corpus embeddings could be loaded")
        
        self._corpus_indexed = True
        logger.info("Corpus indexing completed")
    
    def predict(self, 
                queries: List[Dict[str, str]], 
                corpus: Dict[str, Dict[str, Any]], 
                top_k: int = 1000,
                **kwargs) -> Dict[str, Dict[str, float]]:
        """
        Perform retrieval using pre-computed OpenAI embeddings.
        
        Args:
            queries: List of query dictionaries
            corpus: Dictionary of documents
            top_k: Number of top results to return
            **kwargs: Additional keyword arguments
            
        Returns:
            Dict[str, Dict[str, float]]: Results mapping query IDs to doc scores
        """
        # Call parent validation
        super().predict(queries, corpus, top_k, **kwargs)
        
        # Index corpus if not already done
        if not self._corpus_indexed:
            self.index_corpus(corpus, **kwargs)
        
        # Load query embeddings
        query_embeddings = self._load_query_embeddings(queries)
        
        if not query_embeddings:
            logger.warning("No query embeddings could be loaded")
            return {q['query_id']: {} for q in queries}
        
        # Compute similarities
        similarities = self._compute_similarities(query_embeddings, self.corpus_embeddings)
        
        # Sort and limit to top_k
        results = {}
        min_score = kwargs.get('min_score', 0.0)
        
        for query_id, doc_scores in similarities.items():
            # Sort by score descending
            sorted_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)
            
            # Apply filters
            filtered_docs = [(doc_id, score) for doc_id, score in sorted_docs 
                           if score >= min_score]
            
            # Take top_k
            top_docs = filtered_docs[:top_k]
            
            # Convert to final format
            results[query_id] = {doc_id: score for doc_id, score in top_docs}
        
        logger.info(f"OpenAI embedding retrieval completed for {len(queries)} queries")
        return results
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information."""
        return {
            "model_name": self.name,
            "model_type": "OpenAI Embedding",
            "embedding_model": self.embedding_model,
            "dataset_name": self.dataset_name,
            "embedding_dim": self.embedding_dim,
            "parameter_count": 0,  # No trainable parameters
            "corpus_indexed": self._corpus_indexed,
            "normalize_embeddings": self.normalize_embeddings,
            "embeddings_path": str(self.model_embedding_dir),
            "corpus_embeddings_loaded": len(self.corpus_embeddings),
            "supported_features": ["cosine_similarity", "pre_computed_embeddings"]
        }
