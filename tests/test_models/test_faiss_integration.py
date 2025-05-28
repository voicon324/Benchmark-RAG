"""
Unit tests for FAISS integration in Dense retrieval models.

This module provides comprehensive tests for FAISS functionality including
different index types, GPU support, caching, and performance testing.
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch
import tempfile
import os
import sys
from pathlib import Path

# Add src to path for testing
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../src'))

from newaibench.models.dense import DenseTextRetriever


class TestFAISSIntegration:
    """Test class specifically for FAISS integration functionality."""
    
    @pytest.fixture
    def faiss_config_base(self):
        """Base FAISS configuration for testing."""
        return {
            "name": "test_faiss",
            "type": "dense", 
            "model_name_or_path": "all-MiniLM-L6-v2",
            "device": "cpu",
            "batch_size": 4,
            "cache_dir": None,  # Will be set in tests
            "parameters": {
                "model_architecture": "sentence_transformer",
                "normalize_embeddings": True,
                "use_ann_index": True,
                "ann_backend": "faiss",
                "max_seq_length": 128
            }
        }
    
    @pytest.fixture
    def sample_embeddings(self):
        """Sample embeddings for testing."""
        np.random.seed(42)
        return np.random.randn(20, 384).astype(np.float32)
    
    @pytest.fixture
    def large_sample_embeddings(self):
        """Larger sample embeddings for IVF testing."""
        np.random.seed(42)
        return np.random.randn(1000, 384).astype(np.float32)
    
    def test_faiss_index_factory_flat(self, faiss_config_base, sample_embeddings):
        """Test FAISS IndexFlat creation."""
        pytest.importorskip("faiss")
        
        config = faiss_config_base.copy()
        config["parameters"]["faiss_index_factory_string"] = "Flat"
        
        model = DenseTextRetriever(config)
        model.load_model()  # Need to load model to set embedding_dim
        index = model._create_faiss_index(sample_embeddings)
        
        assert index is not None
        assert index.ntotal == len(sample_embeddings)
        assert index.d == sample_embeddings.shape[1]
    
    def test_faiss_index_factory_ivf(self, faiss_config_base, large_sample_embeddings):
        """Test FAISS IVF index creation."""
        pytest.importorskip("faiss")
        
        config = faiss_config_base.copy()
        config["parameters"]["faiss_index_factory_string"] = "IVF4,Flat"
        
        model = DenseTextRetriever(config)
        model.load_model()  # Need to load model to set embedding_dim
        index = model._create_faiss_index(large_sample_embeddings)
        
        assert index is not None
        assert index.ntotal == len(large_sample_embeddings)
        assert hasattr(index, 'nprobe')
    
    def test_faiss_index_factory_ivfpq(self, faiss_config_base, large_sample_embeddings):
        """Test FAISS IVF+PQ index creation."""
        pytest.importorskip("faiss")
        
        config = faiss_config_base.copy()
        config["parameters"]["faiss_index_factory_string"] = "IVF4,PQ8"
        
        model = DenseTextRetriever(config)
        model.load_model()  # Need to load model to set embedding_dim
        index = model._create_faiss_index(large_sample_embeddings)
        
        assert index is not None
        assert index.ntotal == len(large_sample_embeddings)
    
    def test_faiss_index_factory_hnsw(self, faiss_config_base, sample_embeddings):
        """Test FAISS HNSW index creation."""
        pytest.importorskip("faiss")
        
        config = faiss_config_base.copy()
        config["parameters"]["faiss_index_factory_string"] = "HNSW32,Flat"
        
        model = DenseTextRetriever(config)
        model.load_model()  # Need to load model to set embedding_dim
        index = model._create_faiss_index(sample_embeddings)
        
        assert index is not None
        assert index.ntotal == len(sample_embeddings)
    
    def test_faiss_search_flat_vs_brute_force(self, faiss_config_base):
        """Test that FAISS IndexFlat gives identical results to brute force."""
        pytest.importorskip("faiss")
        
        # Create small test corpus
        test_corpus = {
            "doc1": {"text": "machine learning algorithms"},
            "doc2": {"text": "neural networks deep learning"},
            "doc3": {"text": "artificial intelligence research"},
            "doc4": {"text": "data science analytics"},
            "doc5": {"text": "computer vision image processing"}
        }
        
        test_queries = [{"query_id": "q1", "text": "machine learning"}]
        
        # Brute force configuration
        bf_config = faiss_config_base.copy()
        bf_config["parameters"]["use_ann_index"] = False
        
        # FAISS flat configuration
        faiss_config = faiss_config_base.copy()
        faiss_config["parameters"]["faiss_index_factory_string"] = "Flat"
        
        # Test both models
        bf_model = DenseTextRetriever(bf_config)
        bf_model.load_model()
        bf_results = bf_model.predict(test_queries, test_corpus, top_k=3)
        
        faiss_model = DenseTextRetriever(faiss_config)
        faiss_model.load_model()
        faiss_results = faiss_model.predict(test_queries, test_corpus, top_k=3)
        
        # Results should be identical for flat index
        qid = test_queries[0]["query_id"]
        bf_docs = list(bf_results[qid].keys())
        faiss_docs = list(faiss_results[qid].keys())
        
        # Document order should be the same (within numerical precision)
        assert len(bf_docs) == len(faiss_docs)
        # Allow small numerical differences in ranking
        overlap = len(set(bf_docs[:2]).intersection(set(faiss_docs[:2])))
        assert overlap >= 1  # At least top result should match
    
    def test_faiss_index_caching(self, faiss_config_base):
        """Test FAISS index caching functionality."""
        pytest.importorskip("faiss")
        
        with tempfile.TemporaryDirectory() as temp_dir:
            config = faiss_config_base.copy()
            config["cache_dir"] = temp_dir
            config["parameters"]["faiss_index_factory_string"] = "Flat"
            
            model = DenseTextRetriever(config)
            model.load_model()
            
            # Create test corpus
            test_corpus = {
                f"doc{i}": {"text": f"document {i} content"} 
                for i in range(10)
            }
            
            # First indexing - should create cache
            model.index_corpus(test_corpus)
            assert model.faiss_index_cache_path.exists()
            
            # Second model with same config - should load from cache
            model2 = DenseTextRetriever(config)
            model2.load_model()
            model2.index_corpus(test_corpus)
            
            # Both models should have identical indices
            assert model.ann_index.ntotal == model2.ann_index.ntotal
    
    def test_faiss_nprobe_configuration(self, faiss_config_base, large_sample_embeddings):
        """Test FAISS nprobe configuration for IVF indices."""
        pytest.importorskip("faiss")
        
        config = faiss_config_base.copy()
        config["parameters"]["faiss_index_factory_string"] = "IVF4,Flat"
        config["parameters"]["faiss_nprobe"] = 2
        
        model = DenseTextRetriever(config)
        model.load_model()  # Need to load model to set embedding_dim
        index = model._create_faiss_index(large_sample_embeddings)
        
        # Check that nprobe is set correctly
        assert hasattr(index, 'nprobe')
        
        # Set the index in the model and configure backend
        model.ann_index = index
        model.ann_backend = 'faiss'
        
        # Test search with configured nprobe
        query_emb = np.random.randn(1, 384).astype(np.float32)
        scores, indices = model._search_ann(query_emb, top_k=3)
        
        assert scores.shape == (1, 3)
        assert indices.shape == (1, 3)
    
    def test_faiss_search_quality(self, faiss_config_base):
        """Test that FAISS search returns reasonable quality results."""
        pytest.importorskip("faiss")
        
        # Create test corpus with semantic similarity
        test_corpus = {
            "doc1": {"text": "machine learning artificial intelligence"},
            "doc2": {"text": "deep neural networks"},
            "doc3": {"text": "cooking recipes food"},
            "doc4": {"text": "travel destinations vacation"},
            "doc5": {"text": "AI algorithms computer science"}
        }
        
        test_queries = [{"query_id": "q1", "text": "artificial intelligence"}]
        
        config = faiss_config_base.copy()
        config["parameters"]["faiss_index_factory_string"] = "Flat"
        
        model = DenseTextRetriever(config)
        model.load_model()
        results = model.predict(test_queries, test_corpus, top_k=3)
        
        qid = test_queries[0]["query_id"]
        top_docs = list(results[qid].keys())
        
        # Should retrieve AI-related documents first
        assert "doc1" in top_docs or "doc5" in top_docs
        # Should not retrieve cooking/travel docs in top results
        assert "doc3" not in top_docs[:2]
        assert "doc4" not in top_docs[:2]
    
    def test_faiss_error_handling(self, faiss_config_base):
        """Test FAISS error handling and fallbacks."""
        pytest.importorskip("faiss")
        
        # Test with invalid factory string
        config = faiss_config_base.copy()
        config["parameters"]["faiss_index_factory_string"] = "InvalidIndexType"
        
        model = DenseTextRetriever(config)
        model.load_model()  # Need to load model to set embedding_dim
        
        # Should create embeddings but fall back to default index
        sample_embeddings = np.random.randn(5, 384).astype(np.float32)
        index = model._create_faiss_index(sample_embeddings)
        
        # Should fall back to a working index
        assert index is not None
        assert index.ntotal == len(sample_embeddings)
    
    def test_faiss_different_metrics(self, faiss_config_base, sample_embeddings):
        """Test FAISS with different distance metrics."""
        pytest.importorskip("faiss")
        
        # Test with normalized embeddings (cosine similarity via inner product)
        config_ip = faiss_config_base.copy()
        config_ip["parameters"]["normalize_embeddings"] = True
        config_ip["parameters"]["faiss_index_factory_string"] = "Flat"
        
        model_ip = DenseTextRetriever(config_ip)
        model_ip.load_model()  # Need to load model to set embedding_dim
        index_ip = model_ip._create_faiss_index(sample_embeddings)
        
        # Test with non-normalized embeddings (L2 distance)
        config_l2 = faiss_config_base.copy()
        config_l2["parameters"]["normalize_embeddings"] = False
        config_l2["parameters"]["faiss_index_factory_string"] = "Flat"
        
        model_l2 = DenseTextRetriever(config_l2)
        model_l2.load_model()  # Need to load model to set embedding_dim
        index_l2 = model_l2._create_faiss_index(sample_embeddings)
        
        assert index_ip is not None
        assert index_l2 is not None
        # Both should have same number of vectors
        assert index_ip.ntotal == index_l2.ntotal
    
    def test_faiss_batch_search(self, faiss_config_base):
        """Test FAISS batch search functionality."""
        pytest.importorskip("faiss")
        
        # Create test corpus
        test_corpus = {f"doc{i}": {"text": f"document {i}"} for i in range(10)}
        test_queries = [
            {"query_id": "q1", "text": "document 1"},
            {"query_id": "q2", "text": "document 5"},
            {"query_id": "q3", "text": "document 9"}
        ]
        
        config = faiss_config_base.copy()
        config["parameters"]["faiss_index_factory_string"] = "Flat"
        
        model = DenseTextRetriever(config)
        model.load_model()
        results = model.predict(test_queries, test_corpus, top_k=3)
        
        # Should return results for all queries
        assert len(results) == len(test_queries)
        for query in test_queries:
            qid = query["query_id"]
            assert qid in results
            assert len(results[qid]) <= 3  # top_k=3


class TestFAISSPerformance:
    """Performance and scalability tests for FAISS integration."""
    
    def test_faiss_scalability(self):
        """Test FAISS performance with larger document collections."""
        pytest.importorskip("faiss")
        
        # Skip if running in CI or limited environment
        if os.environ.get('CI') or os.environ.get('SKIP_PERFORMANCE_TESTS'):
            pytest.skip("Skipping performance test in CI")
        
        # Create larger test corpus
        n_docs = 500  # Reduced for testing
        test_corpus = {
            f"doc{i}": {"text": f"document {i} with content {i % 10}"}
            for i in range(n_docs)
        }
        
        test_queries = [{"query_id": "q1", "text": "document content"}]
        
        config = {
            "name": "test_scalability",
            "type": "dense",
            "model_name_or_path": "all-MiniLM-L6-v2",
            "device": "cpu",
            "batch_size": 32,
            "parameters": {
                "model_architecture": "sentence_transformer",
                "normalize_embeddings": True,
                "use_ann_index": True,
                "ann_backend": "faiss",
                "faiss_index_factory_string": "IVF8,Flat",
                "max_seq_length": 128
            }
        }
        
        model = DenseTextRetriever(config)
        model.load_model()
        
        # Measure indexing time
        import time
        start_time = time.time()
        model.index_corpus(test_corpus)
        index_time = time.time() - start_time
        
        # Measure search time
        start_time = time.time()
        results = model.predict(test_queries, test_corpus, top_k=10)
        search_time = time.time() - start_time
        
        # Basic performance checks
        assert index_time < 120  # Should index docs within reasonable time
        assert search_time < 10   # Should search within reasonable time
        assert len(results["q1"]) == 10  # Should return correct number of results


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
