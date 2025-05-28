"""
Unit tests for BM25Model sparse retrieval implementation.

This module provides comprehensive tests for the BM25Model class,
including initialization, corpus indexing, and retrieval functionality.
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch
import tempfile
import os
import sys

# Add src to path for testing
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../src'))

from newaibench.models.sparse import BM25Model
from newaibench.models.base import ModelType


class TestBM25Model:
    """Test class for BM25Model functionality."""
    
    @pytest.fixture
    def sample_config(self):
        """Sample configuration for BM25Model."""
        return {
            "name": "test_bm25",
            "type": "sparse",
            "parameters": {
                "k1": 1.6,
                "b": 0.75,
                "tokenizer": "simple",
                "lowercase": True,
                "remove_stopwords": False
            }
        }
    
    @pytest.fixture
    def sample_corpus(self):
        """Sample corpus for testing."""
        return {
            "doc1": {
                "text": "Machine learning is a powerful tool for data analysis",
                "title": "ML Overview"
            },
            "doc2": {
                "text": "Deep learning neural networks are used for complex tasks",
                "title": "Deep Learning"
            },
            "doc3": {
                "text": "Natural language processing helps computers understand text",
                "title": "NLP Basics"
            },
            "doc4": {
                "text": "Information retrieval systems find relevant documents",
                "title": "IR Systems"
            }
        }
    
    @pytest.fixture
    def sample_queries(self):
        """Sample queries for testing."""
        return [
            {"query_id": "q1", "text": "machine learning algorithms"},
            {"query_id": "q2", "text": "deep neural networks"},
            {"query_id": "q3", "text": "text processing"}
        ]
    
    def test_bm25_initialization_default_params(self):
        """Test BM25Model initialization with default parameters."""
        config = {
            "name": "test_bm25",
            "type": "sparse"
        }
        
        model = BM25Model(config)
        
        assert model.name == "test_bm25"
        assert model.model_type == ModelType.SPARSE
        assert model.k1 == 1.6  # default
        assert model.b == 0.75  # default
        assert model.tokenizer_type == "simple"  # default
        assert model.lowercase is True  # default
        assert model.remove_stopwords is False  # default
        assert not model.is_loaded
        assert not model._corpus_indexed
    
    def test_bm25_initialization_custom_params(self, sample_config):
        """Test BM25Model initialization with custom parameters."""
        sample_config["parameters"].update({
            "k1": 2.0,
            "b": 0.8,
            "tokenizer": "regex",
            "min_token_length": 3,
            "remove_stopwords": True,
            "stopwords": ["custom", "stop"]
        })
        
        model = BM25Model(sample_config)
        
        assert model.k1 == 2.0
        assert model.b == 0.8
        assert model.tokenizer_type == "regex"
        assert model.min_token_length == 3
        assert model.remove_stopwords is True
        assert "custom" in model.custom_stopwords
        assert "stop" in model.custom_stopwords
    
    def test_tokenize_text_simple(self, sample_config):
        """Test text tokenization with simple tokenizer."""
        model = BM25Model(sample_config)
        
        text = "Hello World! This is a test."
        tokens = model._tokenize_text(text)
        
        expected = ["hello", "world", "this", "is", "test"]
        assert tokens == expected
    
    def test_tokenize_text_with_stopwords(self, sample_config):
        """Test text tokenization with stopword removal."""
        sample_config["parameters"]["remove_stopwords"] = True
        model = BM25Model(sample_config)
        
        text = "This is a test of the system"
        tokens = model._tokenize_text(text)
        
        # Should remove common stopwords
        assert "this" not in tokens
        assert "is" not in tokens
        assert "a" not in tokens
        assert "of" not in tokens
        assert "the" not in tokens
        assert "test" in tokens
        assert "system" in tokens
    
    def test_tokenize_text_min_length(self, sample_config):
        """Test text tokenization with minimum token length."""
        sample_config["parameters"]["min_token_length"] = 3
        model = BM25Model(sample_config)
        
        text = "A big cat runs to me"
        tokens = model._tokenize_text(text)
        
        # Should filter out tokens shorter than 3 characters
        assert "big" in tokens
        assert "cat" in tokens
        assert "runs" in tokens
        assert "a" not in tokens  # too short
        assert "to" not in tokens  # too short
        assert "me" not in tokens  # too short
    
    def test_load_model_success(self, sample_config):
        """Test successful model loading."""
        model = BM25Model(sample_config)
        
        model.load_model()
        
        assert model.is_loaded is True
    
    def test_load_model_invalid_params(self, sample_config):
        """Test model loading with invalid parameters."""
        # Test invalid k1
        sample_config["parameters"]["k1"] = -1
        model = BM25Model(sample_config)
        
        with pytest.raises(ValueError, match="k1 parameter must be in"):
            model.load_model()
        
        # Test invalid b
        sample_config["parameters"]["k1"] = 1.6
        sample_config["parameters"]["b"] = 1.5
        model = BM25Model(sample_config)
        
        with pytest.raises(ValueError, match="b parameter must be in"):
            model.load_model()
    
    def test_index_corpus_success(self, sample_config, sample_corpus):
        """Test successful corpus indexing."""
        model = BM25Model(sample_config)
        model.load_model()
        
        model.index_corpus(sample_corpus)
        
        assert model._corpus_indexed is True
        assert model.bm25_model is not None
        assert len(model.doc_ids) == len(sample_corpus)
        assert len(model.tokenized_corpus) == len(sample_corpus)
    
    def test_index_corpus_without_loading(self, sample_config, sample_corpus):
        """Test corpus indexing without loading model first."""
        model = BM25Model(sample_config)
        
        with pytest.raises(RuntimeError, match="Model must be loaded"):
            model.index_corpus(sample_corpus)
    
    def test_predict_basic_functionality(self, sample_config, sample_corpus, sample_queries):
        """Test basic prediction functionality."""
        model = BM25Model(sample_config)
        model.load_model()
        model.index_corpus(sample_corpus)
        
        results = model.predict(sample_queries, sample_corpus, top_k=2)
        
        # Check result structure
        assert isinstance(results, dict)
        assert len(results) == len(sample_queries)
        
        for query_data in sample_queries:
            query_id = query_data["query_id"]
            assert query_id in results
            assert isinstance(results[query_id], dict)
            assert len(results[query_id]) <= 2  # top_k=2
            
            # Check that scores are floats and in descending order
            scores = list(results[query_id].values())
            assert all(isinstance(s, float) for s in scores)
            assert scores == sorted(scores, reverse=True)
    
    def test_predict_without_indexing(self, sample_config, sample_corpus, sample_queries):
        """Test prediction with automatic indexing."""
        model = BM25Model(sample_config)
        model.load_model()
        
        # Should automatically index corpus
        results = model.predict(sample_queries, sample_corpus, top_k=2)
        
        assert model._corpus_indexed is True
        assert len(results) == len(sample_queries)
    
    def test_predict_empty_query(self, sample_config, sample_corpus):
        """Test prediction with empty query."""
        model = BM25Model(sample_config)
        model.load_model()
        model.index_corpus(sample_corpus)
        
        empty_queries = [{"query_id": "empty", "text": ""}]
        results = model.predict(empty_queries, sample_corpus)
        
        assert "empty" in results
        assert len(results["empty"]) == 0
    
    def test_predict_score_normalization(self, sample_config, sample_corpus, sample_queries):
        """Test prediction with score normalization."""
        model = BM25Model(sample_config)
        model.load_model()
        model.index_corpus(sample_corpus)
        
        results = model.predict(
            sample_queries, 
            sample_corpus, 
            top_k=2,
            normalize_scores=True
        )
        
        # Check that maximum score is 1.0 (normalized)
        for query_id, doc_scores in results.items():
            if doc_scores:  # Skip empty results
                max_score = max(doc_scores.values())
                assert max_score <= 1.0
                # For normalized scores, max should be 1.0
                if max_score > 0:
                    assert abs(max_score - 1.0) < 1e-6
    
    def test_predict_min_score_threshold(self, sample_config, sample_corpus, sample_queries):
        """Test prediction with minimum score threshold."""
        model = BM25Model(sample_config)
        model.load_model()
        model.index_corpus(sample_corpus)
        
        # First get normal results to see score range
        normal_results = model.predict(sample_queries, sample_corpus, top_k=10)
        
        # Set a high minimum score threshold
        high_threshold = 5.0
        filtered_results = model.predict(
            sample_queries, 
            sample_corpus, 
            top_k=10,
            min_score=high_threshold
        )
        
        # Should have fewer or equal results with threshold
        for query_id in normal_results:
            assert len(filtered_results[query_id]) <= len(normal_results[query_id])
            
            # All remaining scores should be above threshold
            for score in filtered_results[query_id].values():
                assert score >= high_threshold
    
    def test_get_model_info(self, sample_config, sample_corpus):
        """Test model information retrieval."""
        model = BM25Model(sample_config)
        model.load_model()
        model.index_corpus(sample_corpus)
        
        info = model.get_model_info()
        
        assert info["model_name"] == "test_bm25"
        assert info["model_type"] == "BM25"
        assert info["parameters"]["k1"] == 1.6
        assert info["parameters"]["b"] == 0.75
        assert info["corpus_indexed"] is True
        assert info["num_documents"] == len(sample_corpus)
        assert info["vocabulary_size"] > 0
        assert info["is_loaded"] is True
    
    def test_corpus_caching(self, sample_config, sample_corpus):
        """Test corpus caching functionality."""
        model = BM25Model(sample_config)
        model.load_model()
        
        # Index corpus first time
        model.index_corpus(sample_corpus)
        first_bm25_model = model.bm25_model
        
        # Index same corpus again (should use cache)
        with patch.object(model, '_tokenize_text') as mock_tokenize:
            model.index_corpus(sample_corpus)
            # Should not call tokenization again due to caching
            mock_tokenize.assert_not_called()
        
        # Should be the same model instance
        assert model.bm25_model is first_bm25_model
    
    def test_corpus_consistency_check(self, sample_config, sample_corpus, sample_queries):
        """Test corpus consistency checking during prediction."""
        model = BM25Model(sample_config)
        model.load_model()
        model.index_corpus(sample_corpus)
        
        # Modify corpus
        modified_corpus = sample_corpus.copy()
        modified_corpus["doc5"] = {"text": "New document"}
        
        # Should trigger re-indexing
        with patch.object(model, 'index_corpus') as mock_index:
            model.predict(sample_queries, modified_corpus)
            mock_index.assert_called_once()
    
    def test_different_tokenizers(self, sample_config):
        """Test different tokenization methods."""
        text = "Hello, World! This is a test-case."
        
        # Test simple tokenizer
        sample_config["parameters"]["tokenizer"] = "simple"
        model_simple = BM25Model(sample_config)
        tokens_simple = model_simple._tokenize_text(text)
        
        # Test regex tokenizer
        sample_config["parameters"]["tokenizer"] = "regex"
        model_regex = BM25Model(sample_config)
        tokens_regex = model_regex._tokenize_text(text)
        
        # Both should produce tokens, but potentially different
        assert len(tokens_simple) > 0
        assert len(tokens_regex) > 0
        assert isinstance(tokens_simple, list)
        assert isinstance(tokens_regex, list)
    
    def test_edge_cases(self, sample_config):
        """Test various edge cases."""
        model = BM25Model(sample_config)
        
        # Test empty text tokenization
        assert model._tokenize_text("") == []
        assert model._tokenize_text(None) == []
        
        # Test non-string input
        assert model._tokenize_text(123) == []
        
        # Test text with only punctuation
        tokens = model._tokenize_text("!@#$%^&*()")
        assert tokens == []
        
        # Test text with only short tokens
        sample_config["parameters"]["min_token_length"] = 5
        model_long = BM25Model(sample_config)
        tokens = model_long._tokenize_text("a b c d")
        assert tokens == []


# Integration tests
class TestBM25Integration:
    """Integration tests for BM25Model with realistic scenarios."""
    
    def test_realistic_ir_scenario(self):
        """Test BM25 with a realistic information retrieval scenario."""
        # Create a more realistic corpus
        corpus = {
            "paper1": {
                "text": "Deep learning has revolutionized computer vision and natural language processing. "
                       "Convolutional neural networks are particularly effective for image recognition tasks.",
                "title": "Deep Learning in Computer Vision"
            },
            "paper2": {
                "text": "Machine learning algorithms can be broadly categorized into supervised, "
                       "unsupervised, and reinforcement learning approaches.",
                "title": "Machine Learning Categories"
            },
            "paper3": {
                "text": "Natural language processing techniques enable computers to understand and "
                       "generate human language. This includes tasks like sentiment analysis and machine translation.",
                "title": "NLP Applications"
            },
            "paper4": {
                "text": "Information retrieval systems help users find relevant documents from large collections. "
                       "Search engines are the most common example of IR systems.",
                "title": "Information Retrieval Systems"
            }
        }
        
        queries = [
            {"query_id": "q1", "text": "deep learning computer vision"},
            {"query_id": "q2", "text": "machine learning supervised"},
            {"query_id": "q3", "text": "natural language processing"},
            {"query_id": "q4", "text": "information retrieval search"}
        ]
        
        config = {
            "name": "bm25_realistic",
            "type": "sparse",
            "parameters": {
                "k1": 1.2,
                "b": 0.75
            }
        }
        
        model = BM25Model(config)
        model.load_model()
        
        results = model.predict(queries, corpus, top_k=3)
        
        # Verify that relevant documents are retrieved
        # q1 about deep learning should rank paper1 highly
        assert "paper1" in results["q1"]
        
        # q2 about machine learning should rank paper2 highly  
        assert "paper2" in results["q2"]
        
        # q3 about NLP should rank paper3 highly
        assert "paper3" in results["q3"]
        
        # q4 about IR should rank paper4 highly
        assert "paper4" in results["q4"]
        
        # Check that results are reasonable
        for query_id, doc_scores in results.items():
            assert len(doc_scores) <= 3  # top_k=3
            assert all(score > 0 for score in doc_scores.values())
    
    def test_performance_with_larger_corpus(self):
        """Test BM25 performance with a larger corpus."""
        # Create a larger corpus
        large_corpus = {}
        for i in range(100):
            large_corpus[f"doc_{i}"] = {
                "text": f"This is document number {i}. It contains some content about topic {i % 10}. "
                       f"Additional text to make it more realistic. Random words: data science artificial intelligence."
            }
        
        queries = [
            {"query_id": "q1", "text": "document number"},
            {"query_id": "q2", "text": "topic data science"},
            {"query_id": "q3", "text": "artificial intelligence"}
        ]
        
        config = {
            "name": "bm25_large",
            "type": "sparse"
        }
        
        model = BM25Model(config)
        model.load_model()
        
        # Should handle larger corpus efficiently
        results = model.predict(queries, large_corpus, top_k=10)
        
        assert len(results) == len(queries)
        for query_id, doc_scores in results.items():
            assert len(doc_scores) <= 10
            assert len(doc_scores) > 0  # Should find some relevant documents


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
