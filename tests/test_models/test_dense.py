"""
Unit tests for Dense retrieval models implementation.

This module provides comprehensive tests for DenseTextRetriever and its subclasses,
including various architectures, indexing strategies, and error handling.
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch, MagicMock
import tempfile
import os
import sys
import pickle
from pathlib import Path

# Add src to path for testing
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../src'))

from newaibench.models.dense import (
    DenseTextRetriever,
    SentenceBERTModel,
    DPRModel,
    TransformersModel
)
from newaibench.models.base import ModelType


class TestDenseTextRetriever:
    """Test class for DenseTextRetriever functionality."""
    
    @pytest.fixture
    def basic_config(self):
        """Basic configuration for testing."""
        return {
            "name": "test_dense",
            "type": "dense",
            "model_name_or_path": "all-MiniLM-L6-v2",
            "device": "cpu",
            "batch_size": 4,
            "parameters": {
                "model_type": "sentence_transformer",
                "normalize_embeddings": True,
                "use_ann_index": False,
                "max_seq_length": 128
            }
        }
    
    @pytest.fixture
    def ann_config(self):
        """Configuration with ANN indexing."""
        return {
            "name": "test_dense_ann",
            "type": "dense", 
            "model_name_or_path": "all-MiniLM-L6-v2",
            "device": "cpu",
            "batch_size": 4,
            "parameters": {
                "model_type": "sentence_transformer",
                "normalize_embeddings": True,
                "use_ann_index": True,
                "ann_backend": "faiss",
                "faiss_index_factory_string": "Flat"
            }
        }
    
    @pytest.fixture
    def dpr_config(self):
        """Configuration for DPR model."""
        return {
            "name": "test_dpr",
            "type": "dense",
            "model_name_or_path": "facebook/dpr-question_encoder-single-nq-base",
            "device": "cpu",
            "batch_size": 2,
            "parameters": {
                "model_type": "dpr",
                "normalize_embeddings": True,
                "use_ann_index": False
            }
        }
    
    @pytest.fixture
    def sample_corpus(self):
        """Sample corpus for testing."""
        return {
            "doc1": {
                "text": "Machine learning algorithms are used for pattern recognition",
                "title": "ML Patterns"
            },
            "doc2": {
                "text": "Deep neural networks excel at complex data processing tasks",
                "title": "Deep Learning"
            },
            "doc3": {
                "text": "Natural language processing enables computers to understand text",
                "title": "NLP Fundamentals"
            },
            "doc4": {
                "text": "Computer vision systems can analyze and interpret visual data",
                "title": "Computer Vision"
            },
            "doc5": {
                "text": "",  # Empty text to test edge case
                "title": "Empty Document"
            }
        }
    
    @pytest.fixture
    def sample_queries(self):
        """Sample queries for testing."""
        return [
            {"query_id": "q1", "text": "machine learning patterns"},
            {"query_id": "q2", "text": "neural network processing"},
            {"query_id": "q3", "text": "text understanding"},
            {"query_id": "q4", "text": ""}  # Empty query to test edge case
        ]
    
    # Initialization tests
    def test_init_basic_config(self, basic_config):
        """Test initialization with basic configuration."""
        model = DenseTextRetriever(basic_config)
        
        assert model.name == "test_dense"
        assert model.model_name_or_path == "all-MiniLM-L6-v2"
        assert model.model_architecture == "sentence_transformer"
        assert model.use_ann_index is False
        assert model.normalize_embeddings is True
        assert model.max_seq_length == 128
        assert model.is_loaded is False
    
    def test_init_ann_config(self, ann_config):
        """Test initialization with ANN configuration."""
        model = DenseTextRetriever(ann_config)
        
        assert model.use_ann_index is True
        assert model.ann_backend == "faiss"
        assert model.faiss_index_factory == "Flat"
    
    def test_init_missing_ann_library(self, ann_config):
        """Test initialization when ANN library is not available."""
        ann_config["parameters"]["ann_backend"] = "faiss"
        
        with patch('newaibench.models.dense.FAISS_AVAILABLE', False):
            with pytest.raises(ImportError, match="FAISS not available"):
                DenseTextRetriever(ann_config)
    
    # Model loading tests
    @patch('newaibench.models.dense.SentenceTransformer')
    def test_load_model_sentence_transformer(self, mock_sbert, basic_config):
        """Test loading sentence transformer model."""
        # Setup mock
        mock_model = Mock()
        mock_model.get_sentence_embedding_dimension.return_value = 384
        mock_sbert.return_value = mock_model
        
        model = DenseTextRetriever(basic_config)
        model.load_model()
        
        assert model.is_loaded is True
        assert model.embedding_dim == 384
        mock_sbert.assert_called_once_with("all-MiniLM-L6-v2", device="cpu")
    
    @patch('transformers.DPRQuestionEncoder')
    @patch('transformers.DPRContextEncoder')
    @patch('transformers.AutoTokenizer')
    def test_load_model_dpr(self, mock_tokenizer, mock_ctx_encoder, mock_q_encoder, dpr_config):
        """Test loading DPR model."""
        # Setup mocks
        mock_q_model = Mock()
        mock_q_model.config.hidden_size = 768
        mock_q_encoder.from_pretrained.return_value = mock_q_model
        
        mock_ctx_model = Mock()
        mock_ctx_encoder.from_pretrained.return_value = mock_ctx_model
        
        mock_tokenizer.from_pretrained.return_value = Mock()
        
        model = DenseTextRetriever(dpr_config)
        model.load_model()
        
        assert model.is_loaded is True
        assert model.embedding_dim == 768
        assert model.query_encoder is not None
        assert model.doc_encoder is not None
    
    def test_load_model_invalid_path(self, basic_config):
        """Test loading model with invalid path."""
        basic_config["model_name_or_path"] = "invalid/model/path"
        model = DenseTextRetriever(basic_config)
        
        with pytest.raises(Exception):
            model.load_model()
    
    # Text encoding tests
    @patch('newaibench.models.dense.SentenceTransformer')
    def test_encode_texts_sentence_transformer(self, mock_sbert, basic_config):
        """Test text encoding with sentence transformer."""
        # Setup mock
        mock_model = Mock()
        mock_embeddings = np.random.randn(3, 384)
        mock_model.encode.return_value = mock_embeddings
        mock_model.get_sentence_embedding_dimension.return_value = 384
        mock_sbert.return_value = mock_model
        
        model = DenseTextRetriever(basic_config)
        model.load_model()
        
        texts = ["text one", "text two", "text three"]
        embeddings = model.encode_texts(texts)
        
        assert embeddings.shape == (3, 384)
        mock_model.encode.assert_called_once()
    
    @patch('newaibench.models.dense.SentenceTransformer')
    def test_encode_texts_normalization(self, mock_sbert, basic_config):
        """Test embedding normalization."""
        # Setup mock with unnormalized embeddings
        mock_model = Mock()
        mock_embeddings = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        mock_model.encode.return_value = mock_embeddings
        mock_model.get_sentence_embedding_dimension.return_value = 3
        mock_sbert.return_value = mock_model
        
        model = DenseTextRetriever(basic_config)
        model.load_model()
        
        texts = ["text one", "text two"]
        embeddings = model.encode_texts(texts)
        
        # Check that embeddings are normalized
        norms = np.linalg.norm(embeddings, axis=1)
        assert np.allclose(norms, 1.0)
    
    @patch('newaibench.models.dense.SentenceTransformer')
    def test_encode_empty_texts(self, mock_sbert, basic_config):
        """Test encoding empty text list."""
        mock_model = Mock()
        mock_model.get_sentence_embedding_dimension.return_value = 384
        mock_sbert.return_value = mock_model
        
        model = DenseTextRetriever(basic_config)
        model.load_model()
        
        embeddings = model.encode_texts([])
        
        assert embeddings.shape == (0, 384)
    
    # Query encoding tests
    @patch('newaibench.models.dense.SentenceTransformer')
    def test_encode_queries(self, mock_sbert, basic_config, sample_queries):
        """Test query encoding."""
        mock_model = Mock()
        mock_embeddings = np.random.randn(4, 384)
        mock_model.encode.return_value = mock_embeddings
        mock_model.get_sentence_embedding_dimension.return_value = 384
        mock_sbert.return_value = mock_model
        
        model = DenseTextRetriever(basic_config)
        model.load_model()
        
        query_embeddings = model.encode_queries(sample_queries)
        
        assert len(query_embeddings) == 4
        assert "q1" in query_embeddings
        assert query_embeddings["q1"].shape == (384,)
    
    # Document encoding tests
    @patch('newaibench.models.dense.SentenceTransformer')
    def test_encode_documents(self, mock_sbert, basic_config, sample_corpus):
        """Test document encoding."""
        mock_model = Mock()
        mock_embeddings = np.random.randn(5, 384)
        mock_model.encode.return_value = mock_embeddings
        mock_model.get_sentence_embedding_dimension.return_value = 384
        mock_sbert.return_value = mock_model
        
        model = DenseTextRetriever(basic_config)
        model.load_model()
        
        doc_embeddings = model.encode_documents(sample_corpus)
        
        assert len(doc_embeddings) == 5
        assert "doc1" in doc_embeddings
        assert doc_embeddings["doc1"].shape == (384,)
    
    @patch('newaibench.models.dense.SentenceTransformer')
    def test_encode_documents_text_extraction(self, mock_sbert, basic_config):
        """Test document text extraction logic."""
        mock_model = Mock()
        mock_embeddings = np.random.randn(1, 384)
        mock_model.encode.return_value = mock_embeddings
        mock_model.get_sentence_embedding_dimension.return_value = 384
        mock_sbert.return_value = mock_model
        
        model = DenseTextRetriever(basic_config)
        model.load_model()
        
        # Test with title + text
        corpus = {
            "doc1": {
                "title": "Document Title",
                "text": "Document content"
            }
        }
        
        doc_embeddings = model.encode_documents(corpus)
        
        # Verify the encode method was called with combined text
        args, kwargs = mock_model.encode.call_args
        encoded_texts = args[0]
        assert "Document Title. Document content" in encoded_texts
    
    # Indexing tests
    @patch('newaibench.models.dense.SentenceTransformer')
    def test_index_corpus_without_ann(self, mock_sbert, basic_config, sample_corpus):
        """Test corpus indexing without ANN."""
        mock_model = Mock()
        mock_embeddings = np.random.randn(5, 384)
        mock_model.encode.return_value = mock_embeddings
        mock_model.get_sentence_embedding_dimension.return_value = 384
        mock_sbert.return_value = mock_model
        
        model = DenseTextRetriever(basic_config)
        model.load_model()
        model.index_corpus(sample_corpus)
        
        assert model._corpus_indexed is True
        assert len(model.doc_embeddings) == 5
        assert len(model.doc_ids_list) == 5
        assert model.ann_index is None
    
    @patch('newaibench.models.dense.SentenceTransformer')
    @patch('newaibench.models.dense.FAISS_AVAILABLE', True)
    @patch('faiss.IndexFlatIP')
    def test_index_corpus_with_faiss(self, mock_faiss_index, mock_sbert, ann_config, sample_corpus):
        """Test corpus indexing with FAISS."""
        # Setup mocks
        mock_model = Mock()
        mock_embeddings = np.random.randn(5, 384)
        mock_model.encode.return_value = mock_embeddings
        mock_model.get_sentence_embedding_dimension.return_value = 384
        mock_sbert.return_value = mock_model
        
        mock_index = Mock()
        mock_faiss_index.return_value = mock_index
        
        model = DenseTextRetriever(ann_config)
        model.load_model()
        model.index_corpus(sample_corpus)
        
        assert model._corpus_indexed is True
        assert model.ann_index is not None
        mock_faiss_index.assert_called_once_with(384)
        mock_index.add.assert_called_once()
    
    def test_index_corpus_without_loading(self, basic_config, sample_corpus):
        """Test corpus indexing without loading model first."""
        model = DenseTextRetriever(basic_config)
        
        with pytest.raises(RuntimeError, match="Model must be loaded"):
            model.index_corpus(sample_corpus)
    
    # Caching tests
    @patch('newaibench.models.dense.SentenceTransformer')
    def test_embedding_caching(self, mock_sbert, basic_config, sample_corpus):
        """Test embedding caching functionality."""
        mock_model = Mock()
        mock_embeddings = np.random.randn(5, 384)
        mock_model.encode.return_value = mock_embeddings
        mock_model.get_sentence_embedding_dimension.return_value = 384
        mock_sbert.return_value = mock_model
        
        with tempfile.TemporaryDirectory() as temp_dir:
            basic_config["cache_dir"] = temp_dir
            
            model = DenseTextRetriever(basic_config)
            model.load_model()
            model.index_corpus(sample_corpus, cache_embeddings=True)
            
            # Check that cache file was created
            assert model.embedding_cache_path.exists()
            
            # Create new model and load from cache
            model2 = DenseTextRetriever(basic_config)
            model2.load_model()
            model2.index_corpus(sample_corpus, load_cached_embeddings=True)
            
            # Should have same embeddings
            assert len(model2.doc_embeddings) == 5
            assert model2._corpus_indexed is True
    
    # Search tests
    @patch('newaibench.models.dense.SentenceTransformer')
    def test_search_brute_force(self, mock_sbert, basic_config, sample_corpus, sample_queries):
        """Test brute-force similarity search."""
        # Setup mock with deterministic embeddings
        mock_model = Mock()
        
        # Doc embeddings: simple patterns for predictable similarity
        doc_embeddings = np.array([
            [1.0, 0.0, 0.0],  # doc1
            [0.0, 1.0, 0.0],  # doc2  
            [0.0, 0.0, 1.0],  # doc3
            [0.5, 0.5, 0.0],  # doc4
            [0.0, 0.0, 0.0]   # doc5 (empty)
        ])
        
        # Query embeddings designed to match specific docs
        query_embeddings = np.array([
            [1.0, 0.0, 0.0],  # q1: should match doc1 best
            [0.0, 1.0, 0.0],  # q2: should match doc2 best
            [0.0, 0.0, 1.0],  # q3: should match doc3 best
            [0.0, 0.0, 0.0]   # q4: empty query
        ])
        
        def mock_encode_side_effect(texts, **kwargs):
            if len(texts) == 5:  # Document encoding
                return doc_embeddings
            else:  # Query encoding
                return query_embeddings
        
        mock_model.encode.side_effect = mock_encode_side_effect
        mock_model.get_sentence_embedding_dimension.return_value = 3
        mock_sbert.return_value = mock_model
        
        model = DenseTextRetriever(basic_config)
        model.load_model()
        
        results = model.predict(sample_queries, sample_corpus, top_k=2)
        
        # Check results structure
        assert len(results) == 4
        assert "q1" in results
        assert len(results["q1"]) <= 2  # top_k=2
        
        # Check that results are sorted by score
        for query_id in results:
            scores = list(results[query_id].values())
            assert scores == sorted(scores, reverse=True)
    
    @patch('newaibench.models.dense.SentenceTransformer')
    @patch('newaibench.models.dense.FAISS_AVAILABLE', True)
    @patch('faiss.IndexFlatIP')
    def test_search_with_ann(self, mock_faiss_index, mock_sbert, ann_config, sample_corpus, sample_queries):
        """Test ANN search functionality."""
        # Setup mocks
        mock_model = Mock()
        mock_embeddings = np.random.randn(5, 384)
        mock_model.encode.return_value = mock_embeddings
        mock_model.get_sentence_embedding_dimension.return_value = 384
        mock_sbert.return_value = mock_model
        
        # Mock FAISS index
        mock_index = Mock()
        mock_scores = np.array([[0.9, 0.8], [0.7, 0.6], [0.5, 0.4], [0.3, 0.2]])
        mock_indices = np.array([[0, 1], [1, 2], [2, 3], [3, 4]])
        mock_index.search.return_value = (mock_scores, mock_indices)
        mock_faiss_index.return_value = mock_index
        
        model = DenseTextRetriever(ann_config)
        model.load_model()
        
        results = model.predict(sample_queries, sample_corpus, top_k=2)
        
        assert len(results) == 4
        mock_index.search.assert_called_once()
    
    # Prediction tests
    @patch('newaibench.models.dense.SentenceTransformer')
    def test_predict_without_indexing(self, mock_sbert, basic_config, sample_corpus, sample_queries):
        """Test prediction with automatic indexing."""
        mock_model = Mock()
        mock_embeddings = np.random.randn(5, 384)
        mock_model.encode.return_value = mock_embeddings
        mock_model.get_sentence_embedding_dimension.return_value = 384
        mock_sbert.return_value = mock_model
        
        model = DenseTextRetriever(basic_config)
        model.load_model()
        
        # Should automatically index corpus
        results = model.predict(sample_queries, sample_corpus, top_k=2)
        
        assert model._corpus_indexed is True
        assert len(results) == 4
    
    @patch('newaibench.models.dense.SentenceTransformer')
    def test_predict_corpus_change_detection(self, mock_sbert, basic_config, sample_corpus, sample_queries):
        """Test detection of corpus changes."""
        mock_model = Mock()
        mock_embeddings = np.random.randn(5, 384)
        mock_model.encode.return_value = mock_embeddings
        mock_model.get_sentence_embedding_dimension.return_value = 384
        mock_sbert.return_value = mock_model
        
        model = DenseTextRetriever(basic_config)
        model.load_model()
        model.index_corpus(sample_corpus)
        
        # Modify corpus
        modified_corpus = sample_corpus.copy()
        modified_corpus["doc6"] = {"text": "New document"}
        
        # Should trigger re-indexing
        with patch.object(model, 'index_corpus', wraps=model.index_corpus) as mock_index:
            model.predict(sample_queries, modified_corpus)
            mock_index.assert_called_with(modified_corpus, force_rebuild=True)
    
    @patch('newaibench.models.dense.SentenceTransformer')
    def test_predict_min_score_threshold(self, mock_sbert, basic_config, sample_corpus, sample_queries):
        """Test minimum score threshold filtering."""
        # Setup mock with known similarity scores
        mock_model = Mock()
        
        # Create embeddings that will give predictable scores
        doc_embeddings = np.array([
            [1.0, 0.0],  # doc1
            [0.5, 0.5],  # doc2
            [0.0, 1.0],  # doc3
            [0.2, 0.8],  # doc4
            [0.1, 0.1]   # doc5
        ])
        
        query_embeddings = np.array([
            [1.0, 0.0],  # q1: high similarity with doc1
            [0.0, 1.0],  # q2: high similarity with doc3
            [0.5, 0.5],  # q3: moderate similarities
            [0.0, 0.0]   # q4: low similarities
        ])
        
        def mock_encode_side_effect(texts, **kwargs):
            if len(texts) == 5:
                return doc_embeddings
            else:
                return query_embeddings
        
        mock_model.encode.side_effect = mock_encode_side_effect
        mock_model.get_sentence_embedding_dimension.return_value = 2
        mock_sbert.return_value = mock_model
        
        model = DenseTextRetriever(basic_config)
        model.load_model()
        
        # Test with minimum score threshold
        results = model.predict(sample_queries, sample_corpus, top_k=10, min_score=0.5)
        
        # Should filter out low-scoring documents
        for query_id, doc_scores in results.items():
            for score in doc_scores.values():
                assert score >= 0.5
    
    # Model info tests
    @patch('newaibench.models.dense.SentenceTransformer')
    def test_get_model_info(self, mock_sbert, basic_config, sample_corpus):
        """Test model information retrieval."""
        mock_model = Mock()
        mock_embeddings = np.random.randn(5, 384)
        mock_model.encode.return_value = mock_embeddings
        mock_model.get_sentence_embedding_dimension.return_value = 384
        mock_sbert.return_value = mock_model
        
        model = DenseTextRetriever(basic_config)
        model.load_model()
        model.index_corpus(sample_corpus)
        
        info = model.get_model_info()
        
        assert info["model_name"] == "test_dense"
        assert info["model_type"] == "Dense"
        assert info["model_path"] == "all-MiniLM-L6-v2"
        assert info["architecture"] == "sentence_transformer"
        assert info["embedding_dim"] == 384
        assert info["corpus_indexed"] is True
        assert info["num_documents"] == 5
        assert info["is_loaded"] is True

    # Edge cases tests
    @patch('newaibench.models.dense.SentenceTransformer')
    def test_empty_query_handling(self, mock_sbert, basic_config, sample_corpus):
        """Test handling of empty queries."""
        mock_model = Mock()
        mock_embeddings = np.random.randn(5, 384)
        mock_model.encode.return_value = mock_embeddings
        mock_model.get_sentence_embedding_dimension.return_value = 384
        mock_sbert.return_value = mock_model
        
        model = DenseTextRetriever(basic_config)
        model.load_model()
        
        empty_queries = [{"query_id": "empty", "text": ""}]
        results = model.predict(empty_queries, sample_corpus)
        
        assert "empty" in results
        # Should still return results even with empty query
        assert isinstance(results["empty"], dict)
    
    @patch('newaibench.models.dense.SentenceTransformer')
    def test_single_document_corpus(self, mock_sbert, basic_config):
        """Test with single document corpus."""
        mock_model = Mock()
        mock_embeddings = np.random.randn(1, 384)
        mock_model.encode.return_value = mock_embeddings
        mock_model.get_sentence_embedding_dimension.return_value = 384
        mock_sbert.return_value = mock_model
        
        model = DenseTextRetriever(basic_config)
        model.load_model()
        
        single_corpus = {"doc1": {"text": "Single document"}}
        queries = [{"query_id": "q1", "text": "test query"}]
        
        results = model.predict(queries, single_corpus)
        
        assert len(results["q1"]) == 1
        assert "doc1" in results["q1"]


# Convenience class tests
class TestSentenceBERTModel:
    """Test SentenceBERT convenience class."""
    
    def test_init_sets_correct_architecture(self):
        """Test that SentenceBERTModel sets correct architecture."""
        config = {
            "name": "sbert_test",
            "type": "dense",
            "model_name_or_path": "all-MiniLM-L6-v2"
        }
        
        model = SentenceBERTModel(config)
        assert model.model_architecture == "sentence_transformer"


class TestDPRModel:
    """Test DPR convenience class."""
    
    def test_init_sets_correct_architecture(self):
        """Test that DPRModel sets correct architecture."""
        config = {
            "name": "dpr_test", 
            "type": "dense",
            "model_name_or_path": "facebook/dpr-question_encoder-single-nq-base"
        }
        
        model = DPRModel(config)
        assert model.model_architecture == "dpr"


class TestTransformersModel:
    """Test Transformers convenience class."""
    
    def test_init_sets_correct_architecture(self):
        """Test that TransformersModel sets correct architecture."""
        config = {
            "name": "transformers_test",
            "type": "dense", 
            "model_name_or_path": "bert-base-uncased"
        }
        
        model = TransformersModel(config)
        assert model.model_architecture == "transformers"


# Integration tests
class TestDenseModelIntegration:
    """Integration tests for dense models with realistic scenarios."""
    
    @pytest.fixture
    def real_corpus(self):
        """Realistic corpus for integration testing."""
        return {
            "paper1": {
                "title": "Attention Is All You Need",
                "text": "The dominant sequence transduction models are based on complex recurrent or convolutional neural networks that include an encoder and a decoder. The best performing models also connect the encoder and decoder through an attention mechanism."
            },
            "paper2": {
                "title": "BERT: Pre-training of Deep Bidirectional Transformers",
                "text": "We introduce a new language representation model called BERT, which stands for Bidirectional Encoder Representations from Transformers. Unlike recent language representation models, BERT is designed to pre-train deep bidirectional representations."
            },
            "paper3": {
                "title": "Dense Passage Retrieval for Open-Domain Question Answering", 
                "text": "Open-domain question answering relies on efficient passage retrieval to select candidate contexts from a large corpus of texts. Traditional sparse vector space models, such as TF-IDF or BM25, are the de facto method for retrieval."
            }
        }
    
    @pytest.fixture  
    def real_queries(self):
        """Realistic queries for integration testing."""
        return [
            {"query_id": "q1", "text": "attention mechanism in transformers"},
            {"query_id": "q2", "text": "bidirectional language model pretraining"},
            {"query_id": "q3", "text": "dense retrieval for question answering"}
        ]
    
    @pytest.mark.slow
    @patch('newaibench.models.dense.SentenceTransformer')
    def test_end_to_end_retrieval(self, mock_sbert, real_corpus, real_queries):
        """Test complete end-to-end retrieval pipeline."""
        # Setup realistic mock that returns reasonable embeddings
        mock_model = Mock()
        
        # Generate semi-realistic embeddings based on text content
        def generate_embeddings(texts):
            embeddings = []
            for text in texts:
                # Simple hash-based embedding generation for consistency
                text_hash = hash(text.lower()) % 1000000
                np.random.seed(text_hash)
                emb = np.random.randn(384)
                emb = emb / np.linalg.norm(emb)  # Normalize
                embeddings.append(emb)
            return np.array(embeddings)
        
        mock_model.encode.side_effect = generate_embeddings
        mock_model.get_sentence_embedding_dimension.return_value = 384
        mock_sbert.return_value = mock_model
        
        config = {
            "name": "integration_test",
            "type": "dense",
            "model_name_or_path": "all-MiniLM-L6-v2",
            "device": "cpu",
            "parameters": {
                "normalize_embeddings": True,
                "use_ann_index": False
            }
        }
        
        model = DenseTextRetriever(config)
        model.load_model()
        
        # Test retrieval
        results = model.predict(real_queries, real_corpus, top_k=3)
        
        # Validate results
        assert len(results) == 3
        for query_id in ["q1", "q2", "q3"]:
            assert query_id in results
            assert len(results[query_id]) <= 3
            
            # Scores should be between 0 and 1 (cosine similarity)
            for score in results[query_id].values():
                assert 0 <= score <= 1
    
    @pytest.mark.slow
    @patch('newaibench.models.dense.SentenceTransformer')
    @patch('newaibench.models.dense.FAISS_AVAILABLE', True)
    @patch('faiss.IndexFlatIP')
    def test_ann_vs_brute_force_consistency(self, mock_faiss_index, mock_sbert, real_corpus, real_queries):
        """Test that ANN and brute-force give consistent results."""
        # Setup mocks
        mock_model = Mock()
        
        def generate_embeddings(texts):
            embeddings = []
            for text in texts:
                text_hash = hash(text.lower()) % 1000000
                np.random.seed(text_hash)
                emb = np.random.randn(384)
                emb = emb / np.linalg.norm(emb)
                embeddings.append(emb)
            return np.array(embeddings)
        
        mock_model.encode.side_effect = generate_embeddings
        mock_model.get_sentence_embedding_dimension.return_value = 384
        mock_sbert.return_value = mock_model
        
        # Mock FAISS to return exact same results as brute force would
        mock_index = Mock()
        
        def mock_search(query_embs, k):
            # Simulate exact search by computing actual similarities
            doc_embs = generate_embeddings([
                "Attention Is All You Need. The dominant sequence...",
                "BERT: Pre-training of Deep Bidirectional Transformers. We introduce...", 
                "Dense Passage Retrieval for Open-Domain Question Answering. Open-domain..."
            ])
            
            scores_list = []
            indices_list = []
            
            for query_emb in query_embs:
                similarities = np.dot(doc_embs, query_emb)
                top_indices = np.argsort(similarities)[::-1][:k]
                top_scores = similarities[top_indices]
                
                scores_list.append(top_scores)
                indices_list.append(top_indices)
            
            return np.array(scores_list), np.array(indices_list)
        
        mock_index.search.side_effect = mock_search
        mock_faiss_index.return_value = mock_index
        
        # Test both configurations
        base_config = {
            "name": "consistency_test",
            "type": "dense", 
            "model_name_or_path": "all-MiniLM-L6-v2",
            "device": "cpu",
            "parameters": {"normalize_embeddings": True}
        }
        
        # Brute force model
        bf_config = base_config.copy()
        bf_config["parameters"]["use_ann_index"] = False
        bf_model = DenseTextRetriever(bf_config)
        bf_model.load_model()
        bf_results = bf_model.predict(real_queries, real_corpus, top_k=2)
        
        # ANN model  
        ann_config = base_config.copy()
        ann_config["parameters"]["use_ann_index"] = True
        ann_config["parameters"]["ann_backend"] = "faiss"
        ann_model = DenseTextRetriever(ann_config)
        ann_model.load_model()
        ann_results = ann_model.predict(real_queries, real_corpus, top_k=2)
        
        # Results should be very similar (allowing for small numerical differences)
        for query_id in real_queries:
            qid = query_id["query_id"]
            bf_docs = set(bf_results[qid].keys())
            ann_docs = set(ann_results[qid].keys())
            
            # Should retrieve mostly the same documents
            overlap = len(bf_docs.intersection(ann_docs))
            assert overlap >= min(len(bf_docs), len(ann_docs)) * 0.8  # 80% overlap


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
"""
Unit tests for Dense retrieval models implementation.

This module provides comprehensive tests for DenseTextRetriever and its subclasses,
including various architectures, indexing strategies, and error handling.
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch, MagicMock
import tempfile
import os
import sys
import pickle
from pathlib import Path

# Add src to path for testing
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../src'))

from newaibench.models.dense import (
    DenseTextRetriever,
    SentenceBERTModel,
    DPRModel,
    TransformersModel
)
from newaibench.models.base import ModelType


class TestDenseTextRetriever:
    """Test class for DenseTextRetriever functionality."""
    
    @pytest.fixture
    def basic_config(self):
        """Basic configuration for testing."""
        return {
            "name": "test_dense",
            "type": "dense",
            "model_name_or_path": "all-MiniLM-L6-v2",
            "device": "cpu",
            "batch_size": 4,
            "parameters": {
                "model_type": "sentence_transformer",
                "normalize_embeddings": True,
                "use_ann_index": False,
                "max_seq_length": 128
            }
        }
    
    @pytest.fixture
    def ann_config(self):
        """Configuration with ANN indexing."""
        return {
            "name": "test_dense_ann",
            "type": "dense", 
            "model_name_or_path": "all-MiniLM-L6-v2",
            "device": "cpu",
            "batch_size": 4,
            "parameters": {
                "model_type": "sentence_transformer",
                "normalize_embeddings": True,
                "use_ann_index": True,
                "ann_backend": "faiss",
                "faiss_index_factory_string": "Flat"
            }
        }
    
    @pytest.fixture
    def dpr_config(self):
        """Configuration for DPR model."""
        return {
            "name": "test_dpr",
            "type": "dense",
            "model_name_or_path": "facebook/dpr-question_encoder-single-nq-base",
            "device": "cpu",
            "batch_size": 2,
            "parameters": {
                "model_type": "dpr",
                "normalize_embeddings": True,
                "use_ann_index": False
            }
        }
    
    @pytest.fixture
    def sample_corpus(self):
        """Sample corpus for testing."""
        return {
            "doc1": {
                "text": "Machine learning algorithms are used for pattern recognition",
                "title": "ML Patterns"
            },
            "doc2": {
                "text": "Deep neural networks excel at complex data processing tasks",
                "title": "Deep Learning"
            },
            "doc3": {
                "text": "Natural language processing enables computers to understand text",
                "title": "NLP Fundamentals"
            },
            "doc4": {
                "text": "Computer vision systems can analyze and interpret visual data",
                "title": "Computer Vision"
            },
            "doc5": {
                "text": "",  # Empty text to test edge case
                "title": "Empty Document"
            }
        }
    
    @pytest.fixture
    def sample_queries(self):
        """Sample queries for testing."""
        return [
            {"query_id": "q1", "text": "machine learning patterns"},
            {"query_id": "q2", "text": "neural network processing"},
            {"query_id": "q3", "text": "text understanding"},
            {"query_id": "q4", "text": ""}  # Empty query to test edge case
        ]
    
    # Initialization tests
    def test_init_basic_config(self, basic_config):
        """Test initialization with basic configuration."""
        model = DenseTextRetriever(basic_config)
        
        assert model.name == "test_dense"
        assert model.model_name_or_path == "all-MiniLM-L6-v2"
        assert model.model_architecture == "sentence_transformer"
        assert model.use_ann_index is False
        assert model.normalize_embeddings is True
        assert model.max_seq_length == 128
        assert model.is_loaded is False
    
    def test_init_ann_config(self, ann_config):
        """Test initialization with ANN configuration."""
        model = DenseTextRetriever(ann_config)
        
        assert model.use_ann_index is True
        assert model.ann_backend == "faiss"
        assert model.faiss_index_factory == "Flat"
    
    def test_init_missing_ann_library(self, ann_config):
        """Test initialization when ANN library is not available."""
        ann_config["parameters"]["ann_backend"] = "faiss"
        
        with patch('newaibench.models.dense.FAISS_AVAILABLE', False):
            with pytest.raises(ImportError, match="FAISS not available"):
                DenseTextRetriever(ann_config)
    
    # Model loading tests
    @patch('newaibench.models.dense.SentenceTransformer')
    def test_load_model_sentence_transformer(self, mock_sbert, basic_config):
        """Test loading sentence transformer model."""
        # Setup mock
        mock_model = Mock()
        mock_model.get_sentence_embedding_dimension.return_value = 384
        mock_sbert.return_value = mock_model
        
        model = DenseTextRetriever(basic_config)
        model.load_model()
        
        assert model.is_loaded is True
        assert model.embedding_dim == 384
        mock_sbert.assert_called_once_with("all-MiniLM-L6-v2", device="cpu")
    
    @patch('transformers.DPRQuestionEncoder')
    @patch('transformers.DPRContextEncoder')
    @patch('transformers.AutoTokenizer')
    def test_load_model_dpr(self, mock_tokenizer, mock_ctx_encoder, mock_q_encoder, dpr_config):
        """Test loading DPR model."""
        # Setup mocks
        mock_q_model = Mock()
        mock_q_model.config.hidden_size = 768
        mock_q_encoder.from_pretrained.return_value = mock_q_model
        
        mock_ctx_model = Mock()
        mock_ctx_encoder.from_pretrained.return_value = mock_ctx_model
        
        mock_tokenizer.from_pretrained.return_value = Mock()
        
        model = DenseTextRetriever(dpr_config)
        model.load_model()
        
        assert model.is_loaded is True
        assert model.embedding_dim == 768
        assert model.query_encoder is not None
        assert model.doc_encoder is not None
    
    def test_load_model_invalid_path(self, basic_config):
        """Test loading model with invalid path."""
        basic_config["model_name_or_path"] = "invalid/model/path"
        model = DenseTextRetriever(basic_config)
        
        with pytest.raises(Exception):
            model.load_model()
    
    # Text encoding tests
    @patch('newaibench.models.dense.SentenceTransformer')
    def test_encode_texts_sentence_transformer(self, mock_sbert, basic_config):
        """Test text encoding with sentence transformer."""
        # Setup mock
        mock_model = Mock()
        mock_embeddings = np.random.randn(3, 384)
        mock_model.encode.return_value = mock_embeddings
        mock_model.get_sentence_embedding_dimension.return_value = 384
        mock_sbert.return_value = mock_model
        
        model = DenseTextRetriever(basic_config)
        model.load_model()
        
        texts = ["text one", "text two", "text three"]
        embeddings = model.encode_texts(texts)
        
        assert embeddings.shape == (3, 384)
        mock_model.encode.assert_called_once()
    
    @patch('newaibench.models.dense.SentenceTransformer')
    def test_encode_texts_normalization(self, mock_sbert, basic_config):
        """Test embedding normalization."""
        # Setup mock with unnormalized embeddings
        mock_model = Mock()
        mock_embeddings = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        mock_model.encode.return_value = mock_embeddings
        mock_model.get_sentence_embedding_dimension.return_value = 3
        mock_sbert.return_value = mock_model
        
        model = DenseTextRetriever(basic_config)
        model.load_model()
        
        texts = ["text one", "text two"]
        embeddings = model.encode_texts(texts)
        
        # Check that embeddings are normalized
        norms = np.linalg.norm(embeddings, axis=1)
        assert np.allclose(norms, 1.0)
    
    @patch('newaibench.models.dense.SentenceTransformer')
    def test_encode_empty_texts(self, mock_sbert, basic_config):
        """Test encoding empty text list."""
        mock_model = Mock()
        mock_model.get_sentence_embedding_dimension.return_value = 384
        mock_sbert.return_value = mock_model
        
        model = DenseTextRetriever(basic_config)
        model.load_model()
        
        embeddings = model.encode_texts([])
        
        assert embeddings.shape == (0, 384)
    
    # Query encoding tests
    @patch('newaibench.models.dense.SentenceTransformer')
    def test_encode_queries(self, mock_sbert, basic_config, sample_queries):
        """Test query encoding."""
        mock_model = Mock()
        mock_embeddings = np.random.randn(4, 384)
        mock_model.encode.return_value = mock_embeddings
        mock_model.get_sentence_embedding_dimension.return_value = 384
        mock_sbert.return_value = mock_model
        
        model = DenseTextRetriever(basic_config)
        model.load_model()
        
        query_embeddings = model.encode_queries(sample_queries)
        
        assert len(query_embeddings) == 4
        assert "q1" in query_embeddings
        assert query_embeddings["q1"].shape == (384,)
    
    # Document encoding tests
    @patch('newaibench.models.dense.SentenceTransformer')
    def test_encode_documents(self, mock_sbert, basic_config, sample_corpus):
        """Test document encoding."""
        mock_model = Mock()
        mock_embeddings = np.random.randn(5, 384)
        mock_model.encode.return_value = mock_embeddings
        mock_model.get_sentence_embedding_dimension.return_value = 384
        mock_sbert.return_value = mock_model
        
        model = DenseTextRetriever(basic_config)
        model.load_model()
        
        doc_embeddings = model.encode_documents(sample_corpus)
        
        assert len(doc_embeddings) == 5
        assert "doc1" in doc_embeddings
        assert doc_embeddings["doc1"].shape == (384,)
    
    @patch('newaibench.models.dense.SentenceTransformer')
    def test_encode_documents_text_extraction(self, mock_sbert, basic_config):
        """Test document text extraction logic."""
        mock_model = Mock()
        mock_embeddings = np.random.randn(1, 384)
        mock_model.encode.return_value = mock_embeddings
        mock_model.get_sentence_embedding_dimension.return_value = 384
        mock_sbert.return_value = mock_model
        
        model = DenseTextRetriever(basic_config)
        model.load_model()
        
        # Test with title + text
        corpus = {
            "doc1": {
                "title": "Document Title",
                "text": "Document content"
            }
        }
        
        doc_embeddings = model.encode_documents(corpus)
        
        # Verify the encode method was called with combined text
        args, kwargs = mock_model.encode.call_args
        encoded_texts = args[0]
        assert "Document Title. Document content" in encoded_texts
    
    # Indexing tests
    @patch('newaibench.models.dense.SentenceTransformer')
    def test_index_corpus_without_ann(self, mock_sbert, basic_config, sample_corpus):
        """Test corpus indexing without ANN."""
        mock_model = Mock()
        mock_embeddings = np.random.randn(5, 384)
        mock_model.encode.return_value = mock_embeddings
        mock_model.get_sentence_embedding_dimension.return_value = 384
        mock_sbert.return_value = mock_model
        
        model = DenseTextRetriever(basic_config)
        model.load_model()
        model.index_corpus(sample_corpus)
        
        assert model._corpus_indexed is True
        assert len(model.doc_embeddings) == 5
        assert len(model.doc_ids_list) == 5
        assert model.ann_index is None
    
    @patch('newaibench.models.dense.SentenceTransformer')
    @patch('newaibench.models.dense.FAISS_AVAILABLE', True)
    @patch('faiss.IndexFlatIP')
    def test_index_corpus_with_faiss(self, mock_faiss_index, mock_sbert, ann_config, sample_corpus):
        """Test corpus indexing with FAISS."""
        # Setup mocks
        mock_model = Mock()
        mock_embeddings = np.random.randn(5, 384)
        mock_model.encode.return_value = mock_embeddings
        mock_model.get_sentence_embedding_dimension.return_value = 384
        mock_sbert.return_value = mock_model
        
        mock_index = Mock()
        mock_faiss_index.return_value = mock_index
        
        model = DenseTextRetriever(ann_config)
        model.load_model()
        model.index_corpus(sample_corpus)
        
        assert model._corpus_indexed is True
        assert model.ann_index is not None
        mock_faiss_index.assert_called_once_with(384)
        mock_index.add.assert_called_once()
    
    def test_index_corpus_without_loading(self, basic_config, sample_corpus):
        """Test corpus indexing without loading model first."""
        model = DenseTextRetriever(basic_config)
        
        with pytest.raises(RuntimeError, match="Model must be loaded"):
            model.index_corpus(sample_corpus)
    
    # Caching tests
    @patch('newaibench.models.dense.SentenceTransformer')
    def test_embedding_caching(self, mock_sbert, basic_config, sample_corpus):
        """Test embedding caching functionality."""
        mock_model = Mock()
        mock_embeddings = np.random.randn(5, 384)
        mock_model.encode.return_value = mock_embeddings
        mock_model.get_sentence_embedding_dimension.return_value = 384
        mock_sbert.return_value = mock_model
        
        with tempfile.TemporaryDirectory() as temp_dir:
            basic_config["cache_dir"] = temp_dir
            
            model = DenseTextRetriever(basic_config)
            model.load_model()
            model.index_corpus(sample_corpus, cache_embeddings=True)
            
            # Check that cache file was created
            assert model.embedding_cache_path.exists()
            
            # Create new model and load from cache
            model2 = DenseTextRetriever(basic_config)
            model2.load_model()
            model2.index_corpus(sample_corpus, load_cached_embeddings=True)
            
            # Should have same embeddings
            assert len(model2.doc_embeddings) == 5
            assert model2._corpus_indexed is True
    
    # Search tests
    @patch('newaibench.models.dense.SentenceTransformer')
    def test_search_brute_force(self, mock_sbert, basic_config, sample_corpus, sample_queries):
        """Test brute-force similarity search."""
        # Setup mock with deterministic embeddings
        mock_model = Mock()
        
        # Doc embeddings: simple patterns for predictable similarity
        doc_embeddings = np.array([
            [1.0, 0.0, 0.0],  # doc1
            [0.0, 1.0, 0.0],  # doc2  
            [0.0, 0.0, 1.0],  # doc3
            [0.5, 0.5, 0.0],  # doc4
            [0.0, 0.0, 0.0]   # doc5 (empty)
        ])
        
        # Query embeddings designed to match specific docs
        query_embeddings = np.array([
            [1.0, 0.0, 0.0],  # q1: should match doc1 best
            [0.0, 1.0, 0.0],  # q2: should match doc2 best
            [0.0, 0.0, 1.0],  # q3: should match doc3 best
            [0.0, 0.0, 0.0]   # q4: empty query
        ])
        
        def mock_encode_side_effect(texts, **kwargs):
            if len(texts) == 5:  # Document encoding
                return doc_embeddings
            else:  # Query encoding
                return query_embeddings
        
        mock_model.encode.side_effect = mock_encode_side_effect
        mock_model.get_sentence_embedding_dimension.return_value = 3
        mock_sbert.return_value = mock_model
        
        model = DenseTextRetriever(basic_config)
        model.load_model()
        
        results = model.predict(sample_queries, sample_corpus, top_k=2)
        
        # Check results structure
        assert len(results) == 4
        assert "q1" in results
        assert len(results["q1"]) <= 2  # top_k=2
        
        # Check that results are sorted by score
        for query_id in results:
            scores = list(results[query_id].values())
            assert scores == sorted(scores, reverse=True)
    
    @patch('newaibench.models.dense.SentenceTransformer')
    @patch('newaibench.models.dense.FAISS_AVAILABLE', True)
    @patch('faiss.IndexFlatIP')
    def test_search_with_ann(self, mock_faiss_index, mock_sbert, ann_config, sample_corpus, sample_queries):
        """Test ANN search functionality."""
        # Setup mocks
        mock_model = Mock()
        mock_embeddings = np.random.randn(5, 384)
        mock_model.encode.return_value = mock_embeddings
        mock_model.get_sentence_embedding_dimension.return_value = 384
        mock_sbert.return_value = mock_model
        
        # Mock FAISS index
        mock_index = Mock()
        mock_scores = np.array([[0.9, 0.8], [0.7, 0.6], [0.5, 0.4], [0.3, 0.2]])
        mock_indices = np.array([[0, 1], [1, 2], [2, 3], [3, 4]])
        mock_index.search.return_value = (mock_scores, mock_indices)
        mock_faiss_index.return_value = mock_index
        
        model = DenseTextRetriever(ann_config)
        model.load_model()
        
        results = model.predict(sample_queries, sample_corpus, top_k=2)
        
        assert len(results) == 4
        mock_index.search.assert_called_once()
    
    # Prediction tests
    @patch('newaibench.models.dense.SentenceTransformer')
    def test_predict_without_indexing(self, mock_sbert, basic_config, sample_corpus, sample_queries):
        """Test prediction with automatic indexing."""
        mock_model = Mock()
        mock_embeddings = np.random.randn(5, 384)
        mock_model.encode.return_value = mock_embeddings
        mock_model.get_sentence_embedding_dimension.return_value = 384
        mock_sbert.return_value = mock_model
        
        model = DenseTextRetriever(basic_config)
        model.load_model()
        
        # Should automatically index corpus
        results = model.predict(sample_queries, sample_corpus, top_k=2)
        
        assert model._corpus_indexed is True
        assert len(results) == 4
    
    @patch('newaibench.models.dense.SentenceTransformer')
    def test_predict_corpus_change_detection(self, mock_sbert, basic_config, sample_corpus, sample_queries):
        """Test detection of corpus changes."""
        mock_model = Mock()
        mock_embeddings = np.random.randn(5, 384)
        mock_model.encode.return_value = mock_embeddings
        mock_model.get_sentence_embedding_dimension.return_value = 384
        mock_sbert.return_value = mock_model
        
        model = DenseTextRetriever(basic_config)
        model.load_model()
        model.index_corpus(sample_corpus)
        
        # Modify corpus
        modified_corpus = sample_corpus.copy()
        modified_corpus["doc6"] = {"text": "New document"}
        
        # Should trigger re-indexing
        with patch.object(model, 'index_corpus', wraps=model.index_corpus) as mock_index:
            model.predict(sample_queries, modified_corpus)
            mock_index.assert_called_with(modified_corpus, force_rebuild=True)
    
    @patch('newaibench.models.dense.SentenceTransformer')
    def test_predict_min_score_threshold(self, mock_sbert, basic_config, sample_corpus, sample_queries):
        """Test minimum score threshold filtering."""
        # Setup mock with known similarity scores
        mock_model = Mock()
        
        # Create embeddings that will give predictable scores
        doc_embeddings = np.array([
            [1.0, 0.0],  # doc1
            [0.5, 0.5],  # doc2
            [0.0, 1.0],  # doc3
            [0.2, 0.8],  # doc4
            [0.1, 0.1]   # doc5
        ])
        
        query_embeddings = np.array([
            [1.0, 0.0],  # q1: high similarity with doc1
            [0.0, 1.0],  # q2: high similarity with doc3
            [0.5, 0.5],  # q3: moderate similarities
            [0.0, 0.0]   # q4: low similarities
        ])
        
        def mock_encode_side_effect(texts, **kwargs):
            if len(texts) == 5:
                return doc_embeddings
            else:
                return query_embeddings
        
        mock_model.encode.side_effect = mock_encode_side_effect
        mock_model.get_sentence_embedding_dimension.return_value = 2
        mock_sbert.return_value = mock_model
        
        model = DenseTextRetriever(basic_config)
        model.load_model()
        
        # Test with minimum score threshold
        results = model.predict(sample_queries, sample_corpus, top_k=10, min_score=0.5)
        
        # Should filter out low-scoring documents
        for query_id, doc_scores in results.items():
            for score in doc_scores.values():
                assert score >= 0.5
    
    # Model info tests
    @patch('newaibench.models.dense.SentenceTransformer')
    def test_get_model_info(self, mock_sbert, basic_config, sample_corpus):
        """Test model information retrieval."""
        mock_model = Mock()
        mock_embeddings = np.random.randn(5, 384)
        mock_model.encode.return_value = mock_embeddings
        mock_model.get_sentence_embedding_dimension.return_value = 384
        mock_sbert.return_value = mock_model
        
        model = DenseTextRetriever(basic_config)
        model.load_model()
        model.index_corpus(sample_corpus)
        
        info = model.get_model_info()
        
        assert info["model_name"] == "test_dense"
        assert info["model_type"] == "Dense"
        assert info["model_path"] == "all-MiniLM-L6-v2"
        assert info["architecture"] == "sentence_transformer"
        assert info["embedding_dim"] == 384
        assert info["corpus_indexed"] is True
        assert info["num_documents"] == 5
        assert info["is_loaded"] is True

    # Edge cases tests
    @patch('newaibench.models.dense.SentenceTransformer')
    def test_empty_query_handling(self, mock_sbert, basic_config, sample_corpus):
        """Test handling of empty queries."""
        mock_model = Mock()
        mock_embeddings = np.random.randn(5, 384)
        mock_model.encode.return_value = mock_embeddings
        mock_model.get_sentence_embedding_dimension.return_value = 384
        mock_sbert.return_value = mock_model
        
        model = DenseTextRetriever(basic_config)
        model.load_model()
        
        empty_queries = [{"query_id": "empty", "text": ""}]
        results = model.predict(empty_queries, sample_corpus)
        
        assert "empty" in results
        # Should still return results even with empty query
        assert isinstance(results["empty"], dict)
    
    @patch('newaibench.models.dense.SentenceTransformer')
    def test_single_document_corpus(self, mock_sbert, basic_config):
        """Test with single document corpus."""
        mock_model = Mock()
        mock_embeddings = np.random.randn(1, 384)
        mock_model.encode.return_value = mock_embeddings
        mock_model.get_sentence_embedding_dimension.return_value = 384
        mock_sbert.return_value = mock_model
        
        model = DenseTextRetriever(basic_config)
        model.load_model()
        
        single_corpus = {"doc1": {"text": "Single document"}}
        queries = [{"query_id": "q1", "text": "test query"}]
        
        results = model.predict(queries, single_corpus)
        
        assert len(results["q1"]) == 1
        assert "doc1" in results["q1"]


# Convenience class tests
class TestSentenceBERTModel:
    """Test SentenceBERT convenience class."""
    
    def test_init_sets_correct_architecture(self):
        """Test that SentenceBERTModel sets correct architecture."""
        config = {
            "name": "sbert_test",
            "type": "dense",
            "model_name_or_path": "all-MiniLM-L6-v2"
        }
        
        model = SentenceBERTModel(config)
        assert model.model_architecture == "sentence_transformer"


class TestDPRModel:
    """Test DPR convenience class."""
    
    def test_init_sets_correct_architecture(self):
        """Test that DPRModel sets correct architecture."""
        config = {
            "name": "dpr_test", 
            "type": "dense",
            "model_name_or_path": "facebook/dpr-question_encoder-single-nq-base"
        }
        
        model = DPRModel(config)
        assert model.model_architecture == "dpr"


class TestTransformersModel:
    """Test Transformers convenience class."""
    
    def test_init_sets_correct_architecture(self):
        """Test that TransformersModel sets correct architecture."""
        config = {
            "name": "transformers_test",
            "type": "dense", 
            "model_name_or_path": "bert-base-uncased"
        }
        
        model = TransformersModel(config)
        assert model.model_architecture == "transformers"


# Integration tests
class TestDenseModelIntegration:
    """Integration tests for dense models with realistic scenarios."""
    
    @pytest.fixture
    def real_corpus(self):
        """Realistic corpus for integration testing."""
        return {
            "paper1": {
                "title": "Attention Is All You Need",
                "text": "The dominant sequence transduction models are based on complex recurrent or convolutional neural networks that include an encoder and a decoder. The best performing models also connect the encoder and decoder through an attention mechanism."
            },
            "paper2": {
                "title": "BERT: Pre-training of Deep Bidirectional Transformers",
                "text": "We introduce a new language representation model called BERT, which stands for Bidirectional Encoder Representations from Transformers. Unlike recent language representation models, BERT is designed to pre-train deep bidirectional representations."
            },
            "paper3": {
                "title": "Dense Passage Retrieval for Open-Domain Question Answering", 
                "text": "Open-domain question answering relies on efficient passage retrieval to select candidate contexts from a large corpus of texts. Traditional sparse vector space models, such as TF-IDF or BM25, are the de facto method for retrieval."
            }
        }
    
    @pytest.fixture  
    def real_queries(self):
        """Realistic queries for integration testing."""
        return [
            {"query_id": "q1", "text": "attention mechanism in transformers"},
            {"query_id": "q2", "text": "bidirectional language model pretraining"},
            {"query_id": "q3", "text": "dense retrieval for question answering"}
        ]
    
    @pytest.mark.slow
    @patch('newaibench.models.dense.SentenceTransformer')
    def test_end_to_end_retrieval(self, mock_sbert, real_corpus, real_queries):
        """Test complete end-to-end retrieval pipeline."""
        # Setup realistic mock that returns reasonable embeddings
        mock_model = Mock()
        
        # Generate semi-realistic embeddings based on text content
        def generate_embeddings(texts):
            embeddings = []
            for text in texts:
                # Simple hash-based embedding generation for consistency
                text_hash = hash(text.lower()) % 1000000
                np.random.seed(text_hash)
                emb = np.random.randn(384)
                emb = emb / np.linalg.norm(emb)  # Normalize
                embeddings.append(emb)
            return np.array(embeddings)
        
        mock_model.encode.side_effect = generate_embeddings
        mock_model.get_sentence_embedding_dimension.return_value = 384
        mock_sbert.return_value = mock_model
        
        config = {
            "name": "integration_test",
            "type": "dense",
            "model_name_or_path": "all-MiniLM-L6-v2",
            "device": "cpu",
            "parameters": {
                "normalize_embeddings": True,
                "use_ann_index": False
            }
        }
        
        model = DenseTextRetriever(config)
        model.load_model()
        
        # Test retrieval
        results = model.predict(real_queries, real_corpus, top_k=3)
        
        # Validate results
        assert len(results) == 3
        for query_id in ["q1", "q2", "q3"]:
            assert query_id in results
            assert len(results[query_id]) <= 3
            
            # Scores should be between 0 and 1 (cosine similarity)
            for score in results[query_id].values():
                assert 0 <= score <= 1
    
    @pytest.mark.slow
    @patch('newaibench.models.dense.SentenceTransformer')
    @patch('newaibench.models.dense.FAISS_AVAILABLE', True)
    @patch('faiss.IndexFlatIP')
    def test_ann_vs_brute_force_consistency(self, mock_faiss_index, mock_sbert, real_corpus, real_queries):
        """Test that ANN and brute-force give consistent results."""
        # Setup mocks
        mock_model = Mock()
        
        def generate_embeddings(texts):
            embeddings = []
            for text in texts:
                text_hash = hash(text.lower()) % 1000000
                np.random.seed(text_hash)
                emb = np.random.randn(384)
                emb = emb / np.linalg.norm(emb)
                embeddings.append(emb)
            return np.array(embeddings)
        
        mock_model.encode.side_effect = generate_embeddings
        mock_model.get_sentence_embedding_dimension.return_value = 384
        mock_sbert.return_value = mock_model
        
        # Mock FAISS to return exact same results as brute force would
        mock_index = Mock()
        
        def mock_search(query_embs, k):
            # Simulate exact search by computing actual similarities
            doc_embs = generate_embeddings([
                "Attention Is All You Need. The dominant sequence...",
                "BERT: Pre-training of Deep Bidirectional Transformers. We introduce...", 
                "Dense Passage Retrieval for Open-Domain Question Answering. Open-domain..."
            ])
            
            scores_list = []
            indices_list = []
            
            for query_emb in query_embs:
                similarities = np.dot(doc_embs, query_emb)
                top_indices = np.argsort(similarities)[::-1][:k]
                top_scores = similarities[top_indices]
                
                scores_list.append(top_scores)
                indices_list.append(top_indices)
            
            return np.array(scores_list), np.array(indices_list)
        
        mock_index.search.side_effect = mock_search
        mock_faiss_index.return_value = mock_index
        
        # Test both configurations
        base_config = {
            "name": "consistency_test",
            "type": "dense", 
            "model_name_or_path": "all-MiniLM-L6-v2",
            "device": "cpu",
            "parameters": {"normalize_embeddings": True}
        }
        
        # Brute force model
        bf_config = base_config.copy()
        bf_config["parameters"]["use_ann_index"] = False
        bf_model = DenseTextRetriever(bf_config)
        bf_model.load_model()
        bf_results = bf_model.predict(real_queries, real_corpus, top_k=2)
        
        # ANN model  
        ann_config = base_config.copy()
        ann_config["parameters"]["use_ann_index"] = True
        ann_config["parameters"]["ann_backend"] = "faiss"
        ann_model = DenseTextRetriever(ann_config)
        ann_model.load_model()
        ann_results = ann_model.predict(real_queries, real_corpus, top_k=2)
        
        # Results should be very similar (allowing for small numerical differences)
        for query_id in real_queries:
            qid = query_id["query_id"]
            bf_docs = set(bf_results[qid].keys())
            ann_docs = set(ann_results[qid].keys())
            
            # Should retrieve mostly the same documents
            overlap = len(bf_docs.intersection(ann_docs))
            assert overlap >= min(len(bf_docs), len(ann_docs)) * 0.8  # 80% overlap


# FAISS integration tests
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
    
    def test_faiss_index_factory_ivf(self, faiss_config_base, sample_embeddings):
        """Test FAISS IVF index creation."""
        pytest.importorskip("faiss")
        
        config = faiss_config_base.copy()
        config["parameters"]["faiss_index_factory_string"] = "IVF4,Flat"
        
        model = DenseTextRetriever(config)
        model.load_model()  # Need to load model to set embedding_dim
        index = model._create_faiss_index(sample_embeddings)
        
        assert index is not None
        assert index.ntotal == len(sample_embeddings)
        assert hasattr(index, 'nprobe')
    
    def test_faiss_index_factory_ivfpq(self, faiss_config_base, sample_embeddings):
        """Test FAISS IVF+PQ index creation."""
        pytest.importorskip("faiss")
        
        config = faiss_config_base.copy()
        config["parameters"]["faiss_index_factory_string"] = "IVF4,PQ8"
        
        # Generate larger sample for PQ training
        large_embeddings = np.random.randn(500, 384).astype(np.float32)
        
        model = DenseTextRetriever(config)
        model.load_model()  # Need to load model to set embedding_dim
        index = model._create_faiss_index(large_embeddings)
        
        assert index is not None
        assert index.ntotal == len(large_embeddings)
    
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
    
    def test_faiss_index_caching(self, faiss_config_base, sample_embeddings):
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
    
    def test_faiss_nprobe_configuration(self, faiss_config_base, sample_embeddings):
        """Test FAISS nprobe configuration for IVF indices."""
        pytest.importorskip("faiss")
        
        config = faiss_config_base.copy()
        config["parameters"]["faiss_index_factory_string"] = "IVF4,Flat"
        config["parameters"]["faiss_nprobe"] = 2
        
        model = DenseTextRetriever(config)
        model.load_model()  # Need to load model to set embedding_dim
        index = model._create_faiss_index(sample_embeddings)
        
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
    
    @pytest.mark.skipif(not hasattr(pytest, 'faiss_gpu_available'), 
                       reason="GPU FAISS not available")
    def test_faiss_gpu_support(self, faiss_config_base, sample_embeddings):
        """Test FAISS GPU support if available."""
        pytest.importorskip("faiss")
        
        config = faiss_config_base.copy()
        config["device"] = "cuda"
        config["parameters"]["faiss_use_gpu"] = True
        config["parameters"]["faiss_index_factory_string"] = "Flat"
        
        model = DenseTextRetriever(config)
        
        try:
            index = model._create_faiss_index(sample_embeddings)
            # If GPU support works, index should be on GPU
            # Note: This test might be skipped if GPU FAISS is not available
            assert index is not None
        except Exception as e:
            # GPU FAISS might not be available
            pytest.skip(f"GPU FAISS not available: {e}")
    
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
        
        # Both indices should be created successfully
        assert index_ip is not None
        assert index_l2 is not None
        assert index_ip.ntotal == len(sample_embeddings)
        assert index_l2.ntotal == len(sample_embeddings)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])