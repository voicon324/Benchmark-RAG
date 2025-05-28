"""
Unit tests for NewAIBench dataset loaders.

This module contains comprehensive tests for the base dataset loader functionality,
including configuration validation, error handling, and data processing.
"""

import pytest
import tempfile
import json
import csv
from pathlib import Path
from unittest.mock import Mock, patch
import sys
import os

# Add src to path for testing
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from newaibench.datasets.base import (
    BaseDatasetLoader,
    DatasetConfig,
    DatasetLoadingError,
    DataValidationError
)


class TestDatasetConfig:
    """Test cases for DatasetConfig class."""
    
    def test_dataset_config_initialization(self, tmp_path):
        """Test basic DatasetConfig initialization."""
        config = DatasetConfig(
            dataset_path=tmp_path,
            corpus_file="test_corpus.jsonl",
            queries_file="test_queries.jsonl"
        )
        
        assert config.dataset_path == tmp_path
        assert config.corpus_file == "test_corpus.jsonl"
        assert config.queries_file == "test_queries.jsonl"
        assert config.format_type == "jsonl"
        assert config.encoding == "utf-8"
        assert config.validation_enabled is True
        assert config.cache_enabled is True
    
    def test_dataset_config_nonexistent_path(self):
        """Test DatasetConfig with non-existent path."""
        with pytest.raises(FileNotFoundError):
            DatasetConfig(dataset_path="/nonexistent/path")
    
    def test_dataset_config_invalid_format(self, tmp_path):
        """Test DatasetConfig with invalid format type."""
        with pytest.raises(ValueError):
            DatasetConfig(
                dataset_path=tmp_path,
                format_type="invalid_format"
            )
    
    def test_dataset_config_with_preprocessing_options(self, tmp_path):
        """Test DatasetConfig with preprocessing options."""
        preprocessing_options = {
            "lowercase": True,
            "normalize_whitespace": True,
            "min_length": 5
        }
        
        config = DatasetConfig(
            dataset_path=tmp_path,
            preprocessing_options=preprocessing_options
        )
        
        assert config.preprocessing_options == preprocessing_options


class MockDatasetLoader(BaseDatasetLoader):
    """Mock implementation of BaseDatasetLoader for testing."""
    
    def __init__(self, config: DatasetConfig):
        super().__init__(config)
        self._mock_corpus = {
            "doc1": {"text": "This is document 1"},
            "doc2": {"text": "This is document 2"}
        }
        self._mock_queries = {
            "q1": "Query 1 text",
            "q2": "Query 2 text"
        }
        self._mock_qrels = {
            "q1": {"doc1": 1, "doc2": 0},
            "q2": {"doc1": 0, "doc2": 1}
        }
    
    def load_corpus(self):
        if self.config.cache_enabled and self._corpus is not None:
            return self._corpus
        
        self._corpus = self._mock_corpus
        return self._corpus
    
    def load_queries(self):
        if self.config.cache_enabled and self._queries is not None:
            return self._queries
        
        self._queries = self._mock_queries
        return self._queries
    
    def load_qrels(self):
        if self.config.cache_enabled and self._qrels is not None:
            return self._qrels
        
        self._qrels = self._mock_qrels
        return self._qrels


class TestBaseDatasetLoader:
    """Test cases for BaseDatasetLoader base class."""
    
    def test_base_loader_initialization(self, tmp_path):
        """Test BaseDatasetLoader initialization."""
        config = DatasetConfig(dataset_path=tmp_path)
        loader = MockDatasetLoader(config)
        
        assert loader.config == config
        assert loader._cache == {}
        assert loader._corpus is None
        assert loader._queries is None
        assert loader._qrels is None
    
    def test_base_loader_invalid_config(self, tmp_path):
        """Test BaseDatasetLoader with invalid config."""
        with pytest.raises(ValueError):
            MockDatasetLoader("not_a_config")
    
    def test_load_all_components(self, tmp_path):
        """Test loading all dataset components."""
        config = DatasetConfig(dataset_path=tmp_path)
        loader = MockDatasetLoader(config)
        
        corpus, queries, qrels = loader.load_all()
        
        assert len(corpus) == 2
        assert len(queries) == 2
        assert len(qrels) == 2
        assert "doc1" in corpus
        assert "q1" in queries
        assert "q1" in qrels
    
    def test_caching_functionality(self, tmp_path):
        """Test caching of loaded data."""
        config = DatasetConfig(dataset_path=tmp_path, cache_enabled=True)
        loader = MockDatasetLoader(config)
        
        # First load
        corpus1 = loader.load_corpus()
        queries1 = loader.load_queries()
        qrels1 = loader.load_qrels()
        
        # Second load (should use cache)
        corpus2 = loader.load_corpus()
        queries2 = loader.load_queries()
        qrels2 = loader.load_qrels()
        
        # Should be the same objects (cached)
        assert corpus1 is corpus2
        assert queries1 is queries2
        assert qrels1 is qrels2
    
    def test_clear_cache(self, tmp_path):
        """Test cache clearing functionality."""
        config = DatasetConfig(dataset_path=tmp_path, cache_enabled=True)
        loader = MockDatasetLoader(config)
        
        # Load data to populate cache
        loader.load_corpus()
        loader.load_queries()
        loader.load_qrels()
        
        assert loader._corpus is not None
        assert loader._queries is not None
        assert loader._qrels is not None
        
        # Clear cache
        loader.clear_cache()
        
        assert loader._corpus is None
        assert loader._queries is None
        assert loader._qrels is None
        assert loader._cache == {}
    
    def test_get_statistics(self, tmp_path):
        """Test statistics generation."""
        config = DatasetConfig(dataset_path=tmp_path)
        loader = MockDatasetLoader(config)
        
        # Load data first
        loader.load_all()
        
        stats = loader.get_statistics()
        
        assert "dataset_path" in stats
        assert "format_type" in stats
        assert "encoding" in stats
        assert "corpus_size" in stats
        assert "queries_count" in stats
        assert "qrels_count" in stats
        assert stats["corpus_size"] == 2
        assert stats["queries_count"] == 2
        assert stats["qrels_count"] == 2
    
    def test_data_validation_success(self, tmp_path):
        """Test successful data validation."""
        config = DatasetConfig(dataset_path=tmp_path)
        loader = MockDatasetLoader(config)
        
        corpus = {"doc1": {"text": "Document 1"}}
        queries = {"q1": "Query 1"}
        qrels = {"q1": {"doc1": 1}}
        
        result = loader.validate_data(corpus, queries, qrels)
        assert result is True
    
    def test_data_validation_empty_corpus(self, tmp_path):
        """Test data validation with empty corpus."""
        config = DatasetConfig(dataset_path=tmp_path)
        loader = MockDatasetLoader(config)
        
        corpus = {}
        queries = {"q1": "Query 1"}
        qrels = {"q1": {"doc1": 1}}
        
        with pytest.raises(DataValidationError):
            loader.validate_data(corpus, queries, qrels)
    
    def test_data_validation_missing_text_field(self, tmp_path):
        """Test data validation with missing text field."""
        config = DatasetConfig(dataset_path=tmp_path)
        loader = MockDatasetLoader(config)
        
        corpus = {"doc1": {"title": "Document 1"}}  # Missing 'text' field
        queries = {"q1": "Query 1"}
        qrels = {"q1": {"doc1": 1}}
        
        with pytest.raises(DataValidationError):
            loader.validate_data(corpus, queries, qrels)
    
    def test_data_validation_disabled(self, tmp_path):
        """Test data validation when disabled."""
        config = DatasetConfig(dataset_path=tmp_path, validation_enabled=False)
        loader = MockDatasetLoader(config)
        
        # Invalid data that would normally fail validation
        corpus = {}
        queries = {}
        qrels = {}
        
        result = loader.validate_data(corpus, queries, qrels)
        assert result is True
    
    def test_preprocessing_lowercase(self, tmp_path):
        """Test text preprocessing with lowercase option."""
        config = DatasetConfig(
            dataset_path=tmp_path,
            preprocessing_options={"lowercase": True}
        )
        loader = MockDatasetLoader(config)
        
        text = "This IS a TEST Text"
        processed = loader._apply_preprocessing(text)
        
        assert processed == "this is a test text"
    
    def test_preprocessing_remove_special_chars(self, tmp_path):
        """Test text preprocessing with special character removal."""
        config = DatasetConfig(
            dataset_path=tmp_path,
            preprocessing_options={"remove_special_chars": True}
        )
        loader = MockDatasetLoader(config)
        
        text = "Hello, world! How are you?"
        processed = loader._apply_preprocessing(text)
        
        assert "," not in processed
        assert "!" not in processed
        assert "?" not in processed
    
    def test_preprocessing_normalize_whitespace(self, tmp_path):
        """Test text preprocessing with whitespace normalization."""
        config = DatasetConfig(
            dataset_path=tmp_path,
            preprocessing_options={"normalize_whitespace": True}
        )
        loader = MockDatasetLoader(config)
        
        text = "  This   has    extra     spaces  "
        processed = loader._apply_preprocessing(text)
        
        assert processed == "This has extra spaces"
    
    def test_read_file_safely_success(self, tmp_path):
        """Test safe file reading with success."""
        config = DatasetConfig(dataset_path=tmp_path)
        loader = MockDatasetLoader(config)
        
        # Create a test file
        test_file = tmp_path / "test.txt"
        test_content = "This is test content"
        test_file.write_text(test_content, encoding="utf-8")
        
        content = loader._read_file_safely(test_file)
        assert content == test_content
    
    def test_read_file_safely_encoding_error(self, tmp_path):
        """Test safe file reading with encoding error."""
        config = DatasetConfig(dataset_path=tmp_path, encoding="ascii")
        loader = MockDatasetLoader(config)
        
        # Create a file with non-ASCII content
        test_file = tmp_path / "test.txt"
        test_file.write_bytes("This contains unicode: ñüñé".encode("utf-8"))
        
        with pytest.raises(DatasetLoadingError):
            loader._read_file_safely(test_file)
    
    def test_read_file_safely_nonexistent_file(self, tmp_path):
        """Test safe file reading with non-existent file."""
        config = DatasetConfig(dataset_path=tmp_path)
        loader = MockDatasetLoader(config)
        
        nonexistent_file = tmp_path / "nonexistent.txt"
        
        with pytest.raises(DatasetLoadingError):
            loader._read_file_safely(nonexistent_file)
    
    def test_repr_method(self, tmp_path):
        """Test string representation of loader."""
        config = DatasetConfig(dataset_path=tmp_path)
        loader = MockDatasetLoader(config)
        
        repr_str = repr(loader)
        assert "MockDatasetLoader" in repr_str
        assert str(tmp_path) in repr_str


class TestDatasetLoadingError:
    """Test cases for DatasetLoadingError exception."""
    
    def test_dataset_loading_error_creation(self):
        """Test DatasetLoadingError creation."""
        error = DatasetLoadingError("Test error message")
        assert str(error) == "Test error message"
        assert isinstance(error, Exception)


class TestDataValidationError:
    """Test cases for DataValidationError exception."""
    
    def test_data_validation_error_creation(self):
        """Test DataValidationError creation."""
        error = DataValidationError("Validation failed")
        assert str(error) == "Validation failed"
        assert isinstance(error, Exception)


# Fixtures for testing
@pytest.fixture
def sample_corpus_data():
    """Sample corpus data for testing."""
    return {
        "doc1": {
            "text": "This is the first document about machine learning",
            "title": "ML Document 1"
        },
        "doc2": {
            "text": "This is the second document about information retrieval",
            "title": "IR Document 2"
        },
        "doc3": {
            "text": "This is the third document about natural language processing",
            "title": "NLP Document 3"
        }
    }


@pytest.fixture
def sample_queries_data():
    """Sample queries data for testing."""
    return {
        "q1": "machine learning algorithms",
        "q2": "information retrieval systems",
        "q3": "natural language processing techniques"
    }


@pytest.fixture
def sample_qrels_data():
    """Sample qrels data for testing."""
    return {
        "q1": {"doc1": 2, "doc2": 0, "doc3": 1},
        "q2": {"doc1": 0, "doc2": 2, "doc3": 1},
        "q3": {"doc1": 1, "doc2": 1, "doc3": 2}
    }


class TestDatasetLoaderIntegration:
    """Integration tests for dataset loader functionality."""
    
    def test_full_workflow_with_mock_data(self, tmp_path, sample_corpus_data, 
                                        sample_queries_data, sample_qrels_data):
        """Test complete workflow with sample data."""
        config = DatasetConfig(dataset_path=tmp_path)
        
        # Create a custom mock loader with sample data
        class SampleDataLoader(BaseDatasetLoader):
            def load_corpus(self):
                if self.config.cache_enabled and self._corpus is not None:
                    return self._corpus
                self._corpus = sample_corpus_data
                return self._corpus
            
            def load_queries(self):
                if self.config.cache_enabled and self._queries is not None:
                    return self._queries
                self._queries = sample_queries_data
                return self._queries
            
            def load_qrels(self):
                if self.config.cache_enabled and self._qrels is not None:
                    return self._qrels
                self._qrels = sample_qrels_data
                return self._qrels
        
        loader = SampleDataLoader(config)
        
        # Test loading all components
        corpus, queries, qrels = loader.load_all()
        
        # Validate loaded data
        assert len(corpus) == 3
        assert len(queries) == 3
        assert len(qrels) == 3
        
        # Test data validation
        assert loader.validate_data(corpus, queries, qrels) is True
        
        # Test statistics
        stats = loader.get_statistics()
        assert stats["corpus_size"] == 3
        assert stats["queries_count"] == 3
        assert stats["qrels_count"] == 3
        assert stats["avg_doc_length"] > 0
        assert stats["avg_query_length"] > 0
        assert stats["total_judgments"] == 9
    
    def test_error_handling_workflow(self, tmp_path):
        """Test error handling in dataset loading workflow."""
        config = DatasetConfig(dataset_path=tmp_path)
        
        # Create a loader that raises errors
        class ErrorDataLoader(BaseDatasetLoader):
            def load_corpus(self):
                raise DatasetLoadingError("Failed to load corpus")
            
            def load_queries(self):
                raise DatasetLoadingError("Failed to load queries")
            
            def load_qrels(self):
                raise DatasetLoadingError("Failed to load qrels")
        
        loader = ErrorDataLoader(config)
        
        # Test that load_all properly handles and propagates errors
        with pytest.raises(DatasetLoadingError):
            loader.load_all()
    
    def test_performance_with_large_dataset_simulation(self, tmp_path):
        """Test performance characteristics with simulated large dataset."""
        config = DatasetConfig(dataset_path=tmp_path, max_samples=1000)
        
        # Create a loader that simulates large dataset
        class LargeDataLoader(BaseDatasetLoader):
            def load_corpus(self):
                if self.config.cache_enabled and self._corpus is not None:
                    return self._corpus
                
                # Simulate large corpus
                corpus = {}
                limit = self.config.max_samples or 10000
                for i in range(min(limit, 10000)):
                    corpus[f"doc{i}"] = {
                        "text": f"This is document {i} with some sample text content."
                    }
                
                self._corpus = corpus
                return self._corpus
            
            def load_queries(self):
                if self.config.cache_enabled and self._queries is not None:
                    return self._queries
                
                # Simulate queries
                queries = {}
                for i in range(100):
                    queries[f"q{i}"] = f"Query {i} text"
                
                self._queries = queries
                return self._queries
            
            def load_qrels(self):
                if self.config.cache_enabled and self._qrels is not None:
                    return self._qrels
                
                # Simulate qrels
                qrels = {}
                for i in range(100):
                    qrels[f"q{i}"] = {f"doc{j}": 1 if j % 10 == 0 else 0 
                                     for j in range(min(50, self.config.max_samples or 1000))}
                
                self._qrels = qrels
                return self._qrels
        
        loader = LargeDataLoader(config)
        
        import time
        start_time = time.time()
        
        corpus, queries, qrels = loader.load_all()
        
        load_time = time.time() - start_time
        
        # Verify data was loaded correctly
        assert len(corpus) <= 1000  # Respects max_samples
        assert len(queries) == 100
        assert len(qrels) == 100
        
        # Basic performance check (should complete in reasonable time)
        assert load_time < 10.0  # Should complete within 10 seconds
        
        # Test statistics generation performance
        start_time = time.time()
        stats = loader.get_statistics()
        stats_time = time.time() - start_time
        
        assert stats_time < 5.0  # Statistics should be fast
        assert "corpus_size" in stats
        assert "relevance_distribution" in stats


if __name__ == "__main__":
    pytest.main([__file__])