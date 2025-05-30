"""
Unit tests for HuggingFace Dataset Loader.

This module contains comprehensive tests for the HuggingFaceDatasetLoader,
including configuration validation, data loading, feature mapping, and error handling.
"""

import pytest
import tempfile
import json
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, Any
import sys
import os

# Add src to path for testing
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))


# Mock datasets library for testing
mock_dataset = MagicMock()
mock_dataset.__len__ = lambda: 3
mock_dataset.__iter__ = lambda: iter([
    {
        "id": "doc1",
        "text": "This is document 1",
        "title": "Document 1 Title",
        "question": "What is document 1?",
        "positive_ctxs": [{"passage_id": "doc1"}],
        "relevance": 2,
        "nested": {"content": {"main_text": "Nested content 1"}}
    },
    {
        "id": "doc2", 
        "text": "This is document 2",
        "title": "Document 2 Title",
        "question": "What is document 2?",
        "positive_ctxs": [{"passage_id": "doc2"}],
        "relevance": 1,
        "nested": {"content": {"main_text": "Nested content 2"}}
    },
    {
        "id": "doc3",
        "text": "This is document 3", 
        "title": "Document 3 Title",
        "question": "What is document 3?",
        "positive_ctxs": [{"passage_id": "doc3"}],
        "relevance": 3,
        "nested": {"content": {"main_text": "Nested content 3"}}
    }
])
mock_dataset.column_names = ["id", "text", "title", "question", "positive_ctxs", "relevance", "nested"]


class TestHuggingFaceDatasetConfig:
    """Test cases for HuggingFaceDatasetConfig class."""
    
    def test_config_initialization_basic(self, tmp_path):
        """Test basic HuggingFaceDatasetConfig initialization."""
        with patch.dict('sys.modules', {'datasets': MagicMock(), 'PIL': MagicMock()}):
            from newaibench.datasets.hf_loader import HuggingFaceDatasetConfig
            
            config = HuggingFaceDatasetConfig(
                dataset_path=tmp_path,
                hf_path="squad"
            )
            
            assert config.hf_path == "squad"
            assert config.hf_split == "train"
            assert config.hf_config_name is None
            assert config.hf_auto_detect_columns is True
            assert config.hf_default_relevance_score == 1
            assert config.hf_separate_splits is False
    
    def test_config_with_config_name_in_path(self, tmp_path):
        """Test parsing config name from path."""
        with patch.dict('sys.modules', {'datasets': MagicMock(), 'PIL': MagicMock()}):
            from newaibench.datasets.hf_loader import HuggingFaceDatasetConfig
            
            config = HuggingFaceDatasetConfig(
                dataset_path=tmp_path,
                hf_path="ms_marco,v1.1"
            )
            
            assert config.hf_path == "ms_marco"
            assert config.hf_config_name == "v1.1"
    
    def test_config_custom_feature_mappings(self, tmp_path):
        """Test custom feature mappings."""
        with patch.dict('sys.modules', {'datasets': MagicMock(), 'PIL': MagicMock()}):
            from newaibench.datasets.hf_loader import HuggingFaceDatasetConfig
            
            corpus_mapping = {
                "doc_id": "passage_id",
                "text": "passage_text",
                "title": "passage_title"
            }
            queries_mapping = {
                "query_id": "question_id",
                "text": "question_text"
            }
            qrels_mapping = {
                "query_id": "qid",
                "doc_id": "pid",
                "relevance": "label"
            }
            
            config = HuggingFaceDatasetConfig(
                dataset_path=tmp_path,
                hf_path="custom_dataset",
                corpus_feature_mapping=corpus_mapping,
                queries_feature_mapping=queries_mapping,
                qrels_feature_mapping=qrels_mapping
            )
            
            assert config.corpus_feature_mapping == corpus_mapping
            assert config.queries_feature_mapping == queries_mapping
            assert config.qrels_feature_mapping == qrels_mapping
    
    def test_config_separate_splits(self, tmp_path):
        """Test separate splits configuration."""
        with patch.dict('sys.modules', {'datasets': MagicMock(), 'PIL': MagicMock()}):
            from newaibench.datasets.hf_loader import HuggingFaceDatasetConfig
            
            config = HuggingFaceDatasetConfig(
                dataset_path=tmp_path,
                hf_path="dataset_with_splits",
                hf_separate_splits=True,
                hf_corpus_split="corpus",
                hf_queries_split="queries", 
                hf_qrels_split="qrels"
            )
            
            assert config.hf_separate_splits is True
            assert config.hf_corpus_split == "corpus"
            assert config.hf_queries_split == "queries"
            assert config.hf_qrels_split == "qrels"
    
    def test_config_missing_hf_path(self, tmp_path):
        """Test configuration with missing hf_path."""
        with patch.dict('sys.modules', {'datasets': MagicMock(), 'PIL': MagicMock()}):
            from newaibench.datasets.hf_loader import HuggingFaceDatasetConfig
            
            with pytest.raises(ValueError, match="hf_path must be provided"):
                HuggingFaceDatasetConfig(dataset_path=tmp_path)


class TestHuggingFaceDatasetLoader:
    """Test cases for HuggingFaceDatasetLoader class."""
    
    def test_loader_initialization(self, tmp_path):
        """Test HuggingFaceDatasetLoader initialization."""
        with patch.dict('sys.modules', {'datasets': MagicMock(), 'PIL': MagicMock()}):
            from newaibench.datasets.hf_loader import HuggingFaceDatasetLoader, HuggingFaceDatasetConfig
            
            config = HuggingFaceDatasetConfig(
                dataset_path=tmp_path,
                hf_path="squad"
            )
            
            with patch('newaibench.datasets.hf_loader.HF_DATASETS_AVAILABLE', True):
                loader = HuggingFaceDatasetLoader(config)
                assert loader.config == config
                assert loader._hf_dataset is None
                assert loader._hf_corpus_dataset is None
                assert loader._hf_queries_dataset is None
                assert loader._hf_qrels_dataset is None
    
    def test_loader_invalid_config(self, tmp_path):
        """Test loader with invalid config type."""
        with patch.dict('sys.modules', {'datasets': MagicMock(), 'PIL': MagicMock()}):
            from newaibench.datasets.hf_loader import HuggingFaceDatasetLoader
            from newaibench.datasets.base import DatasetConfig
            
            config = DatasetConfig(dataset_path=tmp_path)
            
            with patch('newaibench.datasets.hf_loader.HF_DATASETS_AVAILABLE', True):
                with pytest.raises(ValueError, match="config must be an instance of HuggingFaceDatasetConfig"):
                    HuggingFaceDatasetLoader(config)
    
    def test_loader_datasets_not_available(self, tmp_path):
        """Test loader when datasets library is not available."""
        with patch.dict('sys.modules', {'datasets': MagicMock(), 'PIL': MagicMock()}):
            from newaibench.datasets.hf_loader import HuggingFaceDatasetLoader, HuggingFaceDatasetConfig
            
            config = HuggingFaceDatasetConfig(
                dataset_path=tmp_path,
                hf_path="squad"
            )
            
            with patch('newaibench.datasets.hf_loader.HF_DATASETS_AVAILABLE', False):
                with pytest.raises(ImportError, match="Hugging Face datasets library not available"):
                    HuggingFaceDatasetLoader(config)
    
    @patch('newaibench.datasets.hf_loader.load_dataset')
    def test_load_hf_dataset_basic(self, mock_load_dataset, tmp_path):
        """Test basic HF dataset loading."""
        with patch.dict('sys.modules', {'datasets': MagicMock(), 'PIL': MagicMock()}):
            from newaibench.datasets.hf_loader import HuggingFaceDatasetLoader, HuggingFaceDatasetConfig
            
            mock_load_dataset.return_value = mock_dataset
            
            config = HuggingFaceDatasetConfig(
                dataset_path=tmp_path,
                hf_path="squad"
            )
            
            with patch('newaibench.datasets.hf_loader.HF_DATASETS_AVAILABLE', True):
                loader = HuggingFaceDatasetLoader(config)
                dataset = loader._load_hf_dataset()
                
                mock_load_dataset.assert_called_once_with(
                    "squad",
                    split="train",
                    streaming=False,
                    trust_remote_code=False
                )
                assert dataset == mock_dataset
    
    @patch('newaibench.datasets.hf_loader.load_dataset')
    def test_load_hf_dataset_with_config_name(self, mock_load_dataset, tmp_path):
        """Test HF dataset loading with config name."""
        with patch.dict('sys.modules', {'datasets': MagicMock(), 'PIL': MagicMock()}):
            from newaibench.datasets.hf_loader import HuggingFaceDatasetLoader, HuggingFaceDatasetConfig
            
            mock_load_dataset.return_value = mock_dataset
            
            config = HuggingFaceDatasetConfig(
                dataset_path=tmp_path,
                hf_path="ms_marco",
                hf_config_name="v1.1"
            )
            
            with patch('newaibench.datasets.hf_loader.HF_DATASETS_AVAILABLE', True):
                loader = HuggingFaceDatasetLoader(config)
                dataset = loader._load_hf_dataset()
                
                mock_load_dataset.assert_called_once_with(
                    "ms_marco",
                    split="train",
                    streaming=False,
                    trust_remote_code=False,
                    name="v1.1"
                )
    
    @patch('newaibench.datasets.hf_loader.load_dataset')
    def test_load_hf_dataset_separate_splits(self, mock_load_dataset, tmp_path):
        """Test HF dataset loading with separate splits."""
        with patch.dict('sys.modules', {'datasets': MagicMock(), 'PIL': MagicMock()}):
            from newaibench.datasets.hf_loader import HuggingFaceDatasetLoader, HuggingFaceDatasetConfig
            
            mock_load_dataset.return_value = mock_dataset
            
            config = HuggingFaceDatasetConfig(
                dataset_path=tmp_path,
                hf_path="dataset_with_splits",
                hf_separate_splits=True,
                hf_corpus_split="corpus",
                hf_queries_split="queries"
            )
            
            with patch('newaibench.datasets.hf_loader.HF_DATASETS_AVAILABLE', True):
                loader = HuggingFaceDatasetLoader(config)
                
                # Test corpus split
                dataset = loader._load_hf_dataset('corpus')
                mock_load_dataset.assert_called_with(
                    "dataset_with_splits",
                    split="corpus",
                    streaming=False,
                    trust_remote_code=False
                )
    
    def test_get_nested_field(self, tmp_path):
        """Test nested field access."""
        with patch.dict('sys.modules', {'datasets': MagicMock(), 'PIL': MagicMock()}):
            from newaibench.datasets.hf_loader import HuggingFaceDatasetLoader, HuggingFaceDatasetConfig
            
            config = HuggingFaceDatasetConfig(
                dataset_path=tmp_path,
                hf_path="test_dataset"
            )
            
            with patch('newaibench.datasets.hf_loader.HF_DATASETS_AVAILABLE', True):
                loader = HuggingFaceDatasetLoader(config)
                
                item = {
                    "simple": "value",
                    "nested": {
                        "content": {
                            "text": "nested text"
                        }
                    },
                    "list": ["item1", "item2"]
                }
                
                # Test simple field
                assert loader._get_nested_field(item, "simple") == "value"
                
                # Test nested field
                assert loader._get_nested_field(item, "nested.content.text") == "nested text"
                
                # Test list access
                assert loader._get_nested_field(item, "list.0") == "item1"
                
                # Test non-existent field
                assert loader._get_nested_field(item, "non.existent") is None
    
    @patch('newaibench.datasets.hf_loader.load_dataset')
    def test_load_corpus_basic(self, mock_load_dataset, tmp_path):
        """Test basic corpus loading."""
        with patch.dict('sys.modules', {'datasets': MagicMock(), 'PIL': MagicMock()}):
            from newaibench.datasets.hf_loader import HuggingFaceDatasetLoader, HuggingFaceDatasetConfig
            
            mock_load_dataset.return_value = mock_dataset
            
            config = HuggingFaceDatasetConfig(
                dataset_path=tmp_path,
                hf_path="test_dataset"
            )
            
            with patch('newaibench.datasets.hf_loader.HF_DATASETS_AVAILABLE', True):
                loader = HuggingFaceDatasetLoader(config)
                corpus = loader.load_corpus()
                
                assert len(corpus) == 3
                assert "doc1" in corpus
                assert corpus["doc1"]["text"] == "This is document 1"
                assert corpus["doc1"]["title"] == "Document 1 Title"
                assert corpus["doc1"]["doc_id"] == "doc1"
                assert "ocr_text" in corpus["doc1"]
                assert corpus["doc1"]["ocr_text"] is None
    
    @patch('newaibench.datasets.hf_loader.load_dataset')
    def test_load_corpus_with_custom_mapping(self, mock_load_dataset, tmp_path):
        """Test corpus loading with custom feature mapping."""
        with patch.dict('sys.modules', {'datasets': MagicMock(), 'PIL': MagicMock()}):
            from newaibench.datasets.hf_loader import HuggingFaceDatasetLoader, HuggingFaceDatasetConfig
            
            # Create mock dataset with different column names
            custom_mock_dataset = MagicMock()
            custom_mock_dataset.__len__ = lambda: 2
            custom_mock_dataset.__iter__ = lambda: iter([
                {
                    "passage_id": "p1",
                    "passage_text": "Custom passage 1",
                    "passage_title": "Custom Title 1"
                },
                {
                    "passage_id": "p2",
                    "passage_text": "Custom passage 2",
                    "passage_title": "Custom Title 2"
                }
            ])
            custom_mock_dataset.column_names = ["passage_id", "passage_text", "passage_title"]
            
            mock_load_dataset.return_value = custom_mock_dataset
            
            config = HuggingFaceDatasetConfig(
                dataset_path=tmp_path,
                hf_path="custom_dataset",
                corpus_feature_mapping={
                    "doc_id": "passage_id",
                    "text": "passage_text",
                    "title": "passage_title"
                }
            )
            
            with patch('newaibench.datasets.hf_loader.HF_DATASETS_AVAILABLE', True):
                loader = HuggingFaceDatasetLoader(config)
                corpus = loader.load_corpus()
                
                assert len(corpus) == 2
                assert "p1" in corpus
                assert corpus["p1"]["text"] == "Custom passage 1"
                assert corpus["p1"]["title"] == "Custom Title 1"
    
    @patch('newaibench.datasets.hf_loader.load_dataset')
    def test_load_queries_basic(self, mock_load_dataset, tmp_path):
        """Test basic queries loading."""
        with patch.dict('sys.modules', {'datasets': MagicMock(), 'PIL': MagicMock()}):
            from newaibench.datasets.hf_loader import HuggingFaceDatasetLoader, HuggingFaceDatasetConfig
            
            mock_load_dataset.return_value = mock_dataset
            
            config = HuggingFaceDatasetConfig(
                dataset_path=tmp_path,
                hf_path="test_dataset"
            )
            
            with patch('newaibench.datasets.hf_loader.HF_DATASETS_AVAILABLE', True):
                loader = HuggingFaceDatasetLoader(config)
                queries = loader.load_queries()
                
                assert len(queries) == 3
                assert "doc1" in queries
                assert queries["doc1"] == "What is document 1?"
    
    @patch('newaibench.datasets.hf_loader.load_dataset')
    def test_load_qrels_basic(self, mock_load_dataset, tmp_path):
        """Test basic qrels loading."""
        with patch.dict('sys.modules', {'datasets': MagicMock(), 'PIL': MagicMock()}):
            from newaibench.datasets.hf_loader import HuggingFaceDatasetLoader, HuggingFaceDatasetConfig
            
            mock_load_dataset.return_value = mock_dataset
            
            config = HuggingFaceDatasetConfig(
                dataset_path=tmp_path,
                hf_path="test_dataset"
            )
            
            with patch('newaibench.datasets.hf_loader.HF_DATASETS_AVAILABLE', True):
                loader = HuggingFaceDatasetLoader(config)
                qrels = loader.load_qrels()
                
                assert len(qrels) >= 1
                # Check that positive_ctxs are processed correctly
                for query_id, judgments in qrels.items():
                    assert isinstance(judgments, dict)
                    for doc_id, relevance in judgments.items():
                        assert isinstance(relevance, int)
    
    @patch('newaibench.datasets.hf_loader.load_dataset')
    def test_load_corpus_with_nested_mapping(self, mock_load_dataset, tmp_path):
        """Test corpus loading with nested field mapping."""
        with patch.dict('sys.modules', {'datasets': MagicMock(), 'PIL': MagicMock()}):
            from newaibench.datasets.hf_loader import HuggingFaceDatasetLoader, HuggingFaceDatasetConfig
            
            mock_load_dataset.return_value = mock_dataset
            
            config = HuggingFaceDatasetConfig(
                dataset_path=tmp_path,
                hf_path="test_dataset",
                corpus_feature_mapping={
                    "doc_id": "id",
                    "text": "nested.content.main_text",
                    "title": "title"
                }
            )
            
            with patch('newaibench.datasets.hf_loader.HF_DATASETS_AVAILABLE', True):
                loader = HuggingFaceDatasetLoader(config)
                corpus = loader.load_corpus()
                
                assert len(corpus) == 3
                assert "doc1" in corpus
                assert corpus["doc1"]["text"] == "Nested content 1"
    
    @patch('newaibench.datasets.hf_loader.load_dataset')
    def test_load_with_max_samples(self, mock_load_dataset, tmp_path):
        """Test loading with max_samples limit."""
        with patch.dict('sys.modules', {'datasets': MagicMock(), 'PIL': MagicMock()}):
            from newaibench.datasets.hf_loader import HuggingFaceDatasetLoader, HuggingFaceDatasetConfig
            
            mock_load_dataset.return_value = mock_dataset
            
            config = HuggingFaceDatasetConfig(
                dataset_path=tmp_path,
                hf_path="test_dataset",
                max_samples=2
            )
            
            with patch('newaibench.datasets.hf_loader.HF_DATASETS_AVAILABLE', True):
                loader = HuggingFaceDatasetLoader(config)
                corpus = loader.load_corpus()
                queries = loader.load_queries()
                
                assert len(corpus) == 2
                assert len(queries) == 2
    
    @patch('newaibench.datasets.hf_loader.load_dataset')
    def test_auto_detect_columns(self, mock_load_dataset, tmp_path):
        """Test automatic column detection."""
        with patch.dict('sys.modules', {'datasets': MagicMock(), 'PIL': MagicMock()}):
            from newaibench.datasets.hf_loader import HuggingFaceDatasetLoader, HuggingFaceDatasetConfig
            
            # Mock dataset with different but detectable column names
            auto_detect_mock = MagicMock()
            auto_detect_mock.__len__ = lambda: 1
            auto_detect_mock.__iter__ = lambda: iter([
                {
                    "document_id": "auto1",
                    "content": "Auto detected content",
                    "question": "Auto detected question"
                }
            ])
            auto_detect_mock.column_names = ["document_id", "content", "question"]
            
            mock_load_dataset.return_value = auto_detect_mock
            
            config = HuggingFaceDatasetConfig(
                dataset_path=tmp_path,
                hf_path="auto_detect_dataset",
                hf_auto_detect_columns=True,
                corpus_feature_mapping={
                    "doc_id": "nonexistent_id",  # This should be auto-detected
                    "text": "nonexistent_text"  # This should be auto-detected
                }
            )
            
            with patch('newaibench.datasets.hf_loader.HF_DATASETS_AVAILABLE', True):
                loader = HuggingFaceDatasetLoader(config)
                corpus = loader.load_corpus()
                
                assert len(corpus) == 1
                # Should auto-detect document_id for doc_id and content for text
                assert "auto1" in corpus or "doc_0" in corpus  # Either auto-detected or generated
    
    def test_dataset_loading_error(self, tmp_path):
        """Test error handling during dataset loading."""
        with patch.dict('sys.modules', {'datasets': MagicMock(), 'PIL': MagicMock()}):
            from newaibench.datasets.hf_loader import HuggingFaceDatasetLoader, HuggingFaceDatasetConfig
            from newaibench.datasets.base import DatasetLoadingError
            
            config = HuggingFaceDatasetConfig(
                dataset_path=tmp_path,
                hf_path="nonexistent_dataset"
            )
            
            with patch('newaibench.datasets.hf_loader.HF_DATASETS_AVAILABLE', True):
                loader = HuggingFaceDatasetLoader(config)
                
                with patch('newaibench.datasets.hf_loader.load_dataset', side_effect=Exception("Dataset not found")):
                    with pytest.raises(DatasetLoadingError, match="Failed to load HF dataset"):
                        loader._load_hf_dataset()


if __name__ == "__main__":
    pytest.main([__file__])
