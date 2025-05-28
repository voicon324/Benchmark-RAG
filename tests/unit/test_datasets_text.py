"""
Unit tests for datasets.text module.
"""
import pytest
import tempfile
import json
import csv
from pathlib import Path
from unittest.mock import patch, mock_open, MagicMock

from newaibench.datasets.base import DatasetConfig
from newaibench.datasets.text import TextDatasetLoader


class TestTextDatasetLoader:
    """Test cases for TextDatasetLoader class."""
    
    @pytest.mark.unit
    def test_initialization(self):
        """Test TextDatasetLoader initialization."""
        config = DatasetConfig(dataset_path="/test/path")
        loader = TextDatasetLoader(config)
        
        assert loader.config == config
        assert loader.config.dataset_path == "/test/path"
    
    @pytest.mark.unit
    def test_initialization_with_preprocessing_options(self):
        """Test initialization with custom preprocessing options."""
        preprocessing_options = {
            "lowercase": True,
            "remove_punctuation": False,
            "normalize_unicode": True
        }
        
        config = DatasetConfig(
            dataset_path="/test/path",
            preprocessing_options=preprocessing_options
        )
        loader = TextDatasetLoader(config)
        
        assert loader.config.preprocessing_options["lowercase"] is True
        assert loader.config.preprocessing_options["remove_punctuation"] is False
        assert loader.config.preprocessing_options["normalize_unicode"] is True
    
    @pytest.mark.unit
    @patch('builtins.open', new_callable=mock_open)
    @patch('os.path.exists', return_value=True)
    def test_load_corpus_jsonl(self, mock_exists, mock_file):
        """Test loading corpus from JSONL format."""
        # Mock JSONL content
        jsonl_content = [
            '{"_id": "doc1", "title": "Test Document 1", "text": "This is test content 1"}',
            '{"_id": "doc2", "title": "Test Document 2", "text": "This is test content 2"}'
        ]
        mock_file.return_value.__iter__ = lambda self: iter(jsonl_content)
        
        config = DatasetConfig(
            dataset_path="/test/path",
            corpus_file="corpus.jsonl",
            format_type="jsonl"
        )
        loader = TextDatasetLoader(config)
        
        corpus = loader.load_corpus()
        
        assert len(corpus) == 2
        assert "doc1" in corpus
        assert "doc2" in corpus
        assert corpus["doc1"]["title"] == "Test Document 1"
        assert corpus["doc1"]["text"] == "This is test content 1"
        assert corpus["doc2"]["title"] == "Test Document 2"
        assert corpus["doc2"]["text"] == "This is test content 2"
    
    @pytest.mark.unit
    @patch('builtins.open', new_callable=mock_open)
    @patch('os.path.exists', return_value=True)
    def test_load_queries_jsonl(self, mock_exists, mock_file):
        """Test loading queries from JSONL format."""
        # Mock JSONL content
        jsonl_content = [
            '{"_id": "q1", "text": "What is machine learning?"}',
            '{"_id": "q2", "text": "How does neural network work?"}'
        ]
        mock_file.return_value.__iter__ = lambda self: iter(jsonl_content)
        
        config = DatasetConfig(
            dataset_path="/test/path",
            queries_file="queries.jsonl",
            format_type="jsonl"
        )
        loader = TextDatasetLoader(config)
        
        queries = loader.load_queries()
        
        assert len(queries) == 2
        assert "q1" in queries
        assert "q2" in queries
        assert queries["q1"]["text"] == "What is machine learning?"
        assert queries["q2"]["text"] == "How does neural network work?"
    
    @pytest.mark.unit
    @patch('builtins.open', new_callable=mock_open)
    @patch('os.path.exists', return_value=True)
    def test_load_qrels_tsv(self, mock_exists, mock_file):
        """Test loading qrels from TSV format."""
        # Mock TSV content
        tsv_content = "q1\t0\tdoc1\t1\nq1\t0\tdoc2\t0\nq2\t0\tdoc1\t0\nq2\t0\tdoc2\t1"
        mock_file.return_value.read.return_value = tsv_content
        
        config = DatasetConfig(
            dataset_path="/test/path",
            qrels_file="qrels.tsv"
        )
        loader = TextDatasetLoader(config)
        
        qrels = loader.load_qrels()
        
        assert "q1" in qrels
        assert "q2" in qrels
        assert qrels["q1"]["doc1"] == 1
        assert qrels["q1"]["doc2"] == 0
        assert qrels["q2"]["doc1"] == 0
        assert qrels["q2"]["doc2"] == 1
    
    @pytest.mark.unit
    @patch('pandas.read_csv')
    @patch('os.path.exists', return_value=True)
    def test_load_corpus_csv(self, mock_exists, mock_read_csv):
        """Test loading corpus from CSV format."""
        # Mock pandas DataFrame
        import pandas as pd
        mock_df = pd.DataFrame({
            'id': ['doc1', 'doc2'],
            'title': ['Test Document 1', 'Test Document 2'],
            'text': ['This is test content 1', 'This is test content 2']
        })
        mock_read_csv.return_value = mock_df
        
        config = DatasetConfig(
            dataset_path="/test/path",
            corpus_file="corpus.csv",
            format_type="csv"
        )
        loader = TextDatasetLoader(config)
        
        corpus = loader.load_corpus()
        
        assert len(corpus) == 2
        assert "doc1" in corpus
        assert "doc2" in corpus
        assert corpus["doc1"]["title"] == "Test Document 1"
        assert corpus["doc2"]["text"] == "This is test content 2"
    
    @pytest.mark.unit
    def test_preprocess_text_lowercase(self):
        """Test text preprocessing with lowercase option."""
        config = DatasetConfig(
            dataset_path="/test/path",
            preprocessing_options={"lowercase": True}
        )
        loader = TextDatasetLoader(config)
        
        text = "This Is A Test Text With UPPERCASE"
        processed = loader._preprocess_text(text)
        
        assert processed == "this is a test text with uppercase"
    
    @pytest.mark.unit
    def test_preprocess_text_remove_punctuation(self):
        """Test text preprocessing with punctuation removal."""
        config = DatasetConfig(
            dataset_path="/test/path",
            preprocessing_options={"remove_punctuation": True}
        )
        loader = TextDatasetLoader(config)
        
        text = "Hello, world! How are you?"
        processed = loader._preprocess_text(text)
        
        assert processed == "Hello world How are you"
    
    @pytest.mark.unit
    def test_preprocess_text_multiple_options(self):
        """Test text preprocessing with multiple options."""
        config = DatasetConfig(
            dataset_path="/test/path",
            preprocessing_options={
                "lowercase": True,
                "remove_punctuation": True,
                "normalize_whitespace": True
            }
        )
        loader = TextDatasetLoader(config)
        
        text = "Hello,   World!  How    are YOU???"
        processed = loader._preprocess_text(text)
        
        assert processed == "hello world how are you"
    
    @pytest.mark.unit
    @patch('os.path.exists', return_value=False)
    def test_load_corpus_file_not_found(self, mock_exists):
        """Test loading corpus when file doesn't exist."""
        config = DatasetConfig(
            dataset_path="/test/path",
            corpus_file="nonexistent.jsonl"
        )
        loader = TextDatasetLoader(config)
        
        with pytest.raises(FileNotFoundError):
            loader.load_corpus()
    
    @pytest.mark.unit
    @patch('builtins.open', new_callable=mock_open)
    @patch('os.path.exists', return_value=True)
    def test_load_corpus_invalid_json(self, mock_exists, mock_file):
        """Test loading corpus with invalid JSON."""
        # Mock invalid JSONL content
        jsonl_content = [
            '{"_id": "doc1", "title": "Test Document 1"}',  # Valid
            'invalid json line',  # Invalid
            '{"_id": "doc2", "title": "Test Document 2"}'   # Valid
        ]
        mock_file.return_value.__iter__ = lambda self: iter(jsonl_content)
        
        config = DatasetConfig(dataset_path="/test/path")
        loader = TextDatasetLoader(config)
        
        # Should handle invalid lines gracefully
        corpus = loader.load_corpus()
        
        # Should only load valid entries
        assert len(corpus) == 2
        assert "doc1" in corpus
        assert "doc2" in corpus
    
    @pytest.mark.unit
    def test_validate_dataset_success(self):
        """Test successful dataset validation."""
        config = DatasetConfig(dataset_path="/test/path")
        loader = TextDatasetLoader(config)
        
        # Mock loaded data
        loader._corpus = {"doc1": {"text": "content"}}
        loader._queries = {"q1": {"text": "query"}}
        loader._qrels = {"q1": {"doc1": 1}}
        
        result = loader.validate_dataset()
        assert result is True
    
    @pytest.mark.unit
    def test_validate_dataset_missing_data(self):
        """Test dataset validation with missing data."""
        config = DatasetConfig(dataset_path="/test/path")
        loader = TextDatasetLoader(config)
        
        # Mock empty data
        loader._corpus = {}
        loader._queries = {}
        loader._qrels = {}
        
        result = loader.validate_dataset()
        assert result is False
    
    @pytest.mark.unit
    def test_max_samples_limit(self):
        """Test max_samples configuration."""
        config = DatasetConfig(
            dataset_path="/test/path",
            max_samples=1
        )
        loader = TextDatasetLoader(config)
        
        # Mock data with more than max_samples
        with patch.object(loader, '_load_jsonl_file') as mock_load:
            mock_load.return_value = [
                {"_id": "doc1", "text": "content1"},
                {"_id": "doc2", "text": "content2"},
                {"_id": "doc3", "text": "content3"}
            ]
            
            corpus = loader.load_corpus()
            
            # Should only load max_samples entries
            assert len(corpus) == 1
    
    @pytest.mark.unit
    def test_caching_functionality(self):
        """Test caching functionality."""
        config = DatasetConfig(
            dataset_path="/test/path",
            cache_enabled=True
        )
        loader = TextDatasetLoader(config)
        
        # This would test actual caching implementation
        # For now, just verify cache config
        assert loader.config.cache_enabled is True
    
    @pytest.mark.unit
    def test_format_detection(self):
        """Test automatic format detection."""
        # Test JSONL detection
        config = DatasetConfig(dataset_path="/test/path", corpus_file="corpus.jsonl")
        loader = TextDatasetLoader(config)
        detected_format = loader._detect_format("corpus.jsonl")
        assert detected_format == "jsonl"
        
        # Test CSV detection
        detected_format = loader._detect_format("corpus.csv")
        assert detected_format == "csv"
        
        # Test TSV detection
        detected_format = loader._detect_format("corpus.tsv")
        assert detected_format == "tsv"
        
        # Test JSON detection
        detected_format = loader._detect_format("corpus.json")
        assert detected_format == "json"


@pytest.mark.unit
class TestTextDatasetIntegration:
    """Integration-style tests for TextDatasetLoader."""
    
    def test_complete_dataset_loading(self, temp_dir):
        """Test loading a complete dataset."""
        # Create temporary files
        corpus_file = temp_dir / "corpus.jsonl"
        queries_file = temp_dir / "queries.jsonl"
        qrels_file = temp_dir / "qrels.tsv"
        
        # Write test data
        with open(corpus_file, 'w') as f:
            f.write('{"_id": "doc1", "text": "Machine learning content"}\n')
            f.write('{"_id": "doc2", "text": "Neural network content"}\n')
        
        with open(queries_file, 'w') as f:
            f.write('{"_id": "q1", "text": "What is machine learning?"}\n')
        
        with open(qrels_file, 'w') as f:
            f.write("q1\t0\tdoc1\t1\n")
            f.write("q1\t0\tdoc2\t0\n")
        
        # Load dataset
        config = DatasetConfig(
            dataset_path=str(temp_dir),
            corpus_file="corpus.jsonl",
            queries_file="queries.jsonl",
            qrels_file="qrels.tsv"
        )
        loader = TextDatasetLoader(config)
        
        corpus = loader.load_corpus()
        queries = loader.load_queries()
        qrels = loader.load_qrels()
        
        assert len(corpus) == 2
        assert len(queries) == 1
        assert len(qrels) == 1
        assert loader.validate_dataset() is True


if __name__ == "__main__":
    pytest.main([__file__])
