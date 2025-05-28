"""
Unit tests for TextDatasetLoader.

This module contains tests for text dataset loading functionality,
including various format support and preprocessing options.
"""

import pytest
import tempfile
import json
import csv
from pathlib import Path
import sys
import os

# Add src to path for testing
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from newaibench.datasets.base import DatasetConfig, DatasetLoadingError
from newaibench.datasets.text import TextDatasetLoader


class TestTextDatasetLoader:
    """Test cases for TextDatasetLoader."""
    
    def test_text_loader_initialization(self, tmp_path):
        """Test TextDatasetLoader initialization."""
        config = DatasetConfig(dataset_path=tmp_path)
        loader = TextDatasetLoader(config)
        
        assert isinstance(loader, TextDatasetLoader)
        assert loader.config == config
    
    def test_load_jsonl_corpus(self, tmp_path):
        """Test loading corpus from JSONL format."""
        # Create sample JSONL corpus
        corpus_file = tmp_path / "corpus.jsonl"
        corpus_data = [
            {"_id": "doc1", "text": "This is document 1"},
            {"_id": "doc2", "text": "This is document 2"},
            {"_id": "doc3", "text": "This is document 3"}
        ]
        
        with open(corpus_file, 'w') as f:
            for doc in corpus_data:
                f.write(json.dumps(doc) + '\n')
        
        config = DatasetConfig(dataset_path=tmp_path, format_type="jsonl")
        loader = TextDatasetLoader(config)
        
        corpus = loader.load_corpus()
        
        assert len(corpus) == 3
        assert "doc1" in corpus
        assert corpus["doc1"]["text"] == "This is document 1"
    
    def test_load_tsv_corpus(self, tmp_path):
        """Test loading corpus from TSV format."""
        # Create sample TSV corpus
        corpus_file = tmp_path / "corpus.tsv"
        
        with open(corpus_file, 'w', newline='') as f:
            writer = csv.writer(f, delimiter='\t')
            writer.writerow(["id", "text", "title"])
            writer.writerow(["doc1", "This is document 1", "Title 1"])
            writer.writerow(["doc2", "This is document 2", "Title 2"])
        
        config = DatasetConfig(
            dataset_path=tmp_path, 
            corpus_file="corpus.tsv",
            format_type="tsv"
        )
        loader = TextDatasetLoader(config)
        
        corpus = loader.load_corpus()
        
        assert len(corpus) == 2
        assert "doc1" in corpus
        assert corpus["doc1"]["text"] == "This is document 1"
        assert corpus["doc1"]["title"] == "Title 1"
    
    def test_load_json_queries(self, tmp_path):
        """Test loading queries from JSON format."""
        # Create sample JSON queries
        queries_file = tmp_path / "queries.json"
        queries_data = {
            "q1": {"_id": "q1", "text": "Query 1"},
            "q2": {"_id": "q2", "text": "Query 2"}
        }
        
        with open(queries_file, 'w') as f:
            json.dump(queries_data, f)
        
        config = DatasetConfig(
            dataset_path=tmp_path,
            queries_file="queries.json",
            format_type="json"
        )
        loader = TextDatasetLoader(config)
        
        queries = loader.load_queries()
        
        assert len(queries) == 2
        assert "q1" in queries
        assert queries["q1"] == "Query 1"
    
    def test_load_tsv_qrels(self, tmp_path):
        """Test loading qrels from TSV format."""
        # Create sample TSV qrels
        qrels_file = tmp_path / "qrels.txt"
        
        with open(qrels_file, 'w') as f:
            f.write("q1\tdoc1\t2\n")
            f.write("q1\tdoc2\t0\n")
            f.write("q2\tdoc2\t1\n")
        
        config = DatasetConfig(
            dataset_path=tmp_path,
            qrels_file="qrels.txt"
        )
        loader = TextDatasetLoader(config)
        
        qrels = loader.load_qrels()
        
        assert len(qrels) == 2
        assert "q1" in qrels
        assert qrels["q1"]["doc1"] == 2
        assert qrels["q1"]["doc2"] == 0
    
    def test_preprocessing_options(self, tmp_path):
        """Test text preprocessing options."""
        # Create sample corpus with mixed case text
        corpus_file = tmp_path / "corpus.jsonl"
        corpus_data = [
            {"_id": "doc1", "text": "This IS a TEST Document!!"},
            {"_id": "doc2", "text": "Another   TEST   Document???"}
        ]
        
        with open(corpus_file, 'w') as f:
            for doc in corpus_data:
                f.write(json.dumps(doc) + '\n')
        
        config = DatasetConfig(
            dataset_path=tmp_path,
            format_type="jsonl",
            preprocessing_options={
                "lowercase": True,
                "remove_special_chars": True,
                "normalize_whitespace": True,
                "min_length": 1,
                "max_length": 1000
            }
        )
        loader = TextDatasetLoader(config)
        
        corpus = loader.load_corpus()
        
        # Check preprocessing was applied
        doc1_text = corpus["doc1"]["text"]
        assert doc1_text.islower()
        assert "!" not in doc1_text
        assert "  " not in doc1_text  # No double spaces
    
    def test_max_samples_limitation(self, tmp_path):
        """Test max_samples configuration."""
        # Create sample corpus with more documents
        corpus_file = tmp_path / "corpus.jsonl"
        corpus_data = [
            {"_id": f"doc{i}", "text": f"Document {i}"} 
            for i in range(10)
        ]
        
        with open(corpus_file, 'w') as f:
            for doc in corpus_data:
                f.write(json.dumps(doc) + '\n')
        
        config = DatasetConfig(
            dataset_path=tmp_path,
            format_type="jsonl",
            max_samples=5
        )
        loader = TextDatasetLoader(config)
        
        corpus = loader.load_corpus()
        
        assert len(corpus) == 5
    
    def test_missing_corpus_file(self, tmp_path):
        """Test error handling for missing corpus file."""
        config = DatasetConfig(
            dataset_path=tmp_path,
            corpus_file="nonexistent.jsonl"
        )
        loader = TextDatasetLoader(config)
        
        with pytest.raises(DatasetLoadingError):
            loader.load_corpus()
    
    def test_invalid_json_handling(self, tmp_path):
        """Test handling of invalid JSON in JSONL files."""
        # Create corpus with some invalid JSON lines
        corpus_file = tmp_path / "corpus.jsonl"
        
        with open(corpus_file, 'w') as f:
            f.write('{"_id": "doc1", "text": "Valid document"}\n')
            f.write('{"_id": "doc2", "text": "Another valid"}\n')
            f.write('invalid json line\n')  # Invalid JSON
            f.write('{"_id": "doc3", "text": "Third document"}\n')
        
        config = DatasetConfig(dataset_path=tmp_path, format_type="jsonl")
        loader = TextDatasetLoader(config)
        
        corpus = loader.load_corpus()
        
        # Should skip invalid lines and load valid ones
        assert len(corpus) == 3
        assert "doc1" in corpus
        assert "doc3" in corpus
    
    def test_different_text_field_names(self, tmp_path):
        """Test handling of different text field names."""
        # Create corpus with different text field names
        corpus_file = tmp_path / "corpus.jsonl"
        corpus_data = [
            {"_id": "doc1", "contents": "Document with contents field"},
            {"_id": "doc2", "content": "Document with content field"},
            {"_id": "doc3", "text": "Document with text field"}
        ]
        
        with open(corpus_file, 'w') as f:
            for doc in corpus_data:
                f.write(json.dumps(doc) + '\n')
        
        config = DatasetConfig(dataset_path=tmp_path, format_type="jsonl")
        loader = TextDatasetLoader(config)
        
        corpus = loader.load_corpus()
        
        assert len(corpus) == 3
        assert corpus["doc1"]["text"] == "Document with contents field"
        assert corpus["doc2"]["text"] == "Document with content field"
        assert corpus["doc3"]["text"] == "Document with text field"
    
    def test_length_filtering(self, tmp_path):
        """Test document length filtering."""
        # Create corpus with documents of different lengths
        corpus_file = tmp_path / "corpus.jsonl"
        corpus_data = [
            {"_id": "doc1", "text": "Short"},  # 1 word
            {"_id": "doc2", "text": "This is a longer document"},  # 5 words
            {"_id": "doc3", "text": "This is a very long document with many words"},  # 9 words
        ]
        
        with open(corpus_file, 'w') as f:
            for doc in corpus_data:
                f.write(json.dumps(doc) + '\n')
        
        config = DatasetConfig(
            dataset_path=tmp_path,
            format_type="jsonl",
            preprocessing_options={
                "min_length": 3,  # Minimum 3 words
                "max_length": 7   # Maximum 7 words
            }
        )
        loader = TextDatasetLoader(config)
        
        corpus = loader.load_corpus()
        
        # Should include doc2 (5 words) and doc3 (truncated to 7 words)
        # doc1 is filtered out for being too short (< 3 words)
        assert len(corpus) == 2
        assert "doc2" in corpus
        assert "doc3" in corpus
        assert corpus["doc2"]["text"] == "This is a longer document"
        # doc3 should be truncated to 7 words
        assert len(corpus["doc3"]["text"].split()) == 7
        assert corpus["doc3"]["text"] == "This is a very long document with"
    
    def test_caching_behavior(self, tmp_path):
        """Test caching behavior."""
        # Create sample files
        corpus_file = tmp_path / "corpus.jsonl"
        with open(corpus_file, 'w') as f:
            f.write('{"_id": "doc1", "text": "Document 1"}\n')
        
        queries_file = tmp_path / "queries.jsonl"
        with open(queries_file, 'w') as f:
            f.write('{"_id": "q1", "text": "Query 1"}\n')
        
        qrels_file = tmp_path / "qrels.txt"
        with open(qrels_file, 'w') as f:
            f.write("q1\tdoc1\t1\n")
        
        config = DatasetConfig(
            dataset_path=tmp_path,
            format_type="jsonl",
            cache_enabled=True
        )
        loader = TextDatasetLoader(config)
        
        # First load
        corpus1 = loader.load_corpus()
        queries1 = loader.load_queries()
        qrels1 = loader.load_qrels()
        
        # Second load (should use cache)
        corpus2 = loader.load_corpus()
        queries2 = loader.load_queries()
        qrels2 = loader.load_qrels()
        
        # Should be the same objects due to caching
        assert corpus1 is corpus2
        assert queries1 is queries2
        assert qrels1 is qrels2


# Test fixtures
@pytest.fixture
def sample_text_dataset(tmp_path):
    """Create a complete sample text dataset for testing."""
    # Create corpus
    corpus_file = tmp_path / "corpus.jsonl"
    corpus_data = [
        {"_id": "doc1", "text": "Machine learning is a subset of artificial intelligence", "title": "ML Intro"},
        {"_id": "doc2", "text": "Information retrieval systems help find relevant documents", "title": "IR Systems"},
        {"_id": "doc3", "text": "Natural language processing deals with text understanding", "title": "NLP Basics"}
    ]
    
    with open(corpus_file, 'w') as f:
        for doc in corpus_data:
            f.write(json.dumps(doc) + '\n')
    
    # Create queries
    queries_file = tmp_path / "queries.jsonl"
    queries_data = [
        {"_id": "q1", "text": "machine learning algorithms"},
        {"_id": "q2", "text": "document retrieval methods"},
        {"_id": "q3", "text": "text processing techniques"}
    ]
    
    with open(queries_file, 'w') as f:
        for query in queries_data:
            f.write(json.dumps(query) + '\n')
    
    # Create qrels
    qrels_file = tmp_path / "qrels.txt"
    with open(qrels_file, 'w') as f:
        f.write("q1\tdoc1\t2\n")
        f.write("q1\tdoc2\t0\n")
        f.write("q1\tdoc3\t1\n")
        f.write("q2\tdoc1\t0\n")
        f.write("q2\tdoc2\t2\n")
        f.write("q2\tdoc3\t0\n")
        f.write("q3\tdoc1\t1\n")
        f.write("q3\tdoc2\t0\n")
        f.write("q3\tdoc3\t2\n")
    
    return tmp_path


class TestTextDatasetLoaderIntegration:
    """Integration tests for TextDatasetLoader."""
    
    def test_complete_workflow(self, sample_text_dataset):
        """Test complete dataset loading workflow."""
        config = DatasetConfig(
            dataset_path=sample_text_dataset,
            format_type="jsonl"
        )
        loader = TextDatasetLoader(config)
        
        # Load all components
        corpus, queries, qrels = loader.load_all()
        
        # Verify loaded data
        assert len(corpus) == 3
        assert len(queries) == 3
        assert len(qrels) == 3
        
        # Check data structure
        assert all("text" in doc for doc in corpus.values())
        assert all(isinstance(query, str) for query in queries.values())
        assert all(isinstance(judgments, dict) for judgments in qrels.values())
        
        # Validate data consistency
        assert loader.validate_data(corpus, queries, qrels)
    
    def test_statistics_generation(self, sample_text_dataset):
        """Test statistics generation for text dataset."""
        config = DatasetConfig(
            dataset_path=sample_text_dataset,
            format_type="jsonl"
        )
        loader = TextDatasetLoader(config)
        
        # Load data
        loader.load_all()
        
        # Get statistics
        stats = loader.get_statistics()
        
        # Verify statistics
        assert stats["corpus_size"] == 3
        assert stats["queries_count"] == 3
        assert stats["qrels_count"] == 3
        assert stats["avg_doc_length"] > 0
        assert stats["avg_query_length"] > 0
        assert stats["total_judgments"] == 9
        assert "relevance_distribution" in stats
    
    def test_error_recovery(self, tmp_path):
        """Test error recovery in dataset loading."""
        # Create dataset with some problematic entries
        corpus_file = tmp_path / "corpus.jsonl"
        with open(corpus_file, 'w') as f:
            f.write('{"_id": "doc1", "text": "Valid document"}\n')
            f.write('invalid json\n')  # Invalid line
            f.write('{"_id": "doc2"}\n')  # Missing text field - will be skipped
            f.write('{"_id": "doc3", "text": "Another valid document"}\n')
        
        queries_file = tmp_path / "queries.jsonl"
        with open(queries_file, 'w') as f:
            f.write('{"_id": "q1", "text": "Valid query"}\n')
        
        qrels_file = tmp_path / "qrels.txt"
        with open(qrels_file, 'w') as f:
            f.write("q1\tdoc1\t1\n")
            f.write("q1\tdoc3\t1\n")
        
        config = DatasetConfig(
            dataset_path=tmp_path,
            format_type="jsonl"
        )
        loader = TextDatasetLoader(config)
        
        # Should handle errors gracefully and load valid data
        corpus, queries, qrels = loader.load_all()
        
        assert len(corpus) == 2  # Only valid documents
        assert len(queries) == 1
        assert len(qrels) == 1


if __name__ == "__main__":
    pytest.main([__file__])
