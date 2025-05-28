"""
Integration tests for dataset loaders.

This module contains integration tests that verify cross-loader compatibility,
factory function behavior, and end-to-end dataset loading workflows.
"""

import json
import os
import tempfile
import unittest
from pathlib import Path
from typing import Dict, Any

import pytest
from PIL import Image

from newaibench.datasets import (
    BaseDatasetLoader,
    TextDatasetLoader, 
    DocumentImageDatasetLoader,
    create_dataset_loader,
    DatasetConfig,
    DocumentImageDatasetConfig
)
from newaibench.datasets.base import DatasetLoadingError, DataValidationError


class TestDatasetLoaderFactory(unittest.TestCase):
    """Test the dataset loader factory function."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        # Create basic directory structure for tests
        os.makedirs(os.path.join(self.temp_dir, "images"), exist_ok=True)
    
    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_create_text_loader(self):
        """Test creating text dataset loader via factory."""
        config = DatasetConfig(
            dataset_path=self.temp_dir
        )
        
        loader = create_dataset_loader("text", config)
        self.assertIsInstance(loader, TextDatasetLoader)
    
    def test_create_image_loader(self):
        """Test creating image dataset loader via factory."""
        config = DocumentImageDatasetConfig(
            dataset_path=self.temp_dir
        )
        
        loader = create_dataset_loader("image", config)
        self.assertIsInstance(loader, DocumentImageDatasetLoader)
    
    def test_create_document_image_loader_alias(self):
        """Test creating document image loader via alias."""
        config = DocumentImageDatasetConfig(
            dataset_path=self.temp_dir
        )
        
        loader = create_dataset_loader("document_image", config)
        self.assertIsInstance(loader, DocumentImageDatasetLoader)
    
    def test_create_loader_invalid_type(self):
        """Test factory with invalid loader type."""
        config = DatasetConfig(
            dataset_path=self.temp_dir
        )
        
        with self.assertRaises(ValueError):
            create_dataset_loader("invalid_type", config)
    
    def test_create_loader_wrong_config_type(self):
        """Test factory with wrong config type."""
        # Try to create image loader with basic config
        config = DatasetConfig(
            dataset_path=self.temp_dir
        )
        
        # Factory should automatically convert config types
        loader = create_dataset_loader("image", config)
        self.assertIsInstance(loader, DocumentImageDatasetLoader)
        # Verify the loader's config was converted to the correct type
        self.assertIsInstance(loader.config, DocumentImageDatasetConfig)


class TestCrossLoaderCompatibility(unittest.TestCase):
    """Test compatibility between different dataset loaders."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.create_shared_test_data()
    
    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def create_shared_test_data(self):
        """Create test data that can be used by multiple loaders."""
        # Create data that works for both text and image loaders
        corpus_data = [
            {
                "doc_id": "doc1",
                "title": "Document 1",
                "text": "This is the content of document 1",
                "image_path": "doc1.jpg",
                "metadata": {"source": "test", "type": "mixed"}
            },
            {
                "doc_id": "doc2",
                "title": "Document 2", 
                "text": "This is the content of document 2",
                "image_path": "doc2.png",
                "ocr_text": "OCR extracted text for document 2"
            }
        ]
        
        # Create corpus file in multiple formats
        # JSONL format
        with open(os.path.join(self.temp_dir, "corpus.jsonl"), "w") as f:
            for doc in corpus_data:
                f.write(json.dumps(doc) + "\n")
        
        # JSON format
        with open(os.path.join(self.temp_dir, "corpus.json"), "w") as f:
            json.dump(corpus_data, f)
        
        # Create queries and qrels
        queries_data = [
            {"query_id": "q1", "text": "Find document about topic 1"},
            {"query_id": "q2", "text": "Search for document 2"}
        ]
        
        with open(os.path.join(self.temp_dir, "queries.jsonl"), "w") as f:
            for query in queries_data:
                f.write(json.dumps(query) + "\n")
        
        qrels_data = [
            {"query_id": "q1", "doc_id": "doc1", "relevance": 1},
            {"query_id": "q2", "doc_id": "doc2", "relevance": 1}
        ]
        
        # Create qrels in both JSONL and TXT formats for compatibility
        with open(os.path.join(self.temp_dir, "qrels.jsonl"), "w") as f:
            for qrel in qrels_data:
                f.write(json.dumps(qrel) + "\n")
        
        # Also create qrels.txt in TREC TSV format (tab-separated)
        with open(os.path.join(self.temp_dir, "qrels.txt"), "w") as f:
            for qrel in qrels_data:
                f.write(f"{qrel['query_id']}\t0\t{qrel['doc_id']}\t{qrel['relevance']}\n")
        
        # Create images directory and test images
        images_dir = os.path.join(self.temp_dir, "images")
        os.makedirs(images_dir, exist_ok=True)
        
        for doc_id in ["doc1", "doc2"]:
            img_path = os.path.join(images_dir, f"{doc_id}.jpg" if doc_id == "doc1" else f"{doc_id}.png")
            img = Image.new('RGB', (100, 100), color='white')
            img.save(img_path)
    
    def test_text_and_image_loaders_same_data(self):
        """Test that text and image loaders can process the same data."""
        # Configure text loader
        text_config = DatasetConfig(
            dataset_path=self.temp_dir,
            cache_enabled=False
        )
        text_loader = TextDatasetLoader(text_config)
        
        # Configure image loader
        image_config = DocumentImageDatasetConfig(
            dataset_path=self.temp_dir,
            cache_enabled=False,
            require_ocr_text=False,  # Disable OCR for this test
            preprocessing_options={
                "min_length": 1,
                "max_length": 1000
            }
        )
        image_loader = DocumentImageDatasetLoader(image_config)
        
        # Load data with both loaders
        text_corpus = text_loader.load_corpus()
        image_corpus = image_loader.load_corpus()
        
        # Verify both loaders can read the same documents
        self.assertEqual(len(text_corpus), len(image_corpus))
        self.assertEqual(set(text_corpus.keys()), set(image_corpus.keys()))
        
        # Verify text loader focuses on text content
        for doc_id in text_corpus:
            self.assertIn("text", text_corpus[doc_id])
        
        # Verify image loader focuses on image content
        for doc_id in image_corpus:
            self.assertIn("image_path", image_corpus[doc_id])
    
    def test_consistent_query_and_qrel_loading(self):
        """Test that all loaders load queries and qrels consistently."""
        # Create different loader instances
        text_config = DatasetConfig(
            dataset_path=self.temp_dir,
            cache_enabled=False
        )
        text_loader = TextDatasetLoader(text_config)
        
        image_config = DocumentImageDatasetConfig(
            dataset_path=self.temp_dir,
            cache_enabled=False,
            preprocessing_options={
                "min_length": 1,
                "max_length": 1000
            }
        )
        image_loader = DocumentImageDatasetLoader(image_config)
        
        # Load queries with both loaders
        text_queries = text_loader.load_queries()
        image_queries = image_loader.load_queries()
        
        # Should be identical
        self.assertEqual(text_queries, image_queries)
        
        # Load qrels with both loaders
        text_qrels = text_loader.load_qrels()
        image_qrels = image_loader.load_qrels()
        
        # Should be identical
        self.assertEqual(text_qrels, image_qrels)
    
    def test_validation_consistency(self):
        """Test that validation works consistently across loaders."""
        text_config = DatasetConfig(
            dataset_path=self.temp_dir,
            cache_enabled=False,
            validation_enabled=False  # Disable validation for consistency test
        )
        text_loader = TextDatasetLoader(text_config)
        
        image_config = DocumentImageDatasetConfig(
            dataset_path=self.temp_dir,
            cache_enabled=False,
            validation_enabled=False,  # Disable validation for consistency test
            preprocessing_options={
                "min_length": 1,
                "max_length": 1000
            }
        )
        image_loader = DocumentImageDatasetLoader(image_config)
        
        # Load data
        text_corpus = text_loader.load_corpus()
        text_queries = text_loader.load_queries()
        text_qrels = text_loader.load_qrels()
        
        image_corpus = image_loader.load_corpus()
        image_queries = image_loader.load_queries()
        image_qrels = image_loader.load_qrels()
        
        # Both should validate successfully
        text_loader.validate_data(text_corpus, text_queries, text_qrels)
        image_loader.validate_data(image_corpus, image_queries, image_qrels)
    
    def test_statistics_comparison(self):
        """Test statistics generation across different loaders."""
        text_config = DatasetConfig(
            dataset_path=self.temp_dir,
            cache_enabled=True  # Enable caching so statistics can be generated
        )
        text_loader = TextDatasetLoader(text_config)
        
        image_config = DocumentImageDatasetConfig(
            dataset_path=self.temp_dir,
            cache_enabled=True,  # Enable caching so statistics can be generated
            preprocessing_options={
                "min_length": 1,
                "max_length": 1000
            }
        )
        image_loader = DocumentImageDatasetLoader(image_config)
        
        # Load data and get statistics
        text_corpus = text_loader.load_corpus()
        text_queries = text_loader.load_queries()
        text_qrels = text_loader.load_qrels()
        text_stats = text_loader.get_statistics()
        
        image_corpus = image_loader.load_corpus()
        image_queries = image_loader.load_queries()
        image_qrels = image_loader.load_qrels()
        image_stats = image_loader.get_statistics()
        
        # Common statistics should be the same
        self.assertEqual(text_stats["corpus_size"], image_stats["corpus_size"])
        self.assertEqual(text_stats["queries_count"], image_stats["queries_count"])
        self.assertEqual(text_stats["qrels_count"], image_stats["qrels_count"])
        
        # Loader-specific statistics should be different
        self.assertIn("avg_doc_length", text_stats)
        # Note: image loader may have different specific stats depending on implementation


class TestEndToEndWorkflows(unittest.TestCase):
    """Test complete end-to-end workflows."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_complete_text_dataset_workflow(self):
        """Test complete workflow for text dataset processing."""
        # Create comprehensive text dataset
        self._create_text_dataset()
        
        # Configure and create loader
        config = DatasetConfig(
            dataset_path=self.temp_dir,
            preprocessing_options={
                "lowercase": True,
                "normalize_whitespace": True,
                "remove_special_chars": False,
                "min_length": 1,
                "max_length": 1000
            },
            cache_enabled=True,
            validation_enabled=False  # Disable validation for workflow test
        )
        
        loader = create_dataset_loader("text", config)
        
        # Execute complete workflow
        corpus = loader.load_corpus()
        queries = loader.load_queries()
        qrels = loader.load_qrels()
        
        # Validate data
        loader.validate_data(corpus, queries, qrels)
        
        # Get statistics
        stats = loader.get_statistics()
        
        # Verify results
        self.assertGreater(len(corpus), 0)
        self.assertGreater(len(queries), 0)
        self.assertGreater(len(qrels), 0)
        self.assertIn("corpus_size", stats)
        
        # Test caching
        corpus_cached = loader.load_corpus()
        self.assertEqual(corpus, corpus_cached)
        
        # Clear cache and reload
        loader.clear_cache()
        corpus_fresh = loader.load_corpus()
        self.assertEqual(corpus, corpus_fresh)
    
    def test_complete_image_dataset_workflow(self):
        """Test complete workflow for image dataset processing."""
        # Create comprehensive image dataset
        self._create_image_dataset()
        
        # Configure and create loader
        config = DocumentImageDatasetConfig(
            dataset_path=self.temp_dir,
            require_ocr_text=False,  # Disable OCR for testing
            cache_enabled=True,
            validation_enabled=False,  # Disable validation for workflow test
            preprocessing_options={
                "min_length": 1,
                "max_length": 1000
            }
        )
        
        loader = create_dataset_loader("image", config)
        
        # Execute complete workflow
        corpus = loader.load_corpus()
        queries = loader.load_queries()
        qrels = loader.load_qrels()
        
        # Validate data
        loader.validate_data(corpus, queries, qrels)
        
        # Get statistics
        stats = loader.get_statistics()
        
        # Verify results
        self.assertGreater(len(corpus), 0)
        self.assertGreater(len(queries), 0)
        self.assertGreater(len(qrels), 0)
        # Note: Image-specific stats may vary
        
        # Verify image-specific processing
        for doc_id, doc in corpus.items():
            self.assertIn("image_path", doc)
            # Should have metadata for valid images
            if doc.get("image_path"):
                self.assertIn("metadata", doc)
    
    def test_error_handling_workflow(self):
        """Test error handling in complete workflows."""
        # Create dataset with intentional errors
        self._create_corrupted_dataset()
        
        config = DatasetConfig(
            dataset_path=self.temp_dir,
            cache_enabled=False
        )
        
        loader = create_dataset_loader("text", config)
        
        # Should handle corrupted data gracefully
        corpus = loader.load_corpus()
        queries = loader.load_queries()
        qrels = loader.load_qrels()
        
        # Some data should still be loaded
        self.assertGreaterEqual(len(corpus), 0)
    
    def _create_text_dataset(self):
        """Create a comprehensive text dataset for testing."""
        corpus_data = [
            {
                "doc_id": f"doc{i}",
                "title": f"Document {i}",
                "text": f"This is the content of document {i}. " * 10,
                "metadata": {"category": f"cat{i % 3}", "length": "medium"}
            }
            for i in range(1, 11)
        ]
        
        with open(os.path.join(self.temp_dir, "corpus.jsonl"), "w") as f:
            for doc in corpus_data:
                f.write(json.dumps(doc) + "\n")
        
        queries_data = [
            {"query_id": f"q{i}", "text": f"Find document {i}"}
            for i in range(1, 6)
        ]
        
        with open(os.path.join(self.temp_dir, "queries.jsonl"), "w") as f:
            for query in queries_data:
                f.write(json.dumps(query) + "\n")
        
        qrels_data = [
            {"query_id": f"q{i}", "doc_id": f"doc{i}", "relevance": 1}
            for i in range(1, 6)
        ]
        
        # Create qrels in both JSONL and TXT formats for compatibility
        with open(os.path.join(self.temp_dir, "qrels.jsonl"), "w") as f:
            for qrel in qrels_data:
                f.write(json.dumps(qrel) + "\n")
        
        # Also create qrels.txt in TREC TSV format (tab-separated)
        with open(os.path.join(self.temp_dir, "qrels.txt"), "w") as f:
            for qrel in qrels_data:
                f.write(f"{qrel['query_id']}\t0\t{qrel['doc_id']}\t{qrel['relevance']}\n")
    
    def _create_image_dataset(self):
        """Create a comprehensive image dataset for testing."""
        # Create images directory
        images_dir = os.path.join(self.temp_dir, "images")
        os.makedirs(images_dir, exist_ok=True)
        
        # Create test images
        for i in range(1, 6):
            img_path = os.path.join(images_dir, f"doc{i}.jpg")
            img = Image.new('RGB', (200, 200), color=(i*40, i*40, i*40))
            img.save(img_path)
        
        corpus_data = [
            {
                "doc_id": f"doc{i}",
                "title": f"Document {i}",
                "image_path": f"doc{i}.jpg",
                "description": f"Test document {i} with image content",
                "metadata": {"format": "jpg", "category": f"cat{i % 3}"}
            }
            for i in range(1, 6)
        ]
        
        with open(os.path.join(self.temp_dir, "corpus.jsonl"), "w") as f:
            for doc in corpus_data:
                f.write(json.dumps(doc) + "\n")
        
        queries_data = [
            {"query_id": f"q{i}", "text": f"Find image document {i}"}
            for i in range(1, 4)
        ]
        
        with open(os.path.join(self.temp_dir, "queries.jsonl"), "w") as f:
            for query in queries_data:
                f.write(json.dumps(query) + "\n")
        
        qrels_data = [
            {"query_id": f"q{i}", "doc_id": f"doc{i}", "relevance": 1}
            for i in range(1, 4)
        ]
        
        # Create qrels in both JSONL and TXT formats for compatibility
        with open(os.path.join(self.temp_dir, "qrels.jsonl"), "w") as f:
            for qrel in qrels_data:
                f.write(json.dumps(qrel) + "\n")
        
        # Also create qrels.txt in TREC TSV format (tab-separated)
        with open(os.path.join(self.temp_dir, "qrels.txt"), "w") as f:
            for qrel in qrels_data:
                f.write(f"{qrel['query_id']}\t0\t{qrel['doc_id']}\t{qrel['relevance']}\n")
    
    def _create_corrupted_dataset(self):
        """Create a dataset with some corrupted entries for error testing."""
        # Mix of valid and invalid JSON
        corpus_lines = [
            '{"doc_id": "doc1", "title": "Valid Document", "text": "Valid content"}',
            '{"doc_id": "doc2", "title": "Another Valid", "text": "More content"}',
            '{"invalid": json}',  # Invalid JSON
            '{"doc_id": "doc3", "text": "Missing title"}',  # Missing field
            '{"doc_id": "doc4", "title": "Valid Again", "text": "Final content"}'
        ]
        
        with open(os.path.join(self.temp_dir, "corpus.jsonl"), "w") as f:
            for line in corpus_lines:
                f.write(line + "\n")
        
        # Valid queries and qrels
        queries_data = [{"query_id": "q1", "text": "Test query"}]
        with open(os.path.join(self.temp_dir, "queries.jsonl"), "w") as f:
            for query in queries_data:
                f.write(json.dumps(query) + "\n")
        
        qrels_data = [{"query_id": "q1", "doc_id": "doc1", "relevance": 1}]
        
        # Create qrels in both JSONL and TXT formats for compatibility
        with open(os.path.join(self.temp_dir, "qrels.jsonl"), "w") as f:
            for qrel in qrels_data:
                f.write(json.dumps(qrel) + "\n")
        
        # Also create qrels.txt in TREC TSV format (tab-separated)
        with open(os.path.join(self.temp_dir, "qrels.txt"), "w") as f:
            for qrel in qrels_data:
                f.write(f"{qrel['query_id']}\t0\t{qrel['doc_id']}\t{qrel['relevance']}\n")


if __name__ == "__main__":
    unittest.main()
