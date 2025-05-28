"""
Performance tests for dataset loaders.

This module contains performance tests to measure and benchmark
dataset loading speed, memory usage, and optimization effectiveness.
"""

import json
import os
import tempfile
import time
import unittest
from pathlib import Path
from typing import Dict, Any, List
import tracemalloc
import gc

import pytest
from PIL import Image

from newaibench.datasets import (
    TextDatasetLoader,
    DocumentImageDatasetLoader,
    DatasetConfig,
    DocumentImageDatasetConfig
)


class PerformanceTestBase(unittest.TestCase):
    """Base class for performance tests."""
    
    def setUp(self):
        """Set up performance test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.large_dataset_size = 1000  # Adjust for testing
        self.medium_dataset_size = 100
        
    def tearDown(self):
        """Clean up performance test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
        gc.collect()  # Force garbage collection
    
    def measure_time_and_memory(self, func, *args, **kwargs):
        """Measure execution time and memory usage of a function."""
        # Start memory tracing
        tracemalloc.start()
        
        # Measure execution time
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        
        # Get memory usage
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        
        return {
            'result': result,
            'execution_time': end_time - start_time,
            'current_memory_mb': current / 1024 / 1024,
            'peak_memory_mb': peak / 1024 / 1024
        }
    
    def create_large_text_dataset(self, size: int = None):
        """Create a large text dataset for performance testing."""
        if size is None:
            size = self.large_dataset_size
            
        print(f"Creating large text dataset with {size} documents...")
        
        # Create corpus
        corpus_path = os.path.join(self.temp_dir, "corpus.jsonl")
        with open(corpus_path, "w") as f:
            for i in range(size):
                doc = {
                    "doc_id": f"doc{i}",
                    "title": f"Document {i} Title",
                    "text": f"This is the content of document {i}. " * 50,  # ~2.5KB per doc
                    "metadata": {
                        "category": f"cat{i % 10}",
                        "score": i / size,
                        "tags": [f"tag{j}" for j in range(i % 5)]
                    }
                }
                f.write(json.dumps(doc) + "\n")
        
        # Create queries
        queries_path = os.path.join(self.temp_dir, "queries.jsonl")
        num_queries = min(size // 10, 100)  # 10% of docs or 100 max
        with open(queries_path, "w") as f:
            for i in range(num_queries):
                query = {
                    "query_id": f"q{i}",
                    "text": f"Find document about topic {i}"
                }
                f.write(json.dumps(query) + "\n")
        
        # Create qrels
        qrels_path = os.path.join(self.temp_dir, "qrels.jsonl")
        with open(qrels_path, "w") as f:
            for i in range(num_queries):
                # Each query relevant to 2-3 documents
                for j in range(2):
                    doc_idx = (i * 3 + j) % size
                    qrel = {
                        "query_id": f"q{i}",
                        "doc_id": f"doc{doc_idx}",
                        "relevance": 1 if j == 0 else 0  # First one highly relevant
                    }
                    f.write(json.dumps(qrel) + "\n")
        
        print(f"Created dataset with {size} documents, {num_queries} queries")
        return size, num_queries
    
    def create_large_image_dataset(self, size: int = None):
        """Create a large image dataset for performance testing."""
        if size is None:
            size = self.medium_dataset_size  # Smaller for images
            
        print(f"Creating large image dataset with {size} documents...")
        
        # Create images directory
        images_dir = os.path.join(self.temp_dir, "images")
        os.makedirs(images_dir, exist_ok=True)
        
        # Create test images (small to save space)
        for i in range(size):
            img_path = os.path.join(images_dir, f"doc{i}.jpg")
            # Create small images to save space and time
            img = Image.new('RGB', (50, 50), color=(i % 255, (i*2) % 255, (i*3) % 255))
            img.save(img_path, quality=50)  # Lower quality for speed
        
        # Create corpus
        corpus_path = os.path.join(self.temp_dir, "corpus.jsonl")
        with open(corpus_path, "w") as f:
            for i in range(size):
                doc = {
                    "doc_id": f"doc{i}",
                    "title": f"Image Document {i}",
                    "image_path": f"images/doc{i}.jpg",
                    "description": f"Description for image document {i}",
                    "metadata": {
                        "format": "jpg",
                        "category": f"imgcat{i % 5}",
                        "synthetic": True
                    }
                }
                f.write(json.dumps(doc) + "\n")
        
        # Create queries and qrels (similar to text)
        num_queries = min(size // 10, 50)
        
        queries_path = os.path.join(self.temp_dir, "queries.jsonl")
        with open(queries_path, "w") as f:
            for i in range(num_queries):
                query = {
                    "query_id": f"q{i}",
                    "text": f"Find image about topic {i}"
                }
                f.write(json.dumps(query) + "\n")
        
        qrels_path = os.path.join(self.temp_dir, "qrels.jsonl")
        with open(qrels_path, "w") as f:
            for i in range(num_queries):
                for j in range(2):
                    doc_idx = (i * 2 + j) % size
                    qrel = {
                        "query_id": f"q{i}",
                        "doc_id": f"doc{doc_idx}",
                        "relevance": 1
                    }
                    f.write(json.dumps(qrel) + "\n")
        
        print(f"Created image dataset with {size} documents, {num_queries} queries")
        return size, num_queries


class TestTextDatasetLoaderPerformance(PerformanceTestBase):
    """Performance tests for TextDatasetLoader."""
    
    def test_large_dataset_loading_speed(self):
        """Test loading speed with large text dataset."""
        # Create large dataset
        doc_count, query_count = self.create_large_text_dataset(self.large_dataset_size)
        
        # Configure loader
        config = DatasetConfig(
            dataset_path=self.temp_dir,
            cache_enabled=False  # Test raw loading speed
        )
        loader = TextDatasetLoader(config)
        
        # Measure corpus loading
        corpus_metrics = self.measure_time_and_memory(loader.load_corpus)
        corpus = corpus_metrics['result']
        
        # Measure queries loading
        queries_metrics = self.measure_time_and_memory(loader.load_queries)
        queries = queries_metrics['result']
        
        # Measure qrels loading
        qrels_metrics = self.measure_time_and_memory(loader.load_qrels)
        qrels = qrels_metrics['result']
        
        # Print performance metrics
        print("\n=== Text Dataset Loading Performance ===")
        print(f"Dataset size: {doc_count} documents, {query_count} queries")
        print(f"Corpus loading: {corpus_metrics['execution_time']:.2f}s, "
              f"Peak memory: {corpus_metrics['peak_memory_mb']:.1f}MB")
        print(f"Queries loading: {queries_metrics['execution_time']:.2f}s, "
              f"Peak memory: {queries_metrics['peak_memory_mb']:.1f}MB")
        print(f"QRels loading: {qrels_metrics['execution_time']:.2f}s, "
              f"Peak memory: {qrels_metrics['peak_memory_mb']:.1f}MB")
        
        # Performance assertions
        self.assertEqual(len(corpus), doc_count)
        self.assertEqual(len(queries), query_count)
        self.assertGreater(len(qrels), 0)
        
        # Speed assertions (adjust thresholds as needed)
        docs_per_second = doc_count / corpus_metrics['execution_time']
        print(f"Loading speed: {docs_per_second:.1f} docs/second")
        self.assertGreater(docs_per_second, 100)  # At least 100 docs/second
        
        # Memory assertions (should be reasonable)
        mb_per_doc = corpus_metrics['peak_memory_mb'] / doc_count
        print(f"Memory usage: {mb_per_doc:.3f} MB/document")
        self.assertLess(mb_per_doc, 1.0)  # Less than 1MB per document
    
    def test_caching_performance(self):
        """Test performance improvement with caching."""
        # Create medium dataset
        doc_count, _ = self.create_large_text_dataset(self.medium_dataset_size)
        
        # Configure loader with caching
        config = DatasetConfig(
            dataset_path=self.temp_dir,
            cache_enabled=True,
            metadata={"name": "perf_test_cache"}
        )
        loader = TextDatasetLoader(config)
        
        # First load (cold cache)
        cold_metrics = self.measure_time_and_memory(loader.load_corpus)
        
        # Second load (warm cache)
        warm_metrics = self.measure_time_and_memory(loader.load_corpus)
        
        print("\n=== Caching Performance ===")
        print(f"Cold cache: {cold_metrics['execution_time']:.3f}s")
        print(f"Warm cache: {warm_metrics['execution_time']:.3f}s")
        print(f"Speedup: {cold_metrics['execution_time'] / warm_metrics['execution_time']:.1f}x")
        
        # Cache should provide significant speedup
        speedup = cold_metrics['execution_time'] / warm_metrics['execution_time']
        self.assertGreater(speedup, 5.0)  # At least 5x speedup
        
        # Results should be identical
        self.assertEqual(cold_metrics['result'], warm_metrics['result'])
    
    def test_preprocessing_performance_impact(self):
        """Test performance impact of different preprocessing options."""
        # Create dataset
        doc_count, _ = self.create_large_text_dataset(self.medium_dataset_size)
        
        # Test with no preprocessing
        config_none = DatasetConfig(
            dataset_path=self.temp_dir,
            cache_enabled=False,
            preprocessing_options={},
            metadata={"name": "perf_test_none"}
        )
        loader_none = TextDatasetLoader(config_none)
        none_metrics = self.measure_time_and_memory(loader_none.load_corpus)
        
        # Test with basic preprocessing
        config_basic = DatasetConfig(
            dataset_path=self.temp_dir,
            cache_enabled=False,
            preprocessing_options={"lowercase": True, "normalize_whitespace": True},
            metadata={"name": "perf_test_basic"}
        )
        loader_basic = TextDatasetLoader(config_basic)
        basic_metrics = self.measure_time_and_memory(loader_basic.load_corpus)
        
        # Test with full preprocessing
        config_full = DatasetConfig(
            dataset_path=self.temp_dir,
            cache_enabled=False,
            preprocessing_options={
                "lowercase": True,
                "normalize_whitespace": True,
                "remove_special_chars": True,
                "strip_html": True
            },
            metadata={"name": "perf_test_full"}
        )
        loader_full = TextDatasetLoader(config_full)
        full_metrics = self.measure_time_and_memory(loader_full.load_corpus)
        
        print("\n=== Preprocessing Performance Impact ===")
        print(f"No preprocessing: {none_metrics['execution_time']:.3f}s")
        print(f"Basic preprocessing: {basic_metrics['execution_time']:.3f}s")
        print(f"Full preprocessing: {full_metrics['execution_time']:.3f}s")
        
        # Preprocessing should have reasonable overhead
        basic_overhead = basic_metrics['execution_time'] / none_metrics['execution_time']
        full_overhead = full_metrics['execution_time'] / none_metrics['execution_time']
        
        print(f"Basic overhead: {basic_overhead:.1f}x")
        print(f"Full overhead: {full_overhead:.1f}x")
        
        # Overhead should be reasonable
        self.assertLess(basic_overhead, 2.0)  # Less than 2x overhead
        self.assertLess(full_overhead, 3.0)   # Less than 3x overhead
    
    def test_memory_efficiency_large_dataset(self):
        """Test memory efficiency with very large dataset."""
        # Create large dataset
        large_size = min(self.large_dataset_size * 2, 2000)  # Cap for CI
        doc_count, _ = self.create_large_text_dataset(large_size)
        
        config = DatasetConfig(
            dataset_path=self.temp_dir,
            cache_enabled=False,
            metadata={"name": "perf_test_memory"}
        )
        loader = TextDatasetLoader(config)
        
        # Monitor memory during loading
        tracemalloc.start()
        
        start_memory = tracemalloc.get_traced_memory()[0]
        corpus = loader.load_corpus()
        peak_memory = tracemalloc.get_traced_memory()[1]
        
        tracemalloc.stop()
        
        memory_used_mb = (peak_memory - start_memory) / 1024 / 1024
        memory_per_doc_kb = (memory_used_mb * 1024) / doc_count
        
        print(f"\n=== Memory Efficiency ===")
        print(f"Dataset size: {doc_count} documents")
        print(f"Total memory used: {memory_used_mb:.1f} MB")
        print(f"Memory per document: {memory_per_doc_kb:.1f} KB")
        
        # Memory usage should be efficient
        self.assertLess(memory_per_doc_kb, 100)  # Less than 100KB per doc in memory
        self.assertEqual(len(corpus), doc_count)


class TestImageDatasetLoaderPerformance(PerformanceTestBase):
    """Performance tests for DocumentImageDatasetLoader."""
    
    def test_image_dataset_loading_speed(self):
        """Test loading speed with image dataset."""
        # Create image dataset (smaller than text for practicality)
        doc_count, query_count = self.create_large_image_dataset(self.medium_dataset_size)
        
        # Configure loader without OCR for speed
        config = DocumentImageDatasetConfig(
            dataset_path=self.temp_dir,
            cache_enabled=False,
            require_ocr_text=False,  # Was 'ocr_enabled' in original code
            metadata={"name": "perf_test_image"}
        )
        loader = DocumentImageDatasetLoader(config)
        
        # Measure loading performance
        corpus_metrics = self.measure_time_and_memory(loader.load_corpus)
        corpus = corpus_metrics['result']
        
        print("\n=== Image Dataset Loading Performance ===")
        print(f"Dataset size: {doc_count} images")
        print(f"Loading time: {corpus_metrics['execution_time']:.2f}s")
        print(f"Peak memory: {corpus_metrics['peak_memory_mb']:.1f}MB")
        
        images_per_second = doc_count / corpus_metrics['execution_time']
        print(f"Loading speed: {images_per_second:.1f} images/second")
        
        # Performance assertions
        self.assertEqual(len(corpus), doc_count)
        self.assertGreater(images_per_second, 10)  # At least 10 images/second
    
    def test_image_validation_performance(self):
        """Test performance of image validation."""
        # Create image dataset
        doc_count, _ = self.create_large_image_dataset(50)  # Smaller for validation test
        
        config = DocumentImageDatasetConfig(
            dataset_path=self.temp_dir,
            cache_enabled=False,
            require_ocr_text=False,  # Was 'ocr_enabled' in original code
            metadata={"name": "perf_test_validation"}
        )
        loader = DocumentImageDatasetLoader(config)
        
        # Measure validation overhead
        start_time = time.time()
        corpus = loader.load_corpus()
        end_time = time.time()
        
        validation_time = end_time - start_time
        validations_per_second = doc_count / validation_time
        
        print("\n=== Image Validation Performance ===")
        print(f"Validated {doc_count} images in {validation_time:.2f}s")
        print(f"Validation speed: {validations_per_second:.1f} images/second")
        
        # All images should be validated
        for doc_id, doc in corpus.items():
            self.assertIn("metadata", doc)
            self.assertIn("image_path", doc)
    
    def test_ocr_performance_impact(self):
        """Test performance impact of OCR processing."""
        # Create small image dataset for OCR test
        doc_count, _ = self.create_large_image_dataset(10)  # Very small for OCR
        
        # Test without OCR
        config_no_ocr = DocumentImageDatasetConfig(
            dataset_path=self.temp_dir,
            cache_enabled=False,
            require_ocr_text=False,  # Was 'ocr_enabled' in original code
            metadata={"name": "perf_test_no_ocr"}
        )
        loader_no_ocr = DocumentImageDatasetLoader(config_no_ocr)
        no_ocr_metrics = self.measure_time_and_memory(loader_no_ocr.load_corpus)
        
        # Test with OCR (mock it to avoid dependency issues in CI)
        config_with_ocr = DocumentImageDatasetConfig(
            dataset_path=self.temp_dir,
            cache_enabled=False,
            require_ocr_text=True,  # Was 'ocr_enabled' in original code
            metadata={"name": "perf_test_with_ocr"}
        )
        loader_with_ocr = DocumentImageDatasetLoader(config_with_ocr)
        
        # Mock OCR to simulate processing time
        original_extract = loader_with_ocr._extract_ocr_text
        def mock_extract_ocr(image_path):
            time.sleep(0.01)  # Simulate OCR processing time
            return "mocked ocr text"
        loader_with_ocr._extract_ocr_text = mock_extract_ocr
        
        with_ocr_metrics = self.measure_time_and_memory(loader_with_ocr.load_corpus)
        
        print("\n=== OCR Performance Impact ===")
        print(f"Without OCR: {no_ocr_metrics['execution_time']:.3f}s")
        print(f"With OCR: {with_ocr_metrics['execution_time']:.3f}s")
        
        ocr_overhead = with_ocr_metrics['execution_time'] / no_ocr_metrics['execution_time']
        print(f"OCR overhead: {ocr_overhead:.1f}x")
        
        # OCR should add reasonable overhead
        self.assertGreater(ocr_overhead, 1.0)  # OCR should take some time
        self.assertLess(ocr_overhead, 10.0)    # But not too much for small images


class TestCrossLoaderPerformanceComparison(PerformanceTestBase):
    """Compare performance between different loader types."""
    
    def test_text_vs_image_loader_performance(self):
        """Compare performance between text and image loaders on same data."""
        # Create mixed dataset that both loaders can process
        doc_count = 50
        
        # Create text version
        text_corpus = []
        for i in range(doc_count):
            doc = {
                "doc_id": f"doc{i}",
                "title": f"Document {i}",
                "text": f"Content for document {i} " * 20,
                "image_path": f"images/doc{i}.jpg"  # Include image path
            }
            text_corpus.append(doc)
        
        with open(os.path.join(self.temp_dir, "corpus.jsonl"), "w") as f:
            for doc in text_corpus:
                f.write(json.dumps(doc) + "\n")
        
        # Create images for image loader
        images_dir = os.path.join(self.temp_dir, "images")
        os.makedirs(images_dir, exist_ok=True)
        for i in range(doc_count):
            img_path = os.path.join(images_dir, f"doc{i}.jpg")
            img = Image.new('RGB', (50, 50), color='white')
            img.save(img_path)
        
        # Test text loader
        text_config = DatasetConfig(
            dataset_path=self.temp_dir,
            cache_enabled=False,
            metadata={"name": "perf_comparison_text"}
        )
        text_loader = TextDatasetLoader(text_config)
        text_metrics = self.measure_time_and_memory(text_loader.load_corpus)
        
        # Test image loader
        image_config = DocumentImageDatasetConfig(
            dataset_path=self.temp_dir,
            cache_enabled=False,
            require_ocr_text=False,  # Was 'ocr_enabled' in original code
            metadata={"name": "perf_comparison_image"}
        )
        image_loader = DocumentImageDatasetLoader(image_config)
        image_metrics = self.measure_time_and_memory(image_loader.load_corpus)
        
        print("\n=== Text vs Image Loader Performance ===")
        print(f"Text loader: {text_metrics['execution_time']:.3f}s, "
              f"{text_metrics['peak_memory_mb']:.1f}MB")
        print(f"Image loader: {image_metrics['execution_time']:.3f}s, "
              f"{image_metrics['peak_memory_mb']:.1f}MB")
        
        # Both should process same number of documents
        self.assertEqual(len(text_metrics['result']), len(image_metrics['result']))
        
        # Document processing comparisons
        print(f"Speed ratio (text/image): "
              f"{text_metrics['execution_time'] / image_metrics['execution_time']:.1f}")
        print(f"Memory ratio (text/image): "
              f"{text_metrics['peak_memory_mb'] / image_metrics['peak_memory_mb']:.1f}")


@pytest.mark.performance
class TestPerformanceRegression(PerformanceTestBase):
    """Test for performance regressions."""
    
    def test_baseline_performance_benchmarks(self):
        """Establish baseline performance benchmarks."""
        # Create standardized test datasets
        text_size = 500
        image_size = 50
        
        # Text dataset benchmark
        text_doc_count, _ = self.create_large_text_dataset(text_size)
        text_config = DatasetConfig(
            dataset_path=self.temp_dir,
            cache_enabled=False,
            metadata={"name": "benchmark_text"}
        )
        text_loader = TextDatasetLoader(text_config)
        text_metrics = self.measure_time_and_memory(text_loader.load_corpus)
        
        # Image dataset benchmark
        image_doc_count, _ = self.create_large_image_dataset(image_size)
        image_config = DocumentImageDatasetConfig(
            dataset_path=self.temp_dir,
            cache_enabled=False,
            require_ocr_text=False,  # Was 'ocr_enabled' in original code
            metadata={"name": "benchmark_image"}
        )
        image_loader = DocumentImageDatasetLoader(image_config)
        image_metrics = self.measure_time_and_memory(image_loader.load_corpus)
        
        # Calculate performance metrics
        text_docs_per_sec = text_doc_count / text_metrics['execution_time']
        text_mb_per_doc = text_metrics['peak_memory_mb'] / text_doc_count
        
        image_docs_per_sec = image_doc_count / image_metrics['execution_time']
        image_mb_per_doc = image_metrics['peak_memory_mb'] / image_doc_count
        
        print("\n=== Performance Benchmarks ===")
        print(f"Text Loader Baseline:")
        print(f"  - Speed: {text_docs_per_sec:.1f} docs/second")
        print(f"  - Memory: {text_mb_per_doc:.3f} MB/document")
        print(f"Image Loader Baseline:")
        print(f"  - Speed: {image_docs_per_sec:.1f} docs/second")
        print(f"  - Memory: {image_mb_per_doc:.3f} MB/document")
        
        # Store benchmarks for regression testing
        benchmarks = {
            'text_docs_per_second': text_docs_per_sec,
            'text_mb_per_doc': text_mb_per_doc,
            'image_docs_per_second': image_docs_per_sec,
            'image_mb_per_doc': image_mb_per_doc
        }
        
        # Write benchmarks to file for CI/CD tracking
        benchmark_file = os.path.join(self.temp_dir, "performance_benchmarks.json")
        with open(benchmark_file, "w") as f:
            json.dump(benchmarks, f, indent=2)
        
        print(f"Benchmarks saved to: {benchmark_file}")
        
        # Assert minimum performance requirements
        self.assertGreater(text_docs_per_sec, 50)     # At least 50 text docs/sec
        self.assertLess(text_mb_per_doc, 0.5)         # Less than 0.5MB per text doc
        self.assertGreater(image_docs_per_sec, 5)     # At least 5 image docs/sec
        self.assertLess(image_mb_per_doc, 2.0)        # Less than 2MB per image doc


if __name__ == "__main__":
    # Run performance tests
    unittest.main(verbosity=2)
