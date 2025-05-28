"""
Pytest configuration and shared fixtures for NewAIBench testing.
"""
import pytest
import tempfile
import shutil
import os
from pathlib import Path
from unittest.mock import Mock, MagicMock
import pandas as pd
import numpy as np

# Add src to Python path for testing
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


@pytest.fixture(scope="session")
def temp_dir():
    """Create a temporary directory for the entire test session."""
    temp_path = tempfile.mkdtemp()
    yield Path(temp_path)
    shutil.rmtree(temp_path)


@pytest.fixture
def sample_text_data():
    """Sample text dataset for testing."""
    return {
        "queries": [
            {"id": "q1", "text": "What is machine learning?"},
            {"id": "q2", "text": "How does neural network work?"},
            {"id": "q3", "text": "What is deep learning?"}
        ],
        "corpus": [
            {"id": "d1", "text": "Machine learning is a subset of artificial intelligence."},
            {"id": "d2", "text": "Neural networks are computing systems inspired by biological neural networks."},
            {"id": "d3", "text": "Deep learning uses multiple layers to progressively extract features."},
            {"id": "d4", "text": "Artificial intelligence mimics human intelligence in machines."}
        ],
        "qrels": [
            {"query_id": "q1", "doc_id": "d1", "relevance": 1},
            {"query_id": "q1", "doc_id": "d4", "relevance": 1},
            {"query_id": "q2", "doc_id": "d2", "relevance": 1},
            {"query_id": "q3", "doc_id": "d3", "relevance": 1}
        ]
    }


@pytest.fixture
def sample_image_data():
    """Sample image dataset for testing."""
    return {
        "queries": [
            {"id": "q1", "text": "Find document about contract terms"},
            {"id": "q2", "text": "Search for invoice details"}
        ],
        "images": [
            {"id": "img1", "path": "/fake/path/contract.jpg", "text": "Contract terms and conditions"},
            {"id": "img2", "path": "/fake/path/invoice.jpg", "text": "Invoice number: 12345"}
        ],
        "qrels": [
            {"query_id": "q1", "doc_id": "img1", "relevance": 1},
            {"query_id": "q2", "doc_id": "img2", "relevance": 1}
        ]
    }


@pytest.fixture
def mock_model():
    """Mock model for testing."""
    model = Mock()
    model.predict.return_value = {
        "q1": [("d1", 0.9), ("d4", 0.7), ("d2", 0.3)],
        "q2": [("d2", 0.8), ("d3", 0.6), ("d1", 0.2)],
        "q3": [("d3", 0.9), ("d1", 0.5), ("d2", 0.4)]
    }
    return model


@pytest.fixture
def mock_embeddings():
    """Mock embeddings for testing."""
    return {
        "queries": np.random.rand(3, 768),  # 3 queries, 768-dim embeddings
        "documents": np.random.rand(4, 768)  # 4 documents, 768-dim embeddings
    }


@pytest.fixture
def sample_config():
    """Sample experiment configuration."""
    return {
        "name": "test_experiment",
        "dataset": {
            "type": "text",
            "path": "/fake/dataset/path"
        },
        "models": [
            {"type": "bm25", "params": {"k1": 1.2, "b": 0.75}},
            {"type": "dense", "params": {"model_name": "all-MiniLM-L6-v2"}}
        ],
        "evaluation": {
            "metrics": ["ndcg@10", "recall@5", "map"],
            "k_values": [5, 10]
        },
        "output": {
            "save_results": True,
            "format": "json"
        }
    }


@pytest.fixture
def sample_results():
    """Sample evaluation results."""
    return {
        "bm25": {
            "ndcg@10": 0.75,
            "recall@5": 0.60,
            "map": 0.65
        },
        "dense": {
            "ndcg@10": 0.80,
            "recall@5": 0.65,
            "map": 0.70
        }
    }


@pytest.fixture(autouse=True)
def setup_test_environment(monkeypatch):
    """Setup test environment variables."""
    monkeypatch.setenv("TESTING", "1")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")  # Disable GPU for tests


def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line("markers", "unit: Unit tests")
    config.addinivalue_line("markers", "integration: Integration tests")
    config.addinivalue_line("markers", "e2e: End-to-end tests")
    config.addinivalue_line("markers", "performance: Performance tests")
    config.addinivalue_line("markers", "slow: Slow tests")
    config.addinivalue_line("markers", "external: Tests requiring external resources")
    config.addinivalue_line("markers", "gpu: Tests requiring GPU")


def pytest_collection_modifyitems(config, items):
    """Modify test collection to add markers based on file paths."""
    for item in items:
        # Add markers based on test file path
        if "unit" in str(item.fspath):
            item.add_marker(pytest.mark.unit)
        elif "integration" in str(item.fspath):
            item.add_marker(pytest.mark.integration)
        elif "e2e" in str(item.fspath):
            item.add_marker(pytest.mark.e2e)
        elif "performance" in str(item.fspath):
            item.add_marker(pytest.mark.performance)
