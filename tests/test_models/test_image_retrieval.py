"""
Unit tests for image retrieval models in NewAIBench framework.

This module tests:
1. OCRBasedDocumentRetriever - OCR + dense text retrieval
2. ImageEmbeddingDocumentRetriever - CLIP-based image retrieval
3. Integration with base framework components
"""

import pytest
import numpy as np
import tempfile
import os
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from PIL import Image
import json

# Import the classes to test
from newaibench.models.image_retrieval import (
    OCRBasedDocumentRetriever,
    ImageEmbeddingDocumentRetriever,
    MultimodalDocumentRetriever
)
from newaibench.models.base import ModelType


class TestOCRBasedDocumentRetriever:
    """Test suite for OCRBasedDocumentRetriever."""
    
    @pytest.fixture
    def sample_config(self):
        """Sample configuration for OCR-based retriever."""
        return {
            "model_name": "sentence-transformers/all-MiniLM-L6-v2",
            "model_type": ModelType.DENSE_TEXT,
            "embedding_dim": 384,
            "max_seq_length": 512,
            "batch_size": 32,
            "ocr_engine": "tesseract",
            "ocr_config": {"lang": "eng"},
            "enable_preprocessing": True,
            "normalize_embeddings": True,
            "enable_ann_indexing": True,
            "ann_index_type": "faiss"
        }
    
    @pytest.fixture
    def sample_corpus(self):
        """Sample corpus with OCR text data."""
        return [
            {
                "id": "doc1",
                "image_path": "/path/to/image1.jpg",
                "ocr_text": "This is a sample document with important information about AI.",
                "title": "AI Document 1"
            },
            {
                "id": "doc2", 
                "image_path": "/path/to/image2.jpg",
                "ocr_text": "Machine learning algorithms are powerful tools for data analysis.",
                "title": "ML Document 2"
            },
            {
                "id": "doc3",
                "image_path": "/path/to/image3.jpg", 
                "ocr_text": "Natural language processing enables computers to understand text.",
                "title": "NLP Document 3"
            }
        ]
    
    def test_init(self, sample_config):
        """Test OCRBasedDocumentRetriever initialization."""
        retriever = OCRBasedDocumentRetriever(sample_config)
        
        assert retriever.model_config == sample_config
        assert retriever.ocr_engine == "tesseract"
        assert retriever.ocr_config == {"lang": "eng"}
        assert retriever.enable_preprocessing is True
        
    def test_init_with_defaults(self):
        """Test initialization with minimal config."""
        config = {
            "model_name": "sentence-transformers/all-MiniLM-L6-v2",
            "model_type": ModelType.DENSE_TEXT
        }
        retriever = OCRBasedDocumentRetriever(config)
        
        assert retriever.ocr_engine == "tesseract"
        assert retriever.ocr_config == {"lang": "eng"}
        assert retriever.enable_preprocessing is True
    
    @patch('newaibench.models.image_retrieval.SentenceTransformer')
    def test_load_model(self, mock_sbert, sample_config):
        """Test model loading."""
        mock_model = Mock()
        mock_sbert.return_value = mock_model
        
        retriever = OCRBasedDocumentRetriever(sample_config)
        retriever.load_model()
        
        mock_sbert.assert_called_once_with(sample_config["model_name"])
        assert retriever.model == mock_model
    
    @patch('pytesseract.image_to_string')
    @patch('PIL.Image.open')
    def test_extract_ocr_text(self, mock_image_open, mock_ocr, sample_config):
        """Test OCR text extraction."""
        # Setup mocks
        mock_image = Mock()
        mock_image_open.return_value = mock_image
        mock_ocr.return_value = "Extracted text from image"
        
        retriever = OCRBasedDocumentRetriever(sample_config)
        
        result = retriever._extract_ocr_text("/path/to/image.jpg")
        
        assert result == "Extracted text from image"
        mock_image_open.assert_called_once_with("/path/to/image.jpg")
        mock_ocr.assert_called_once_with(mock_image, lang="eng")
    
    def test_prepare_corpus_text(self, sample_config, sample_corpus):
        """Test corpus text preparation."""
        retriever = OCRBasedDocumentRetriever(sample_config)
        
        texts = retriever._prepare_corpus_text(sample_corpus)
        
        expected_texts = [
            "This is a sample document with important information about AI.",
            "Machine learning algorithms are powerful tools for data analysis.", 
            "Natural language processing enables computers to understand text."
        ]
        
        assert texts == expected_texts
    
    @patch('newaibench.models.image_retrieval.SentenceTransformer')
    def test_predict_with_ocr_text(self, mock_sbert, sample_config, sample_corpus):
        """Test prediction with existing OCR text."""
        # Setup mock model
        mock_model = Mock()
        mock_model.encode.side_effect = [
            np.array([[0.1, 0.2, 0.3]]),  # Query embedding
            np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]])  # Corpus embeddings
        ]
        mock_sbert.return_value = mock_model
        
        retriever = OCRBasedDocumentRetriever(sample_config)
        retriever.load_model()
        
        queries = ["AI information"]
        results = retriever.predict(queries, sample_corpus, top_k=2)
        
        assert len(results) == 1
        assert len(results[0]) == 2
        assert all("score" in result for result in results[0])
        assert all("id" in result for result in results[0])
    
    def test_extract_ocr_text_file_not_found(self, sample_config):
        """Test OCR extraction with non-existent file."""
        retriever = OCRBasedDocumentRetriever(sample_config)
        
        result = retriever._extract_ocr_text("/nonexistent/path.jpg")
        
        assert result == ""


class TestImageEmbeddingDocumentRetriever:
    """Test suite for ImageEmbeddingDocumentRetriever."""
    
    @pytest.fixture
    def sample_config(self):
        """Sample configuration for image embedding retriever."""
        return {
            "model_name": "openai/clip-vit-base-patch32",
            "model_type": ModelType.MULTIMODAL,
            "embedding_dim": 512,
            "batch_size": 16,
            "normalize_embeddings": True,
            "enable_ann_indexing": True,
            "ann_index_type": "faiss",
            "image_preprocessing": {
                "resize": [224, 224],
                "normalize": True
            }
        }
    
    @pytest.fixture
    def trust_remote_code_config(self):
        """Sample configuration with trust_remote_code enabled."""
        return {
            "model_name": "custom/clip-model-with-remote-code",
            "model_type": ModelType.MULTIMODAL,
            "embedding_dim": 512,
            "batch_size": 16,
            "normalize_embeddings": True,
            "enable_ann_indexing": True,
            "ann_index_type": "faiss",
            "trust_remote_code": True,
            "image_preprocessing": {
                "resize": [224, 224],
                "normalize": True
            }
        }
    
    @pytest.fixture
    def sample_corpus(self):
        """Sample corpus with image paths."""
        return [
            {
                "id": "img1",
                "image_path": "/path/to/image1.jpg",
                "title": "Sample Image 1"
            },
            {
                "id": "img2",
                "image_path": "/path/to/image2.jpg", 
                "title": "Sample Image 2"
            },
            {
                "id": "img3",
                "image_path": "/path/to/image3.jpg",
                "title": "Sample Image 3"
            }
        ]
    
    def test_init(self, sample_config):
        """Test ImageEmbeddingDocumentRetriever initialization."""
        retriever = ImageEmbeddingDocumentRetriever(sample_config)
        
        assert retriever.model_config == sample_config
        assert retriever.image_preprocessing == {
            "resize": [224, 224],
            "normalize": True
        }
    
    def test_init_with_defaults(self):
        """Test initialization with minimal config."""
        config = {
            "model_name": "openai/clip-vit-base-patch32",
            "model_type": ModelType.MULTIMODAL
        }
        retriever = ImageEmbeddingDocumentRetriever(config)
        
        assert retriever.image_preprocessing == {"resize": [224, 224]}
    
    @patch('newaibench.models.image_retrieval.CLIPModel')
    @patch('newaibench.models.image_retrieval.CLIPProcessor')
    def test_load_model(self, mock_processor, mock_model, sample_config):
        """Test CLIP model loading."""
        mock_clip_model = Mock()
        mock_clip_processor = Mock()
        mock_model.from_pretrained.return_value = mock_clip_model
        mock_processor.from_pretrained.return_value = mock_clip_processor
        
        retriever = ImageEmbeddingDocumentRetriever(sample_config)
        retriever.load_model()
        
        mock_model.from_pretrained.assert_called_once_with(sample_config["model_name"])
        mock_processor.from_pretrained.assert_called_once_with(sample_config["model_name"])
        assert retriever.model == mock_clip_model
        assert retriever.processor == mock_clip_processor
    
    @patch('PIL.Image.open')
    def test_load_and_preprocess_image(self, mock_image_open, sample_config):
        """Test image loading and preprocessing."""
        # Create a mock image
        mock_image = Mock()
        mock_image.mode = "RGB"
        mock_image_open.return_value = mock_image
        
        retriever = ImageEmbeddingDocumentRetriever(sample_config)
        
        result = retriever._load_and_preprocess_image("/path/to/image.jpg")
        
        assert result == mock_image
        mock_image_open.assert_called_once_with("/path/to/image.jpg")
    
    def test_load_and_preprocess_image_convert_rgba(self, sample_config):
        """Test image conversion from RGBA to RGB."""
        with patch('PIL.Image.open') as mock_image_open:
            mock_image = Mock()
            mock_image.mode = "RGBA"
            mock_converted = Mock()
            mock_image.convert.return_value = mock_converted
            mock_image_open.return_value = mock_image
            
            retriever = ImageEmbeddingDocumentRetriever(sample_config)
            result = retriever._load_and_preprocess_image("/path/to/image.jpg")
            
            mock_image.convert.assert_called_once_with("RGB")
            assert result == mock_converted
    
    def test_load_and_preprocess_image_file_not_found(self, sample_config):
        """Test image loading with non-existent file."""
        retriever = ImageEmbeddingDocumentRetriever(sample_config)
        
        result = retriever._load_and_preprocess_image("/nonexistent/path.jpg")
        
        assert result is None
    
    @patch('newaibench.models.image_retrieval.CLIPModel')
    @patch('newaibench.models.image_retrieval.CLIPProcessor')
    @patch('PIL.Image.open')
    def test_predict(self, mock_image_open, mock_processor_class, mock_model_class, sample_config, sample_corpus):
        """Test prediction with text queries and image corpus."""
        # Setup mocks
        mock_image = Mock()
        mock_image.mode = "RGB"
        mock_image_open.return_value = mock_image
        
        mock_model = Mock()
        mock_processor = Mock()
        mock_model_class.from_pretrained.return_value = mock_model
        mock_processor_class.from_pretrained.return_value = mock_processor
        
        # Mock processor outputs
        mock_processor.return_value = {
            "input_ids": Mock(),
            "attention_mask": Mock()
        }
        
        # Mock model outputs
        mock_text_features = Mock()
        mock_text_features.cpu.return_value.numpy.return_value = np.array([[0.1, 0.2, 0.3]])
        
        mock_image_features = Mock()
        mock_image_features.cpu.return_value.numpy.return_value = np.array([
            [0.1, 0.2, 0.3],
            [0.4, 0.5, 0.6], 
            [0.7, 0.8, 0.9]
        ])
        
        mock_model.get_text_features.return_value = mock_text_features
        mock_model.get_image_features.return_value = mock_image_features
        
        retriever = ImageEmbeddingDocumentRetriever(sample_config)
        retriever.load_model()
        
        queries = ["sample query"]
        results = retriever.predict(queries, sample_corpus, top_k=2)
        
        assert len(results) == 1
        assert len(results[0]) == 2
        assert all("score" in result for result in results[0])
        assert all("id" in result for result in results[0])
    
    def test_init_trust_remote_code(self, trust_remote_code_config):
        """Test initialization with trust_remote_code enabled."""
        retriever = ImageEmbeddingDocumentRetriever(trust_remote_code_config)
        
        assert retriever.model_config == trust_remote_code_config
        assert retriever.trust_remote_code is True
        assert retriever.image_preprocessing == {
            "resize": [224, 224],
            "normalize": True
        }
    
    def test_init_trust_remote_code_default(self, sample_config):
        """Test that trust_remote_code defaults to False."""
        retriever = ImageEmbeddingDocumentRetriever(sample_config)
        
        assert retriever.trust_remote_code is False
    
    @patch('newaibench.models.image_retrieval.SentenceTransformer')
    def test_load_model_with_trust_remote_code(self, mock_sbert, trust_remote_code_config):
        """Test SentenceTransformer loading with trust_remote_code=True."""
        # Setup mock
        mock_model = Mock()
        mock_model.get_sentence_embedding_dimension.return_value = 768
        mock_sbert.return_value = mock_model
        
        retriever = ImageEmbeddingDocumentRetriever(trust_remote_code_config)
        retriever.load_model()
        
        assert retriever.is_loaded is True
        # Verify trust_remote_code is passed to SentenceTransformer
        mock_sbert.assert_called_once_with(
            "custom/clip-model-with-remote-code", 
            trust_remote_code=True
        )


class TestMultimodalDocumentRetriever:
    """Test suite for MultimodalDocumentRetriever (placeholder)."""
    
    def test_init_raises_not_implemented(self):
        """Test that MultimodalDocumentRetriever raises NotImplementedError."""
        config = {"model_name": "test"}
        
        with pytest.raises(NotImplementedError):
            MultimodalDocumentRetriever(config)


class TestImageRetrievalIntegration:
    """Integration tests for image retrieval models."""
    
    def test_ocr_retriever_inheritance(self):
        """Test that OCRBasedDocumentRetriever properly inherits from DenseTextRetriever."""
        from newaibench.models.dense import DenseTextRetriever
        
        config = {
            "model_name": "sentence-transformers/all-MiniLM-L6-v2",
            "model_type": ModelType.DENSE_TEXT
        }
        
        retriever = OCRBasedDocumentRetriever(config)
        assert isinstance(retriever, DenseTextRetriever)
    
    def test_image_retriever_inheritance(self):
        """Test that ImageEmbeddingDocumentRetriever properly inherits from BaseRetrievalModel."""
        from newaibench.models.base import BaseRetrievalModel
        
        config = {
            "model_name": "openai/clip-vit-base-patch32",
            "model_type": ModelType.MULTIMODAL
        }
        
        retriever = ImageEmbeddingDocumentRetriever(config)
        assert isinstance(retriever, BaseRetrievalModel)
    
    @pytest.mark.parametrize("ann_index_type", ["faiss", "hnswlib"])
    def test_ann_indexing_support(self, ann_index_type):
        """Test ANN indexing support for both models."""
        config = {
            "model_name": "sentence-transformers/all-MiniLM-L6-v2",
            "model_type": ModelType.DENSE_TEXT,
            "enable_ann_indexing": True,
            "ann_index_type": ann_index_type
        }
        
        retriever = OCRBasedDocumentRetriever(config)
        assert retriever.enable_ann_indexing is True
        assert retriever.ann_index_type == ann_index_type


if __name__ == "__main__":
    pytest.main([__file__])
