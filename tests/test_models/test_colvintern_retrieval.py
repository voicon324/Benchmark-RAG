"""
Test suite for ColVintern document retrieval model.

This module provides comprehensive tests for ColVinternDocumentRetriever
including model loading, encoding, and retrieval functionality.
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, List, Any

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from newaibench.models.colvintern_retrieval import ColVinternDocumentRetriever
from newaibench.models.base import ModelType


class TestColVinternDocumentRetriever:
    """Test suite for ColVinternDocumentRetriever."""
    
    @pytest.fixture
    def sample_config(self):
        """Sample configuration for testing."""
        return {
            "name": "test_colvintern",
            "type": ModelType.MULTIMODAL,
            "model_name_or_path": "5CD-AI/ColVintern-1B-v1",
            "parameters": {
                "batch_size_images": 4,
                "batch_size_text": 16,
                "normalize_embeddings": True,
                "use_ann_index": False,
                "image_path_field": "image_path"
            },
            "device": "cpu"
        }
    
    @pytest.fixture
    def sample_corpus(self):
        """Sample corpus for testing."""
        return {
            "doc1": {
                "text": "Hóa đơn thanh toán dịch vụ",
                "image_path": "/test/images/invoice_01.jpg"
            },
            "doc2": {
                "text": "Giấy chứng nhận hoàn thành",
                "image_path": "/test/images/certificate_02.jpg"
            },
            "doc3": {
                "text": "Hợp đồng lao động",
                "image_path": "/test/images/contract_03.jpg"
            }
        }
    
    @pytest.fixture
    def sample_queries(self):
        """Sample queries for testing."""
        return [
            {"query_id": "q1", "text": "hóa đơn thanh toán"},
            {"query_id": "q2", "text": "giấy chứng nhận"},
            {"query_id": "q3", "text": "hợp đồng"}
        ]
    
    def test_init_success(self, sample_config):
        """Test successful initialization."""
        with patch('newaibench.models.colvintern_retrieval.COLVINTERN_AVAILABLE', True):
            retriever = ColVinternDocumentRetriever(sample_config)
            
            assert retriever.name == "test_colvintern"
            assert retriever.model_name_or_path == "5CD-AI/ColVintern-1B-v1"
            assert retriever.batch_size_images == 4
            assert retriever.batch_size_text == 16
            assert retriever.normalize_embeddings is True
            assert retriever.use_ann_index is False
    
    def test_init_missing_dependencies(self, sample_config):
        """Test initialization with missing dependencies."""
        with patch('newaibench.models.colvintern_retrieval.COLVINTERN_AVAILABLE', False):
            with pytest.raises(ImportError, match="ColVintern dependencies"):
                ColVinternDocumentRetriever(sample_config)
    
    @patch('newaibench.models.colvintern_retrieval.AutoModel')
    @patch('newaibench.models.colvintern_retrieval.AutoProcessor')
    @patch('newaibench.models.colvintern_retrieval.torch.cuda.is_available')
    def test_load_model_cpu(self, mock_cuda, mock_processor_class, mock_model_class, sample_config):
        """Test model loading on CPU."""
        # Setup mocks
        mock_cuda.return_value = False
        mock_processor = Mock()
        mock_model = Mock()
        mock_model.config.hidden_size = 768
        
        mock_processor_class.from_pretrained.return_value = mock_processor
        mock_model_class.from_pretrained.return_value = mock_model
        
        with patch('newaibench.models.colvintern_retrieval.COLVINTERN_AVAILABLE', True):
            retriever = ColVinternDocumentRetriever(sample_config)
            retriever.load_model()
            
            # Verify model loading
            mock_processor_class.from_pretrained.assert_called_once_with(
                "5CD-AI/ColVintern-1B-v1", trust_remote_code=True
            )
            mock_model_class.from_pretrained.assert_called_once_with(
                "5CD-AI/ColVintern-1B-v1",
                torch_dtype=patch.ANY,
                low_cpu_mem_usage=True,
                trust_remote_code=True
            )
            
            assert retriever.is_loaded is True
            assert retriever.embedding_dim == 768
    
    @patch('newaibench.models.colvintern_retrieval.AutoModel')
    @patch('newaibench.models.colvintern_retrieval.AutoProcessor')
    @patch('newaibench.models.colvintern_retrieval.torch.cuda.is_available')
    def test_load_model_cuda(self, mock_cuda, mock_processor_class, mock_model_class, sample_config):
        """Test model loading on CUDA."""
        # Setup mocks
        mock_cuda.return_value = True
        mock_processor = Mock()
        mock_model = Mock()
        mock_model.config.hidden_size = 768
        mock_model.cuda.return_value = mock_model
        
        mock_processor_class.from_pretrained.return_value = mock_processor
        mock_model_class.from_pretrained.return_value = mock_model
        
        sample_config["device"] = "cuda"
        
        with patch('newaibench.models.colvintern_retrieval.COLVINTERN_AVAILABLE', True):
            retriever = ColVinternDocumentRetriever(sample_config)
            retriever.load_model()
            
            # Verify CUDA usage
            mock_model.cuda.assert_called_once()
            assert retriever.is_loaded is True
    
    def test_load_and_preprocess_image_success(self, sample_config):
        """Test successful image loading and preprocessing."""
        with patch('newaibench.models.colvintern_retrieval.COLVINTERN_AVAILABLE', True):
            retriever = ColVinternDocumentRetriever(sample_config)
            
            # Mock PIL Image
            mock_image = Mock()
            mock_image.mode = 'RGB'
            
            with patch('newaibench.models.colvintern_retrieval.Path') as mock_path, \
                 patch('newaibench.models.colvintern_retrieval.Image.open') as mock_open:
                
                # Setup path mock
                mock_path_instance = Mock()
                mock_path_instance.exists.return_value = True
                mock_path_instance.suffix.lower.return_value = '.jpg'
                mock_path_instance.stat.return_value.st_size = 1024 * 1024  # 1MB
                mock_path.return_value = mock_path_instance
                
                # Setup image mock
                mock_open.return_value = mock_image
                
                result = retriever._load_and_preprocess_image("/test/image.jpg")
                
                assert result == mock_image
                mock_open.assert_called_once_with(mock_path_instance)
    
    def test_load_and_preprocess_image_not_found(self, sample_config):
        """Test image loading when file not found."""
        with patch('newaibench.models.colvintern_retrieval.COLVINTERN_AVAILABLE', True):
            retriever = ColVinternDocumentRetriever(sample_config)
            
            with patch('newaibench.models.colvintern_retrieval.Path') as mock_path:
                mock_path_instance = Mock()
                mock_path_instance.exists.return_value = False
                mock_path.return_value = mock_path_instance
                
                result = retriever._load_and_preprocess_image("/nonexistent/image.jpg")
                
                assert result is None
    
    def test_load_and_preprocess_image_unsupported_format(self, sample_config):
        """Test image loading with unsupported format."""
        with patch('newaibench.models.colvintern_retrieval.COLVINTERN_AVAILABLE', True):
            retriever = ColVinternDocumentRetriever(sample_config)
            
            with patch('newaibench.models.colvintern_retrieval.Path') as mock_path:
                mock_path_instance = Mock()
                mock_path_instance.exists.return_value = True
                mock_path_instance.suffix.lower.return_value = '.xyz'
                mock_path.return_value = mock_path_instance
                
                result = retriever._load_and_preprocess_image("/test/image.xyz")
                
                assert result is None
    
    def test_encode_queries_success(self, sample_config, sample_queries):
        """Test successful query encoding."""
        with patch('newaibench.models.colvintern_retrieval.COLVINTERN_AVAILABLE', True):
            retriever = ColVinternDocumentRetriever(sample_config)
            retriever.is_loaded = True
            
            # Mock processor and model
            mock_processor = Mock()
            mock_model = Mock()
            mock_model.device = "cpu"
            
            # Mock processor output
            mock_processed = {
                "input_ids": Mock(),
                "attention_mask": Mock()
            }
            mock_processed["input_ids"].to.return_value = mock_processed["input_ids"]
            mock_processed["attention_mask"].to.return_value = mock_processed["attention_mask"]
            mock_processor.process_queries.return_value = mock_processed
            
            # Mock model output
            mock_embeddings = Mock()
            mock_embeddings.cpu.return_value.numpy.return_value = np.random.rand(3, 768)
            mock_model.return_value = mock_embeddings
            
            retriever.processor = mock_processor
            retriever.model = mock_model
            
            with patch('newaibench.models.colvintern_retrieval.torch.no_grad'):
                result = retriever.encode_queries(sample_queries)
                
                assert len(result) == 3
                assert "q1" in result
                assert "q2" in result
                assert "q3" in result
                assert isinstance(result["q1"], np.ndarray)
    
    def test_encode_documents_success(self, sample_config, sample_corpus):
        """Test successful document encoding."""
        with patch('newaibench.models.colvintern_retrieval.COLVINTERN_AVAILABLE', True):
            retriever = ColVinternDocumentRetriever(sample_config)
            retriever.is_loaded = True
            
            # Mock processor and model
            mock_processor = Mock()
            mock_model = Mock()
            mock_model.device = "cpu"
            
            # Mock image loading
            mock_image = Mock()
            with patch.object(retriever, '_load_and_preprocess_image', return_value=mock_image):
                
                # Mock processor output
                mock_processed = {
                    "pixel_values": Mock(),
                    "input_ids": Mock(),
                    "attention_mask": Mock()
                }
                for key in mock_processed:
                    mock_processed[key].to.return_value = mock_processed[key]
                mock_processor.process_images.return_value = mock_processed
                
                # Mock model output
                mock_embeddings = Mock()
                mock_embeddings.cpu.return_value.numpy.return_value = np.random.rand(3, 768)
                mock_model.return_value = mock_embeddings
                
                retriever.processor = mock_processor
                retriever.model = mock_model
                
                with patch('newaibench.models.colvintern_retrieval.torch.no_grad'):
                    result = retriever.encode_documents(sample_corpus)
                    
                    assert len(result) == 3
                    assert "doc1" in result
                    assert "doc2" in result
                    assert "doc3" in result
                    assert isinstance(result["doc1"], np.ndarray)
    
    def test_predict_success(self, sample_config, sample_corpus, sample_queries):
        """Test successful prediction."""
        with patch('newaibench.models.colvintern_retrieval.COLVINTERN_AVAILABLE', True):
            retriever = ColVinternDocumentRetriever(sample_config)
            retriever.is_loaded = True
            retriever._corpus_indexed = True
            
            # Mock embeddings
            retriever.doc_embeddings = {
                "doc1": np.random.rand(768),
                "doc2": np.random.rand(768),
                "doc3": np.random.rand(768)
            }
            retriever.doc_ids_list = ["doc1", "doc2", "doc3"]
            
            # Mock encode_queries
            query_embeddings = {
                "q1": np.random.rand(768),
                "q2": np.random.rand(768),
                "q3": np.random.rand(768)
            }
            
            with patch.object(retriever, 'encode_queries', return_value=query_embeddings):
                result = retriever.predict(sample_queries, sample_corpus, top_k=5)
                
                assert len(result) == 3
                assert "q1" in result
                assert "q2" in result
                assert "q3" in result
                assert isinstance(result["q1"], dict)
    
    def test_get_model_info(self, sample_config):
        """Test model info retrieval."""
        with patch('newaibench.models.colvintern_retrieval.COLVINTERN_AVAILABLE', True):
            retriever = ColVinternDocumentRetriever(sample_config)
            retriever.embedding_dim = 768
            
            info = retriever.get_model_info()
            
            assert info["model_name"] == "test_colvintern"
            assert info["model_type"] == "Multimodal"
            assert info["architecture"] == "ColVintern"
            assert info["embedding_dim"] == 768
            assert info["batch_size_images"] == 4
            assert info["batch_size_text"] == 16
    
    def test_score_multi_vector_success(self, sample_config):
        """Test multi-vector scoring."""
        with patch('newaibench.models.colvintern_retrieval.COLVINTERN_AVAILABLE', True):
            retriever = ColVinternDocumentRetriever(sample_config)
            
            # Mock processor with score_multi_vector method
            mock_processor = Mock()
            mock_scores = Mock()
            mock_processor.score_multi_vector.return_value = mock_scores
            retriever.processor = mock_processor
            
            # Mock tensors
            query_embeddings = Mock()
            image_embeddings = Mock()
            
            result = retriever.score_multi_vector(query_embeddings, image_embeddings)
            
            mock_processor.score_multi_vector.assert_called_once_with(
                query_embeddings, image_embeddings
            )
            assert result == mock_scores
    
    def test_score_multi_vector_fallback(self, sample_config):
        """Test multi-vector scoring fallback."""
        with patch('newaibench.models.colvintern_retrieval.COLVINTERN_AVAILABLE', True):
            retriever = ColVinternDocumentRetriever(sample_config)
            
            # Mock processor that raises exception
            mock_processor = Mock()
            mock_processor.score_multi_vector.side_effect = Exception("Method not available")
            retriever.processor = mock_processor
            
            # Mock tensors and torch.matmul
            query_embeddings = Mock()
            image_embeddings = Mock()
            mock_result = Mock()
            
            with patch('newaibench.models.colvintern_retrieval.torch.matmul', return_value=mock_result):
                result = retriever.score_multi_vector(query_embeddings, image_embeddings)
                
                assert result == mock_result


class TestColVinternIntegration:
    """Integration tests for ColVintern model."""
    
    def test_colvintern_inheritance(self):
        """Test that ColVinternDocumentRetriever properly inherits from BaseRetrievalModel."""
        from newaibench.models.base import BaseRetrievalModel
        
        config = {
            "name": "test_colvintern",
            "type": ModelType.MULTIMODAL,
            "model_name_or_path": "5CD-AI/ColVintern-1B-v1"
        }
        
        with patch('newaibench.models.colvintern_retrieval.COLVINTERN_AVAILABLE', True):
            retriever = ColVinternDocumentRetriever(config)
            assert isinstance(retriever, BaseRetrievalModel)
    
    def test_colvintern_model_type(self):
        """Test that ColVintern model has correct type."""
        config = {
            "name": "test_colvintern",
            "type": ModelType.MULTIMODAL,
            "model_name_or_path": "5CD-AI/ColVintern-1B-v1"
        }
        
        with patch('newaibench.models.colvintern_retrieval.COLVINTERN_AVAILABLE', True):
            retriever = ColVinternDocumentRetriever(config)
            assert retriever.model_type == ModelType.MULTIMODAL
    
    def test_colvintern_in_models_init(self):
        """Test that ColVintern model is exported in __init__.py."""
        try:
            from newaibench.models import ColVinternDocumentRetriever
            assert ColVinternDocumentRetriever is not None
        except ImportError:
            pytest.fail("ColVinternDocumentRetriever not found in models __init__.py")
