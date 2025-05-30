"""
Base classes for retrieval models in NewAIBench framework.

This module provides abstract base classes that define the standard interface
for all retrieval models. It enables a unified API for different types of
retrieval models including sparse, dense, vision-based, and multimodal models.
"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, List, Any, Optional
import numpy as np

logger = logging.getLogger(__name__)

# Helper function to count model parameters
def count_model_parameters(model_obj) -> Optional[int]:
    """
    Count the number of parameters in a model.
    
    Args:
        model_obj: The model object (can be PyTorch model, transformers model, etc.)
        
    Returns:
        Number of parameters if countable, None otherwise
    """
    try:
        # For PyTorch models
        if hasattr(model_obj, 'parameters'):
            return sum(p.numel() for p in model_obj.parameters())
        
        # For sentence-transformers models
        if hasattr(model_obj, '_modules') and hasattr(model_obj, 'parameters'):
            return sum(p.numel() for p in model_obj.parameters())
            
        # For transformers models with specific method
        if hasattr(model_obj, 'num_parameters'):
            return model_obj.num_parameters()
            
        # For models that have a config with num_parameters
        if hasattr(model_obj, 'config') and hasattr(model_obj.config, 'num_parameters'):
            return model_obj.config.num_parameters
            
        return None
    except Exception as e:
        logger.warning(f"Failed to count model parameters: {e}")
        return None

def get_model_parameter_count(retrieval_model) -> Optional[int]:
    """
    Get parameter count for a retrieval model by checking different model storage patterns.
    
    Args:
        retrieval_model: A retrieval model instance (DenseTextRetriever, BM25Model, etc.)
        
    Returns:
        Number of parameters if countable, None otherwise
    """
    try:
        # For DenseTextRetriever models
        if hasattr(retrieval_model, 'encoder_model') and retrieval_model.encoder_model is not None:
            return count_model_parameters(retrieval_model.encoder_model)
        
        # For DPR dual-encoder models
        if hasattr(retrieval_model, 'query_encoder') and hasattr(retrieval_model, 'doc_encoder'):
            query_params = count_model_parameters(retrieval_model.query_encoder) or 0
            doc_params = count_model_parameters(retrieval_model.doc_encoder) or 0
            if query_params > 0 or doc_params > 0:
                return query_params + doc_params
        
        # For ColVintern/ColPali models
        if hasattr(retrieval_model, 'model') and retrieval_model.model is not None:
            return count_model_parameters(retrieval_model.model)
        
        # For models that store the model in a different attribute
        if hasattr(retrieval_model, 'processor') and retrieval_model.processor is not None:
            return count_model_parameters(retrieval_model.processor)
        
        # For sparse models like BM25 (no neural parameters)
        if hasattr(retrieval_model, 'model_type') and 'BM25' in str(retrieval_model.model_type):
            return 0  # BM25 has no trainable parameters
        
        # Try direct parameter counting on the retrieval model itself
        return count_model_parameters(retrieval_model)
        
    except Exception as e:
        logger.warning(f"Failed to get parameter count for retrieval model: {e}")
        return None

# Helper function to count model parameters
def count_model_parameters(model) -> Optional[int]:
    """
    Count the number of parameters in a model.
    
    Args:
        model: The model object (can be PyTorch model, transformers model, etc.)
        
    Returns:
        Number of parameters if countable, None otherwise
    """
    try:
        # For PyTorch models
        if hasattr(model, 'parameters'):
            return sum(p.numel() for p in model.parameters())
        
        # For sentence-transformers models
        if hasattr(model, '_modules') and hasattr(model, 'parameters'):
            return sum(p.numel() for p in model.parameters())
            
        # For transformers models with specific method
        if hasattr(model, 'num_parameters'):
            return model.num_parameters()
            
        # For models that have a config with num_parameters
        if hasattr(model, 'config') and hasattr(model.config, 'num_parameters'):
            return model.config.num_parameters
            
        return None
    except Exception as e:
        logger.warning(f"Failed to count model parameters: {e}")
        return None


class ModelType(Enum):
    """Định nghĩa các loại mô hình retrieval được hỗ trợ."""
    SPARSE = "sparse"           # BM25, TF-IDF, traditional sparse models
    DENSE = "dense"             # BERT, DPR, ColBERT, sentence transformers
    VISION = "vision"           # CLIP, Vision models, image-based retrieval
    MULTIMODAL = "multimodal"   # Hybrid text+image models
    CUSTOM = "custom"           # Custom implementations


@dataclass
class ModelConfig:
    """
    Cấu hình cho retrieval model.
    
    Attributes:
        name: Tên model (unique identifier)
        type: Loại model (sparse, dense, vision, multimodal, custom)
        checkpoint_path: Đường dẫn tới model checkpoint hoặc pretrained model
        parameters: Dictionary các tham số đặc thù cho model
        device: Device để chạy model ("cpu", "cuda", "auto")
        batch_size: Batch size cho inference
        max_length: Max token length cho text processing
        cache_dir: Directory để cache embeddings và intermediate results
    """
    name: str                                    
    type: ModelType                             
    checkpoint_path: Optional[str] = None       
    parameters: Dict[str, Any] = field(default_factory=dict)  
    device: str = "cpu"                         
    batch_size: int = 32                        
    max_length: int = 512                       
    cache_dir: Optional[str] = None             
    
    def validate(self) -> None:
        """Validate configuration parameters."""
        if not self.name:
            raise ValueError("Model name cannot be empty")
        if self.batch_size <= 0:
            raise ValueError("Batch size must be positive")
        if self.max_length <= 0:
            raise ValueError("Max length must be positive")
        if self.device not in ["cpu", "cuda", "auto"]:
            logger.warning(f"Device '{self.device}' may not be supported. "
                         "Consider using 'cpu', 'cuda', or 'auto'")


class BaseRetrievalModel(ABC):
    """
    Abstract base class cho tất cả retrieval models trong NewAIBench.
    
    Lớp này định nghĩa giao diện chuẩn mà tất cả các mô hình retrieval
    cụ thể phải tuân theo. Nó cung cấp:
    
    1. Giao diện thống nhất cho prediction và encoding
    2. Khả năng xử lý linh hoạt các loại input/output khác nhau
    3. Support cho cả text và image data
    4. Extensibility cho các loại mô hình mới
    
    Các mô hình con cần implement:
    - load_model(): Load model từ checkpoint
    - predict(): Thực hiện retrieval chính
    
    Các mô hình con có thể override:
    - index_corpus(): Tạo index cho corpus (sparse models)
    - encode_queries(): Encode queries thành embeddings (dense models)
    - encode_documents(): Encode documents thành embeddings (dense models)
    """
    
    def __init__(self, model_config: Dict[str, Any], **kwargs):
        """
        Initialize model với configuration.
        
        Args:
            model_config: Dictionary chứa cấu hình model. Có thể là:
                - ModelConfig object được convert thành dict
                - Raw dictionary với các keys cần thiết
            **kwargs: Các tham số bổ sung
            
        Example:
            >>> config = {
            ...     "name": "my_model",
            ...     "type": "dense",
            ...     "checkpoint_path": "path/to/model",
            ...     "parameters": {"dim": 768}
            ... }
            >>> model = ConcreteModel(config)
        """
        # Handle both ModelConfig object and raw dict
        if hasattr(model_config, '__dict__'):
            # ModelConfig object
            self.config = model_config
        else:
            # Raw dictionary - convert to ModelConfig
            model_type_str = model_config.get('type', 'custom')
            model_type = ModelType(model_type_str) if isinstance(model_type_str, str) else model_type_str
            
            self.config = ModelConfig(
                name=model_config.get('name', 'unnamed_model'),
                type=model_type,
                checkpoint_path=model_config.get('checkpoint_path'),
                parameters=model_config.get('parameters', {}),
                device=model_config.get('device', 'cpu'),
                batch_size=model_config.get('batch_size', 32),
                max_length=model_config.get('max_length', 512),
                cache_dir=model_config.get('cache_dir')
            )
        
        # Validate configuration
        self.config.validate()
        
        # Basic attributes
        self.name = self.config.name
        self.model_type = self.config.type
        self.is_loaded = False
        self._corpus_indexed = False
        
        # Store additional kwargs for subclass use
        self._init_kwargs = kwargs
        
        logger.info(f"Initialized {self.__class__.__name__} model: {self.name}")
    
    @abstractmethod
    def load_model(self) -> None:
        """
        Load model từ checkpoint hoặc initialize model mới.
        
        Method này phải được implement bởi mọi subclass và được gọi
        trước khi sử dụng model. Cần set self.is_loaded = True sau 
        khi load thành công.
        
        Raises:
            NotImplementedError: Nếu subclass không implement
            ModelLoadError: Nếu việc load model thất bại
        """
        pass
    
    @abstractmethod
    def predict(self, 
                queries: List[Dict[str, str]], 
                corpus: Dict[str, Dict[str, Any]], 
                top_k: int = 1000,
                **kwargs) -> Dict[str, Dict[str, float]]:
        """
        Thực hiện retrieval cho danh sách queries trên corpus.
        
        Đây là method chính để thực hiện retrieval. Input và output
        được standardized để tương thích với framework.
        
        Args:
            queries: Danh sách các query. Mỗi query là dict với format:
                {
                    'query_id': str,  # Required: unique identifier
                    'text': str,      # Text content (for text/multimodal)
                    'image_path': str, # Path to image (for vision/multimodal)
                    'title': str,     # Optional: query title
                    # ... other fields specific to model type
                }
                
            corpus: Dictionary các documents với format:
                {
                    'doc_id': {
                        'text': str,           # Text content
                        'image_path': str,     # Path to image file
                        'ocr_text': str,       # OCR extracted text
                        'title': str,          # Document title
                        'metadata': dict,      # Additional metadata
                        # ... other fields
                    },
                    ...
                }
                
            top_k: Số lượng documents có score cao nhất trả về cho mỗi query.
                Default: 1000
                
            **kwargs: Các tham số bổ sung cho specific models:
                - batch_size: Override default batch size
                - device: Override default device
                - normalize_scores: Có normalize scores hay không
                - filter_threshold: Minimum score threshold
                
        Returns:
            Dictionary với format:
            {
                'query_id': {
                    'doc_id': float,  # Score của document cho query này
                    ...
                },
                ...
            }
            
            Scores được sắp xếp theo thứ tự giảm dần (cao nhất trước).
            Chỉ trả về top_k documents có score cao nhất.
            
        Raises:
            RuntimeError: Nếu model chưa được load
            ValueError: Nếu input format không hợp lệ
            
        Example:
            >>> queries = [
            ...     {'query_id': 'q1', 'text': 'machine learning'},
            ...     {'query_id': 'q2', 'text': 'deep learning'}
            ... ]
            >>> corpus = {
            ...     'doc1': {'text': 'Introduction to ML'},
            ...     'doc2': {'text': 'Deep neural networks'}
            ... }
            >>> results = model.predict(queries, corpus, top_k=10)
            >>> # results = {'q1': {'doc1': 0.8, 'doc2': 0.3}, ...}
        """
        if not self.is_loaded:
            raise RuntimeError(f"Model {self.name} must be loaded before prediction")
            
        # Basic validation
        if not queries:
            raise ValueError("Queries list cannot be empty")
        if not corpus:
            raise ValueError("Corpus cannot be empty")
        if top_k <= 0:
            raise ValueError("top_k must be positive")
            
        # Log prediction request
        logger.info(f"Starting prediction for {len(queries)} queries "
                   f"on corpus of {len(corpus)} documents (top_k={top_k})")
    
    def index_corpus(self, 
                    corpus: Dict[str, Dict[str, Any]], 
                    **kwargs) -> None:
        """
        Tạo index cho corpus (optional, dành cho models cần pre-indexing).
        
        Method này được thiết kế cho các sparse models (như BM25) hoặc
        dense models có thể pre-compute embeddings để tăng tốc retrieval.
        
        Args:
            corpus: Dictionary corpus cần index với same format như predict()
            **kwargs: Additional indexing parameters:
                - cache_embeddings: Có cache embeddings hay không
                - rebuild_index: Force rebuild existing index
                - index_batch_size: Batch size cho indexing
                
        Note:
            Default implementation chỉ set flag _corpus_indexed = True.
            Subclasses nên override nếu cần custom indexing logic.
            
        Example:
            >>> model.index_corpus(corpus, cache_embeddings=True)
            >>> # Model đã sẵn sàng cho fast retrieval
        """
        logger.info(f"Indexing corpus with {len(corpus)} documents")
        self._corpus_indexed = True
        logger.info("Corpus indexing completed")
    
    def encode_queries(self, 
                      queries: List[Dict[str, str]], 
                      **kwargs) -> Dict[str, Any]:
        """
        Encode queries thành embeddings (cho dense models).
        
        Method này được thiết kế cho dense retrieval models cần
        convert queries thành vector representations.
        
        Args:
            queries: List queries với same format như predict()
            **kwargs: Encoding parameters:
                - batch_size: Batch size cho encoding
                - normalize: Có normalize embeddings hay không
                - show_progress: Có hiển thị progress bar hay không
                
        Returns:
            Dictionary mapping query_id -> query_embedding:
            {
                'query_id': np.ndarray,  # Shape: (embedding_dim,)
                ...
            }
            
        Note:
            Default implementation raise NotImplementedError.
            Dense models nên override method này.
            
        Raises:
            NotImplementedError: Nếu model không support query encoding
        """
        raise NotImplementedError(
            f"Model {self.name} does not support query encoding. "
            "Dense models should override this method."
        )
    
    def encode_documents(self, 
                        documents: Dict[str, Dict[str, Any]], 
                        **kwargs) -> Dict[str, Any]:
        """
        Encode documents thành embeddings (cho dense models).
        
        Method này được thiết kế cho dense retrieval models cần
        convert documents thành vector representations.
        
        Args:
            documents: Dictionary documents với same format như corpus trong predict()
            **kwargs: Encoding parameters:
                - batch_size: Batch size cho encoding  
                - normalize: Có normalize embeddings hay không
                - show_progress: Có hiển thị progress bar hay không
                - use_cache: Có sử dụng embedding cache hay không
                
        Returns:
            Dictionary mapping doc_id -> document_embedding:
            {
                'doc_id': np.ndarray,  # Shape: (embedding_dim,)
                ...
            }
            
        Note:
            Default implementation raise NotImplementedError.
            Dense models nên override method này.
            
        Raises:
            NotImplementedError: Nếu model không support document encoding
        """
        raise NotImplementedError(
            f"Model {self.name} does not support document encoding. "
            "Dense models should override this method."
        )
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Trả về thông tin metadata của model.
        
        Returns:
            Dictionary chứa thông tin model:
            {
                'name': str,
                'type': str,
                'is_loaded': bool,
                'config': dict,
                'corpus_indexed': bool,
                'supported_features': list
            }
        """
        return {
            "name": self.name,
            "type": self.model_type.value,
            "is_loaded": self.is_loaded,
            "config": self.config.__dict__,
            "corpus_indexed": self._corpus_indexed,
            "supported_features": self._get_supported_features()
        }
    
    def _get_supported_features(self) -> List[str]:
        """
        Trả về list các features được hỗ trợ bởi model.
        
        Returns:
            List các features: ['text_queries', 'image_queries', 'indexing', ...]
        """
        features = []
        
        # Check if model supports different query types
        try:
            # Try to call encode_queries to see if implemented
            self.encode_queries([])
            features.append("query_encoding")
        except NotImplementedError:
            pass
        except Exception:
            # Method exists but failed due to empty input
            features.append("query_encoding")
            
        try:
            # Try to call encode_documents to see if implemented  
            self.encode_documents({})
            features.append("document_encoding")
        except NotImplementedError:
            pass
        except Exception:
            # Method exists but failed due to empty input
            features.append("document_encoding")
            
        # Check model type specific features
        if self.model_type == ModelType.SPARSE:
            features.extend(["sparse_retrieval", "term_matching"])
        elif self.model_type == ModelType.DENSE:
            features.extend(["dense_retrieval", "semantic_similarity"])
        elif self.model_type == ModelType.VISION:
            features.extend(["image_retrieval", "visual_features"])
        elif self.model_type == ModelType.MULTIMODAL:
            features.extend(["multimodal_retrieval", "cross_modal_search"])
            
        return features
    
    def cleanup(self) -> None:
        """
        Cleanup resources khi không sử dụng model nữa.
        
        Override method này nếu model cần custom cleanup logic
        (ví dụ: release GPU memory, close connections, etc.)
        """
        logger.info(f"Cleaning up model: {self.name}")
        pass
    
    def __str__(self) -> str:
        """String representation của model."""
        return f"{self.__class__.__name__}(name='{self.name}', type='{self.model_type.value}')"
    
    def __repr__(self) -> str:
        """Detailed string representation của model."""
        return (f"{self.__class__.__name__}("
                f"name='{self.name}', "
                f"type='{self.model_type.value}', "
                f"is_loaded={self.is_loaded}, "
                f"device='{self.config.device}')")


# Convenience base classes for specific model types
class TextRetrievalModel(BaseRetrievalModel):
    """
    Base class cho text-only retrieval models (sparse và dense).
    
    Lớp này cung cấp specialized interface cho models chỉ xử lý text,
    bao gồm cả sparse models (BM25, TF-IDF) và dense models (BERT, DPR).
    """
    
    def __init__(self, model_config: Dict[str, Any], **kwargs):
        super().__init__(model_config, **kwargs)
        if self.model_type not in [ModelType.SPARSE, ModelType.DENSE, ModelType.CUSTOM]:
            logger.warning(f"TextRetrievalModel with type {self.model_type} "
                         "should typically be SPARSE or DENSE")


class VisionRetrievalModel(BaseRetrievalModel):
    """
    Base class cho vision-based retrieval models.
    
    Lớp này cung cấp specialized interface cho models xử lý images,
    bao gồm CLIP, vision transformers, và OCR-based models.
    """
    
    def __init__(self, model_config: Dict[str, Any], **kwargs):
        super().__init__(model_config, **kwargs)
        if self.model_type not in [ModelType.VISION, ModelType.MULTIMODAL, ModelType.CUSTOM]:
            logger.warning(f"VisionRetrievalModel with type {self.model_type} "
                         "should typically be VISION or MULTIMODAL")
    
    def encode_images(self, 
                     image_paths: List[str], 
                     **kwargs) -> np.ndarray:
        """
        Encode images thành embeddings.
        
        Args:
            image_paths: List đường dẫn tới image files
            **kwargs: Encoding parameters
            
        Returns:
            NumPy array shape (num_images, embedding_dim)
            
        Note:
            Vision models nên override method này.
        """
        raise NotImplementedError(
            f"Model {self.name} does not support image encoding"
        )


class MultiModalRetrievalModel(BaseRetrievalModel):
    """
    Base class cho multimodal retrieval models.
    
    Lớp này cung cấp specialized interface cho models có thể xử lý
    cả text và images, thực hiện cross-modal retrieval.
    """
    
    def __init__(self, model_config: Dict[str, Any], **kwargs):
        super().__init__(model_config, **kwargs)
        if self.model_type not in [ModelType.MULTIMODAL, ModelType.CUSTOM]:
            logger.warning(f"MultiModalRetrievalModel with type {self.model_type} "
                         "should typically be MULTIMODAL")
    
    def encode_text_with_vision(self, 
                               texts: List[str], 
                               **kwargs) -> np.ndarray:
        """
        Encode text trong multimodal space.
        
        Args:
            texts: List text strings
            **kwargs: Encoding parameters
            
        Returns:
            NumPy array shape (num_texts, embedding_dim)
        """
        raise NotImplementedError(
            f"Model {self.name} does not support text encoding in multimodal space"
        )
    
    def encode_images_with_text(self, 
                               image_paths: List[str], 
                               **kwargs) -> np.ndarray:
        """
        Encode images trong multimodal space.
        
        Args:
            image_paths: List đường dẫn tới image files
            **kwargs: Encoding parameters
            
        Returns:
            NumPy array shape (num_images, embedding_dim)
        """
        raise NotImplementedError(
            f"Model {self.name} does not support image encoding in multimodal space"
        )
