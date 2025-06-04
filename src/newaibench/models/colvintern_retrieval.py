"""
ColVintern-based retrieval model for NewAIBench framework.

This module provides a specialized retrieval model using ColVintern (5CD-AI/ColVintern-1B-v1)
for multimodal document retrieval with Vietnamese language optimization.
"""

import logging
from typing import Dict, List, Any, Optional, Tuple, Union
import numpy as np
from pathlib import Path
import warnings

# Core ML imports
try:
    import torch
    import torch.nn.functional as F
    from transformers import AutoModel, AutoProcessor
    from PIL import Image, ImageFile
    COLVINTERN_AVAILABLE = True
    ImageFile.LOAD_TRUNCATED_IMAGES = True  # Handle truncated images gracefully
except ImportError as e:
    COLVINTERN_AVAILABLE = False
    warnings.warn(f"ColVintern dependencies not available: {e}")

# Progress bar
try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False

# Import base classes
from .base import BaseRetrievalModel, ModelType

import numpy as np

def stack_uneven_arrays(list_of_arrays, fill_value=0):
    """
    Gom một danh sách các mảng NumPy 2D có số hàng khác nhau (nhưng cùng số cột)
    thành một mảng NumPy 3D duy nhất.

    Args:
        list_of_arrays (list): Danh sách các mảng NumPy 2D.
                               Mỗi mảng có thể có số hàng khác nhau (y_i)
                               nhưng phải có cùng số cột (z).
        fill_value (float, optional): Giá trị để điền vào các phần tử
                                      của mảng kết quả không được phủ bởi
                                      dữ liệu từ các mảng đầu vào.
                                      Mặc định là 0.

    Returns:
        numpy.ndarray: Mảng NumPy 3D có kích thước [a, b, c], trong đó:
                       a = len(list_of_arrays)
                       b = max(array.shape[0] for array in list_of_arrays)
                       c = list_of_arrays[0].shape[1] (giả sử tất cả đều giống nhau)
                       Hoặc None nếu danh sách đầu vào rỗng hoặc các mảng không hợp lệ.
    """
    if not list_of_arrays:
        print("Danh sách đầu vào rỗng.")
        return None

    # Kiểm tra xem tất cả các mảng có phải là 2D và có cùng số cột không
    num_cols_first_array = -1
    for i, arr in enumerate(list_of_arrays):
        if not isinstance(arr, np.ndarray) or arr.ndim != 2:
            print(f"Phần tử thứ {i} không phải là mảng NumPy 2D.")
            return None
        if i == 0:
            num_cols_first_array = arr.shape[1]
        elif arr.shape[1] != num_cols_first_array:
            print(f"Các mảng không có cùng số cột. Mảng 0 có {num_cols_first_array} cột, mảng {i} có {arr.shape[1]} cột.")
            return None

    # a: số lượng mảng
    a = len(list_of_arrays)

    # c: số cột (z), giả định tất cả các mảng có cùng số cột
    c = num_cols_first_array

    # b: số hàng tối đa (max y)
    # Sử dụng list comprehension để lấy số hàng của mỗi mảng, sau đó tìm max
    # Xử lý trường hợp list_of_arrays rỗng (đã được kiểm tra ở trên)
    max_rows = 0
    if a > 0:
        max_rows = max(arr.shape[0] for arr in list_of_arrays)
    b = max_rows

    # Tạo mảng 3D kết quả, khởi tạo với giá trị fill_value
    # (ví dụ: 0 hoặc np.nan)
    result_array = np.full((a, b, c), fill_value=fill_value, dtype=list_of_arrays[0].dtype) # Giữ nguyên dtype

    # Điền dữ liệu từ các mảng 2D vào mảng 3D
    for i, arr_2d in enumerate(list_of_arrays):
        rows_in_current_array = arr_2d.shape[0]
        result_array[i, :rows_in_current_array, :] = arr_2d

    return result_array

logger = logging.getLogger(__name__)


class ColVinternDocumentRetriever(BaseRetrievalModel):
    """
    ColVintern-based document retrieval model for Vietnamese multimodal documents.
    
    This model uses ColVintern (5CD-AI/ColVintern-1B-v1) for text-to-image retrieval 
    with specialized optimization for Vietnamese language and documents.
    
    Features:
    - Vietnamese-optimized multimodal retrieval
    - Text-to-image and image-to-text capabilities
    - Batch processing for efficient encoding
    - Support for various image formats
    - Robust error handling for corrupted images
    - Optional ANN indexing for fast search
    - Multi-vector scoring with ColVintern
    - Progress tracking for long-running operations
    - Memory-efficient query processing to avoid overflow
    
    Example:
        >>> config = {
        ...     "name": "colvintern_retriever",
        ...     "type": "multimodal",
        ...     "model_name_or_path": "5CD-AI/ColVintern-1B-v1",
        ...     "parameters": {
        ...         "use_ann_index": True,
        ...         "ann_backend": "faiss",
        ...         "normalize_embeddings": True,
        ...         "batch_size_images": 8,
        ...         "batch_size_text": 32,
        ...         "scoring_method": "multi_vector",
        ...         "query_batch_size_scoring": 2
        ...     }
        ... }
        >>> model = ColVinternDocumentRetriever(config)
    """
    
    def __init__(self, model_config: Dict[str, Any], **kwargs):
        """Initialize ColVintern document retriever."""
        super().__init__(model_config, **kwargs)
        
        if not COLVINTERN_AVAILABLE:
            raise ImportError(
                "ColVintern dependencies (transformers, PIL, torch) are required for ColVinternDocumentRetriever"
            )
        
        # Model configuration
        params = self.config.parameters
        self.model_name_or_path = model_config.get('model_name_or_path', '5CD-AI/ColVintern-1B-v1')
        
        # Processing parameters
        self.image_path_field = params.get('image_path_field', 'image_path')
        self.max_image_size_mb = params.get('max_image_size_mb', 50)
        self.supported_formats = params.get('supported_formats', ['.jpg', '.jpeg', '.png', '.bmp', '.tiff'])
        
        # Batch processing parameters
        self.batch_size_images = params.get('batch_size_images', 8)
        self.batch_size_text = params.get('batch_size_text', 32)
        
        # Search parameters
        self.use_ann_index = params.get('use_ann_index', False)
        self.ann_backend = params.get('ann_backend', 'faiss')
        self.normalize_embeddings = params.get('normalize_embeddings', False)  # Changed from True to False
        self.scoring_method = params.get('scoring_method', 'cosine_similarity')  # 'multi_vector' or 'cosine_similarity'
        self.query_batch_size_scoring = params.get('query_batch_size_scoring', 1)  # Batch size for multi-vector scoring
        
        # Model components (initialized in load_model)
        self.model = None
        self.processor = None
        self.embedding_dim = None
        
        # Storage
        self.doc_embeddings = {}
        self.doc_ids_list = []
        self.ann_index = None
        
        # Initialize availability flags
        self.faiss_available = False
        self.hnswlib_available = False
        
        # Validate ANN backend
        if self.use_ann_index:
            if self.ann_backend == 'faiss':
                try:
                    import faiss
                    self.faiss_available = True
                except ImportError:
                    logger.warning("FAISS not available, falling back to brute force search")
                    self.use_ann_index = False
            elif self.ann_backend == 'hnswlib':
                try:
                    import hnswlib
                    self.hnswlib_available = True
                except ImportError:
                    logger.warning("HNSWLIB not available, falling back to brute force search")
                    self.use_ann_index = False
        
        logger.info(f"Initialized ColVinternDocumentRetriever with model: {self.model_name_or_path}")
    
    def load_model(self) -> None:
        """Load ColVintern model and processor."""
        try:
            logger.info(f"Loading ColVintern model: {self.model_name_or_path}")
            
            # Load processor first
            self.processor = AutoProcessor.from_pretrained(
                self.model_name_or_path,
                trust_remote_code=True
            )
            
            # Load model with proper device configuration
            device = self.config.device
            if device == "auto":
                device = "cuda" if torch.cuda.is_available() else "cpu"
            
            # Load model with additional configuration to avoid config issues
            model_kwargs = {
                "torch_dtype": torch.bfloat16,
                "low_cpu_mem_usage": True,
                "trust_remote_code": True,
            }
            
            # Add device configuration for CUDA
            if device == "cuda" and torch.cuda.is_available():
                logger.info("Loading model on CUDA")
            else:
                logger.info("Loading model on CPU")
                model_kwargs["torch_dtype"] = torch.float32  # Use float32 for CPU
            print(f'Model kwargs: {model_kwargs}')
            self.model = AutoModel.from_pretrained(
                self.model_name_or_path,
                **model_kwargs
            ).eval()
            
            # Move model to device
            if device == "cuda" and torch.cuda.is_available():
                self.model = self.model.cuda()
            else:
                self.model = self.model.to(device)
            
            # Try to infer embedding dimension
            try:
                # Try different ways to get embedding dimension
                if hasattr(self.model, 'config'):
                    self.embedding_dim = getattr(self.model.config, 'hidden_size', None)
                    if self.embedding_dim is None:
                        self.embedding_dim = getattr(self.model.config, 'text_config', {}).get('hidden_size', None)
                    if self.embedding_dim is None:
                        self.embedding_dim = getattr(self.model.config, 'vision_config', {}).get('hidden_size', None)
                
                # Fallback to default
                if self.embedding_dim is None:
                    self.embedding_dim = 768  # Default for ColVintern-1B
                    logger.warning(f"Could not determine embedding dimension, using default: {self.embedding_dim}")
                
            except Exception as e:
                logger.warning(f"Could not determine embedding dimension: {e}, using default 768")
                self.embedding_dim = 768
            
            self.is_loaded = True
            logger.info(f"ColVintern model loaded successfully. Embedding dimension: {self.embedding_dim}")
            
        except Exception as e:
            logger.error(f"Failed to load ColVintern model {self.model_name_or_path}: {str(e)}")
            raise

    
    def _load_and_preprocess_image(self, image_path: str) -> Optional[Image.Image]:
        """
        Load and preprocess image for ColVintern encoding.
        
        Args:
            image_path: Path to image file
            
        Returns:
            PIL Image object or None if loading fails
        """
        try:
            # print(f"Loading image from path: {image_path}")
            # Check if file exists
            # if not image_path.exists():
            #     print(f"Image file not found: {image_path}")
            #     logger.warning(f"Image file not found: {image_path}")
            #     return None
            
            # Check file format
            # if image_path.suffix.lower() not in self.supported_formats:
            #     print(f"Unsupported image format: {image_path.suffix}")
            #     logger.warning(f"Unsupported image format: {image_path.suffix}")
            #     return None
            
            # Check file size
            # file_size_mb = image_path.stat().st_size / (1024 * 1024)
            # if file_size_mb > self.max_image_size_mb:
            #     print(f"Image too large ({file_size_mb:.1f}MB): {image_path}")
            #     logger.warning(f"Image too large ({file_size_mb:.1f}MB): {image_path}")
            #     return None
            
            # Load image
            # print(f"Loading image: {image_path}")
            image = Image.open(image_path)
            
            # Convert to RGB if necessary
            if image.mode != 'RGB':
                print(f"Converting image to RGB: {image_path}")
                image = image.convert('RGB')
            
            return image
            
        except Exception as e:
            logger.warning(f"Failed to load image {image_path}: {e}")
            return None
    
    def encode_queries(self, 
                      queries: List[Dict[str, str]], 
                      **kwargs) -> Dict[str, np.ndarray]:
        """
        Encode text queries using ColVintern in batches.
        
        Args:
            queries: List of query dictionaries with 'query_id' and 'text' keys
            
        Returns:
            Dictionary mapping query_id to embedding arrays
        """
        if not self.is_loaded:
            raise RuntimeError("Model must be loaded before encoding queries")
        
        query_embeddings = {}
        batch_size = kwargs.get('batch_size', self.batch_size_text)
        show_progress = kwargs.get('show_progress', True)
        
        # Extract text queries
        text_queries = []
        query_ids = []
        
        for query in queries:
            query_id = query.get('query_id')
            if not query_id:
                logger.warning("Query missing query_id, skipping")
                continue
            
            text = query.get('text', '')
            if not text:
                logger.warning(f"Empty text for query {query_id}")
                text = ""
            
            text_queries.append(text)
            query_ids.append(query_id)
        
        # Encode queries in batches
        if text_queries:
            try:
                # Initialize progress bar
                progress_bar = None
                if show_progress and TQDM_AVAILABLE:
                    try:
                        total_batches = (len(text_queries) + batch_size - 1) // batch_size
                        progress_bar = tqdm(total=total_batches, desc="Encoding queries", unit="batch")
                    except ImportError:
                        pass
                
                # Process queries in batches
                for i in range(0, len(text_queries), batch_size):
                    batch_texts = text_queries[i:i+batch_size]
                    batch_query_ids = query_ids[i:i+batch_size]
                    
                    # Process queries with ColVintern processor
                    batch_queries_processed = self.processor.process_queries(batch_texts)
                    
                    # Move to device and set correct data types (following API example)
                    if torch.cuda.is_available():
                        batch_queries_processed["input_ids"] = batch_queries_processed["input_ids"].cuda()
                        batch_queries_processed["attention_mask"] = batch_queries_processed["attention_mask"].cuda().bfloat16()
                    else:
                        # Fallback for CPU
                        batch_queries_processed["input_ids"] = batch_queries_processed["input_ids"]
                        batch_queries_processed["attention_mask"] = batch_queries_processed["attention_mask"].float()
                    
                    # Get embeddings
                    with torch.no_grad():
                        query_embeddings_tensor = self.model(**batch_queries_processed)
                    
                    # Convert to numpy and normalize if requested
                    embeddings = query_embeddings_tensor.cpu().float().numpy()
                    
                    # Remove automatic normalization since ColVintern works better without it
                    if self.normalize_embeddings:
                        embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
        
                    # Store embeddings
                    for query_id, embedding in zip(batch_query_ids, embeddings):
                        query_embeddings[query_id] = embedding
                        
                    # Update embedding dimension if not set
                    if self.embedding_dim is None:
                        self.embedding_dim = embeddings.shape[1]
                    
                    # Update progress bar
                    if progress_bar:
                        progress_bar.update(1)
                
                # Close progress bar
                if progress_bar:
                    progress_bar.close()
                
            except Exception as e:
                logger.error(f"Failed to encode queries with ColVintern: {e}")
                raise
        
        logger.info(f"Encoded {len(query_embeddings)} text queries in batches of {batch_size}")
        return query_embeddings
    
    def encode_documents(self, 
                        corpus: Dict[str, Dict[str, Any]], 
                        **kwargs) -> Dict[str, np.ndarray]:
        """
        Encode document images using ColVintern.
        
        Args:
            corpus: Dictionary mapping doc_id to document data
            
        Returns:
            Dictionary mapping doc_id to embedding arrays
        """
        if not self.is_loaded:
            raise RuntimeError("Model must be loaded before encoding documents")
        
        doc_embeddings = {}
        batch_size = kwargs.get('batch_size', self.batch_size_images)
        show_progress = kwargs.get('show_progress', True)
        
        # Filter documents with images
        doc_items = [
            (doc_id, doc_data) for doc_id, doc_data in corpus.items()
            if self.image_path_field in doc_data and doc_data[self.image_path_field]
        ]
        
        if not doc_items:
            logger.warning("No documents with images found in corpus")
            return doc_embeddings
        
        processed_count = 0
        failed_count = 0
        
        # Initialize progress bar
        progress_bar = None
        if show_progress and TQDM_AVAILABLE:
            try:
                total_batches = (len(doc_items) + batch_size - 1) // batch_size
                progress_bar = tqdm(total=total_batches, desc="Encoding images", unit="batch")
            except ImportError:
                pass
        
        for i in range(0, len(doc_items), batch_size):
            batch_items = doc_items[i:i+batch_size]
            
            # Load and validate images for batch
            images = []
            valid_doc_ids = []
            
            for doc_id, doc_data in batch_items:
                image_path = doc_data.get(self.image_path_field)
                
                if not image_path:
                    logger.warning(f"No image path found for document {doc_id}")
                    failed_count += 1
                    continue
                
                image = self._load_and_preprocess_image(image_path)
                if image is not None:
                    images.append(image)
                    valid_doc_ids.append(doc_id)
                else:
                    print(f"Failed to load image for document {doc_id}")
                    failed_count += 1
            # Encode batch with ColVintern
            if images:
                try:
                    # Process images with ColVintern processor
                    batch_images_processed = self.processor.process_images(images)
                    
                    # Move to device and set correct data types (following API example)
                    if torch.cuda.is_available():
                        batch_images_processed["pixel_values"] = batch_images_processed["pixel_values"].cuda().bfloat16()
                        batch_images_processed["input_ids"] = batch_images_processed["input_ids"].cuda()
                        batch_images_processed["attention_mask"] = batch_images_processed["attention_mask"].cuda().bfloat16()
                    else:
                        # Fallback for CPU
                        batch_images_processed["pixel_values"] = batch_images_processed["pixel_values"].float()
                        batch_images_processed["input_ids"] = batch_images_processed["input_ids"]
                        batch_images_processed["attention_mask"] = batch_images_processed["attention_mask"].float()
                    
                    # Get embeddings
                    with torch.no_grad():
                        image_embeddings = self.model(**batch_images_processed)
                    
                    # Convert to numpy and normalize if requested
                    embeddings = image_embeddings.cpu().float().numpy()
                    
                    # Remove automatic normalization since ColVintern works better without it
                    if self.normalize_embeddings:
                        embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
                    
                    # Store embeddings
                    for doc_id, embedding in zip(valid_doc_ids, embeddings):
                        doc_embeddings[doc_id] = embedding
                        processed_count += 1
                        
                    # Update embedding dimension if not set
                    if self.embedding_dim is None:
                        self.embedding_dim = embeddings.shape[1]
                        
                except Exception as e:
                    logger.error(f"Failed to encode image batch with ColVintern: {e}")
                    failed_count += len(images)
            
            # Update progress bar
            if progress_bar:
                progress_bar.update(1)
        
        # Close progress bar
        if progress_bar:
            progress_bar.close()
        
        logger.info(f"Encoded {processed_count} images, failed: {failed_count}")
        return doc_embeddings
    
    def _build_ann_index(self, embeddings_array: np.ndarray) -> None:
        """Build ANN index from embeddings array."""
        if self.ann_backend == 'faiss' and self.faiss_available:
            import faiss
            
            # Use inner product index for raw dot product similarity
            self.ann_index = faiss.IndexFlatIP(self.embedding_dim)  # Inner product for raw embeddings
                
            self.ann_index.add(embeddings_array.astype(np.float32))
            logger.info(f"Built FAISS index with {embeddings_array.shape[0]} embeddings")
            
        elif self.ann_backend == 'hnswlib' and self.hnswlib_available:
            import hnswlib
            
            # Use inner product space for raw dot product
            space = 'ip'  # Inner product space
                
            self.ann_index = hnswlib.Index(space=space, dim=self.embedding_dim)
            self.ann_index.init_index(max_elements=len(embeddings_array), ef_construction=200, M=16)
            self.ann_index.add_items(embeddings_array, list(range(len(embeddings_array))))
            self.ann_index.set_ef(50)
            
            logger.info(f"Built HNSWLIB index with {embeddings_array.shape[0]} embeddings")

    def index_corpus(self, corpus: Dict[str, Dict[str, Any]], **kwargs) -> None:
        """Index document images for fast retrieval."""
        if not self.is_loaded:
            raise RuntimeError("Model must be loaded before indexing")
        
        logger.info(f"Indexing corpus with {len(corpus)} documents")
        
        # Encode all documents
        self.doc_embeddings = self.encode_documents(corpus, show_progress=True)
        self.doc_ids_list = list(self.doc_embeddings.keys())
        
        # Build ANN index if requested
        if self.use_ann_index and self.doc_embeddings:
            embeddings_array = np.array([self.doc_embeddings[doc_id] for doc_id in self.doc_ids_list])
            self._build_ann_index(embeddings_array)
        
        self._corpus_indexed = True
        logger.info(f"Corpus indexing completed. Embeddings: {len(self.doc_embeddings)}")
    
    def _search_brute_force(self, query_embeddings: np.ndarray, top_k: int) -> List[List[Tuple[str, float]]]:
        """Perform brute force similarity search."""
        logger.info(f"Starting brute force search for {len(query_embeddings)} queries against {len(self.doc_ids_list)} documents")
        
        doc_embeddings_array = stack_uneven_arrays([self.doc_embeddings[doc_id] for doc_id in self.doc_ids_list])
        
        # Use multi-vector scoring if configured
        if self.scoring_method == 'multi_vector':
            logger.info(f"Using multi-vector scoring method with batch size {self.query_batch_size_scoring}")
            
            # Convert to tensors for multi-vector scoring
            query_embeddings_tensor = torch.tensor(query_embeddings).float()
            doc_embeddings_tensor = torch.tensor(doc_embeddings_array).float()
            # print(f'Query embeddings shape: {query_embeddings_tensor.shape}, Document embeddings shape: {doc_embeddings_tensor.shape}')
            
            # Use ColVintern's multi-vector scoring with batch processing
            similarities_tensor = self.score_multi_vector(
                query_embeddings_tensor, 
                doc_embeddings_tensor, 
                query_batch_size=self.query_batch_size_scoring
            )
            similarities = similarities_tensor.numpy()
            
            logger.debug(f"Using multi-vector scoring for {len(query_embeddings)} queries x {len(doc_embeddings_array)} documents")
        else:
            logger.info("Using raw dot product similarity scoring")
            
            # Raw dot product similarity scoring (no normalization, no cosine)
            similarities = []
            for i, query_emb in enumerate(query_embeddings):
                # Show progress for dot product
                if i % max(1, len(query_embeddings) // 10) == 0 or i == len(query_embeddings) - 1:
                    print(f"Dot product similarity progress: {i+1}/{len(query_embeddings)} queries")
                
                # Compute raw dot product similarities
                query_similarities = np.dot(query_emb, doc_embeddings_array.T)
                similarities.append(query_similarities)
            similarities = np.array(similarities)
        
        logger.info("Processing search results...")
        
        # Process results for each query
        results = []
        for i, query_emb in enumerate(query_embeddings):
            query_similarities = similarities[i]
            
            # Get top-k
            top_indices = np.argsort(query_similarities)[::-1][:top_k]
            query_results = [
                (self.doc_ids_list[idx], float(query_similarities[idx]))
                for idx in top_indices
            ]
            results.append(query_results)
        
        logger.info(f"Brute force search completed for {len(query_embeddings)} queries")
        return results
    
    def _search_ann(self, query_embeddings: np.ndarray, top_k: int) -> Tuple[np.ndarray, np.ndarray]:
        """Perform ANN search."""
        if self.ann_backend == 'faiss':
            scores, indices = self.ann_index.search(query_embeddings.astype(np.float32), top_k)
            return scores, indices
            
        elif self.ann_backend == 'hnswlib':
            all_indices = []
            all_scores = []
            for query_emb in query_embeddings:
                idx, scores = self.ann_index.knn_query(query_emb, k=top_k)
                all_indices.append(idx)
                # For inner product, higher scores are better (no conversion needed)
                all_scores.append(scores)
            return np.array(all_scores), np.array(all_indices)
    
    def score_multi_vector(self, query_embeddings: torch.Tensor, image_embeddings: torch.Tensor, 
                          query_batch_size: int = 1) -> torch.Tensor:
        """
        Compute multi-vector scores using ColVintern's scoring method.
        Processes queries in small batches to avoid memory overflow with large datasets.
        
        Args:
            query_embeddings: Query embeddings from model [num_queries, embedding_dim]
            image_embeddings: Image embeddings from model [num_images, embedding_dim]
            query_batch_size: Number of queries to process at once (default=1 for maximum memory safety)
            
        Returns:
            Similarity scores [num_queries, num_images]
        """
        num_queries = query_embeddings.shape[0]
        num_images = image_embeddings.shape[0]
        
        # Log memory usage info
        logger.info(f"Computing multi-vector scores: {num_queries} queries x {num_images} images (batch_size={query_batch_size})")
        
        # Initialize result tensor
        all_scores = torch.zeros(num_queries, num_images, dtype=query_embeddings.dtype, device=query_embeddings.device)
        
        # Calculate total batches for progress tracking
        total_batches = (num_queries + query_batch_size - 1) // query_batch_size
        
        # Initialize progress bar for multi-vector scoring
        progress_bar = None
        if TQDM_AVAILABLE:
            try:
                progress_bar = tqdm(total=total_batches, desc="Multi-vector scoring", unit="batch")
            except ImportError:
                pass
        
        # Process queries in small batches to avoid memory overflow
        for batch_idx, i in enumerate(range(0, num_queries, query_batch_size)):
            end_idx = min(i + query_batch_size, num_queries)
            query_batch = query_embeddings[i:end_idx]  # [batch_size, embedding_dim]
            batch_size = end_idx - i
            
            try:
                # Use ColVintern's processor scoring method for query batch
                with torch.no_grad():  # Ensure no gradients are computed
                    scores = self.processor.score_multi_vector(query_batch, image_embeddings)
                all_scores[i:end_idx] = scores  # Store batch results
                
                # Clear cache periodically to manage memory
                if batch_idx % 10 == 0 and torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
            except Exception as e:
                logger.warning(f"Failed to use ColVintern multi-vector scoring for batch {i}-{end_idx}, falling back to dot product: {e}")
                # Fallback to simple dot product for this batch
                with torch.no_grad():
                    scores = torch.matmul(query_batch, image_embeddings.T)
                all_scores[i:end_idx] = scores
            
            # Update progress bar
            if progress_bar:
                progress_bar.set_postfix({
                    'Queries': f"{end_idx}/{num_queries}",
                    'Batch': f"{batch_idx+1}/{total_batches}"
                })
                progress_bar.update(1)
            else:
                # Print progress if no tqdm available
                if (batch_idx + 1) % max(1, total_batches // 10) == 0 or batch_idx == total_batches - 1:
                    print(f"Multi-vector scoring progress: {batch_idx+1}/{total_batches} batches ({end_idx}/{num_queries} queries)")
        
        # Close progress bar
        if progress_bar:
            progress_bar.close()
        
        # Final memory cleanup
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        logger.info(f"Multi-vector scoring completed for {num_queries} queries")
        return all_scores
    
    def predict(self, 
                queries: List[Dict[str, str]], 
                corpus: Dict[str, Dict[str, Any]], 
                top_k: int = 1000,
                **kwargs) -> Dict[str, Dict[str, float]]:
        """
        Perform retrieval on document corpus.
        
        Args:
            queries: List of query dictionaries
            corpus: Document corpus
            top_k: Number of top results to return
            
        Returns:
            Dictionary mapping query_id to document scores
        """
        # Call parent predict for basic validation
        super().predict(queries, corpus, top_k, **kwargs)
        
        # Check if corpus needs re-indexing
        current_corpus_keys = set(corpus.keys())
        indexed_corpus_keys = set(self.doc_ids_list) if self.doc_ids_list else set()
        
        if current_corpus_keys != indexed_corpus_keys:
            logger.warning("Corpus has changed since indexing, re-indexing...")
            self.index_corpus(corpus, **kwargs)
        
        print(f'Num corpus: {len(corpus)}, Num indexed: {len(self.doc_ids_list)}')
        # Filter documents with images
        image_corpus = {
            doc_id: doc for doc_id, doc in corpus.items()
            if self.image_path_field in doc and doc[self.image_path_field]
        }
        
        print(f'Num image corpus: {len(image_corpus)}')

        if not image_corpus:
            logger.warning("No documents with images found in corpus")
            return {q['query_id']: {} for q in queries}
        
        # Encode queries
        query_embeddings_dict = self.encode_queries(queries)
        
        # Prepare embeddings array for search
        query_ids = [q['query_id'] for q in queries]
        query_embeddings = stack_uneven_arrays([query_embeddings_dict[qid] for qid in query_ids])
        
        # Perform search
        if self.use_ann_index and self.ann_index is not None:
            logger.debug(f"Performing ANN search with {self.ann_backend}")
            scores, indices = self._search_ann(query_embeddings, top_k)
            
            # Convert to results format
            results = {}
            min_score = kwargs.get('min_score', 0.0)
            
            for i, query_id in enumerate(query_ids):
                query_results = {}
                
                for j in range(len(indices[i])):
                    if indices[i][j] == -1:  # Invalid index
                        break
                    
                    doc_idx = indices[i][j]
                    score = float(scores[i][j])
                    
                    if score >= min_score:
                        doc_id = self.doc_ids_list[doc_idx]
                        query_results[doc_id] = score
                
                results[query_id] = query_results
        else:
            logger.debug("Performing brute-force search")
            search_results = self._search_brute_force(query_embeddings, top_k)
            
            # Convert to results format
            results = {}
            min_score = kwargs.get('min_score', 0.0)
            
            for query_id, query_results_list in zip(query_ids, search_results):
                query_results = {
                    doc_id: score for doc_id, score in query_results_list
                    if score >= min_score
                }
                results[query_id] = query_results
        
        logger.info(f"ColVintern retrieval completed for {len(queries)} queries")
        return results
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information."""
        from .base import count_model_parameters
        
        param_count = None
        if hasattr(self, 'model') and self.model is not None:
            param_count = count_model_parameters(self.model)
        
        return {
            "model_name": self.name,
            "model_type": "Multimodal",
            "model_path": self.model_name_or_path,
            "architecture": "ColVintern",
            "embedding_dim": self.embedding_dim,
            "parameter_count": param_count,
            "corpus_indexed": self._corpus_indexed,
            "use_ann_index": self.use_ann_index,
            "ann_backend": self.ann_backend if self.use_ann_index else None,
            "normalize_embeddings": self.normalize_embeddings,
            "scoring_method": self.scoring_method,
            "query_batch_size_scoring": self.query_batch_size_scoring,
            "supported_formats": self.supported_formats,
            "batch_size_images": self.batch_size_images,
            "batch_size_text": self.batch_size_text
        }
