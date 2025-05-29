"""
Image-based retrieval models for NewAIBench framework.

This module provides specialized retrieval models for document images:
1. OCRBasedDocumentRetriever - Text retrieval on OCR extracted text
2. ImageEmbeddingDocumentRetriever - Visual retrieval using image embeddings (CLIP)
3. MultimodalDocumentRetriever - Combined OCR + image embeddings (future)
"""

import logging
from typing import Dict, List, Any, Optional, Tuple, Union
import numpy as np
from pathlib import Path
import pickle
import warnings

# Core ML imports
try:
    import torch
    import torch.nn.functional as F
    from transformers import CLIPModel, CLIPProcessor, CLIPTokenizer
    from transformers import AutoTokenizer, AutoModel
    from PIL import Image, ImageFile
    VISION_AVAILABLE = True
    ImageFile.LOAD_TRUNCATED_IMAGES = True  # Handle truncated images gracefully
except ImportError as e:
    VISION_AVAILABLE = False
    warnings.warn(f"Vision dependencies not available: {e}")

try:
    from sentence_transformers import SentenceTransformer
    SBERT_AVAILABLE = True
except ImportError:
    SBERT_AVAILABLE = False
    warnings.warn("sentence-transformers not available")

# Import base classes
from .base import BaseRetrievalModel
from .dense import DenseTextRetriever

logger = logging.getLogger(__name__)


class OCRBasedDocumentRetriever(DenseTextRetriever):
    """
    OCR-based document retrieval model.
    
    This class extends DenseTextRetriever to work specifically with document images
    by utilizing OCR extracted text. It inherits all the dense text retrieval
    capabilities while adding specialized handling for OCR text processing.
    
    Features:
    - Inherits all DenseTextRetriever functionality
    - Specialized OCR text extraction and preprocessing
    - Handles missing or poor quality OCR text gracefully
    - Configurable text quality filtering
    - Support for multiple OCR text fields
    
    Example:
        >>> config = {
        ...     "name": "ocr_retriever",
        ...     "type": "dense", 
        ...     "model_name_or_path": "all-MiniLM-L6-v2",
        ...     "parameters": {
        ...         "ocr_text_field": "ocr_text",
        ...         "min_ocr_confidence": 0.5,
        ...         "fallback_to_title": True,
        ...         "normalize_embeddings": True
        ...     }
        ... }
        >>> model = OCRBasedDocumentRetriever(config)
    """
    
    def __init__(self, model_config: Dict[str, Any], **kwargs):
        """Initialize OCR-based document retriever."""
        super().__init__(model_config, **kwargs)
        
        # OCR-specific parameters
        params = self.config.parameters
        self.ocr_text_field = params.get('ocr_text_field', 'ocr_text')
        self.min_ocr_confidence = params.get('min_ocr_confidence', 0.0)
        self.fallback_to_title = params.get('fallback_to_title', True)
        self.fallback_fields = params.get('fallback_fields', ['title', 'text', 'content'])
        self.min_text_length = params.get('min_text_length', 10)
        self.skip_empty_docs = params.get('skip_empty_docs', False)
        
        logger.info(f"Initialized OCRBasedDocumentRetriever with OCR field: {self.ocr_text_field}")
    
    def extract_text(self, document: Dict[str, Any]) -> str:
        """
        Extract text content from document with OCR-specific fallback strategies.
        
        Args:
            document: Document dictionary potentially containing OCR text
            
        Returns:
            Extracted text content, empty string if no text found
        """
        # Primary: Try OCR text field
        ocr_text = document.get(self.ocr_text_field, '')
        if ocr_text and isinstance(ocr_text, str):
            # Check OCR confidence if available
            ocr_confidence = document.get('ocr_confidence', 1.0)
            if ocr_confidence >= self.min_ocr_confidence:
                if len(ocr_text.strip()) >= self.min_text_length:
                    return ocr_text.strip()
        
        # Fallback strategies
        if self.fallback_to_title:
            for field in self.fallback_fields:
                text = document.get(field, '')
                if text and isinstance(text, str) and len(text.strip()) >= self.min_text_length:
                    logger.debug(f"Using fallback field '{field}' for document")
                    return text.strip()
        
        # Final fallback: combine available text fields
        text_parts = []
        for field in ['title', 'text', 'content', self.ocr_text_field]:
            if field in document and document[field]:
                text_parts.append(str(document[field]).strip())
        
        combined_text = ' '.join(text_parts).strip()
        return combined_text if len(combined_text) >= self.min_text_length else ""
    
    def encode_documents(self, 
                        documents: Dict[str, Dict[str, Any]], 
                        **kwargs) -> Dict[str, np.ndarray]:
        """
        Encode documents using OCR text with specialized preprocessing.
        
        Args:
            documents: Dictionary of documents containing OCR text
            **kwargs: Additional encoding parameters
            
        Returns:
            Dictionary mapping doc_id to embedding array
        """
        if not self.is_loaded:
            raise RuntimeError("Model must be loaded before encoding documents")
        
        show_progress = kwargs.get('show_progress', False)
        
        # Extract text from all documents
        doc_texts = []
        doc_ids = []
        skipped_docs = []
        
        for doc_id, doc_data in documents.items():
            text_content = self.extract_text(doc_data)
            
            if not text_content and self.skip_empty_docs:
                skipped_docs.append(doc_id)
                logger.warning(f"Skipping document {doc_id}: no sufficient text content")
                continue
            
            # Use empty string placeholder for empty documents
            if not text_content:
                text_content = ""
                logger.warning(f"No text content found for document {doc_id}, using empty string")
            
            doc_texts.append(text_content)
            doc_ids.append(doc_id)
        
        if skipped_docs:
            logger.info(f"Skipped {len(skipped_docs)} documents due to insufficient text content")
        
        # Encode all documents
        if doc_texts:
            embeddings = self.encode_texts(doc_texts, is_query=False, show_progress=show_progress)
            return {doc_id: emb for doc_id, emb in zip(doc_ids, embeddings)}
        else:
            logger.warning("No documents to encode after text extraction")
            return {}
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get OCR-specific model information."""
        info = super().get_model_info()
        info.update({
            "model_subtype": "OCR-based",
            "ocr_text_field": self.ocr_text_field,
            "min_ocr_confidence": self.min_ocr_confidence,
            "fallback_to_title": self.fallback_to_title,
            "min_text_length": self.min_text_length,
            "skip_empty_docs": self.skip_empty_docs
        })
        return info


class ImageEmbeddingDocumentRetriever(BaseRetrievalModel):
    """
    Image embedding-based document retrieval using vision models like CLIP.
    
    This model performs text-to-image retrieval by encoding document images
    into embeddings and comparing them with text query embeddings in a shared
    semantic space.
    
    Features:
    - CLIP-based text-to-image retrieval
    - Batch processing for efficient image encoding
    - Support for various image formats
    - Robust error handling for corrupted images
    - Optional ANN indexing for fast search
    - Image preprocessing and validation
    
    Example:
        >>> config = {
        ...     "name": "clip_retriever",
        ...     "type": "vision",
        ...     "model_name_or_path": "openai/clip-vit-base-patch32",
        ...     "parameters": {
        ...         "use_ann_index": True,
        ...         "ann_backend": "faiss",
        ...         "image_size": 224,
        ...         "normalize_embeddings": True
        ...     }
        ... }
        >>> model = ImageEmbeddingDocumentRetriever(config)
    """
    
    def __init__(self, model_config: Dict[str, Any], **kwargs):
        """Initialize image embedding document retriever."""
        super().__init__(model_config, **kwargs)
        
        if not VISION_AVAILABLE:
            raise ImportError("Vision dependencies (transformers, PIL, torch) are required for ImageEmbeddingDocumentRetriever")
        
        # Model configuration
        params = self.config.parameters
        self.model_name_or_path = model_config.get('model_name_or_path', 'openai/clip-vit-base-patch32')
        
        # Image processing parameters
        self.image_size = params.get('image_size', 224)
        self.image_path_field = params.get('image_path_field', 'image_path')
        self.max_image_size_mb = params.get('max_image_size_mb', 50)
        self.supported_formats = params.get('supported_formats', ['.jpg', '.jpeg', '.png', '.bmp', '.tiff'])
        
        # Search parameters
        self.use_ann_index = params.get('use_ann_index', False)
        self.ann_backend = params.get('ann_backend', 'faiss')
        self.normalize_embeddings = params.get('normalize_embeddings', True)
        
        # Model components (initialized in load_model)
        self.clip_model = None
        self.clip_processor = None
        self.clip_tokenizer = None
        self.embedding_dim = None
        
        # Determine model type based on path
        self.is_sentence_transformer = self.model_name_or_path.startswith('sentence-transformers/') or 'sentence-transformers' in self.model_name_or_path
        self.sentence_transformer_model = None
        
        # Storage
        self.doc_embeddings = {}
        self.doc_ids_list = []
        self.ann_index = None
        
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
        
        logger.info(f"Initialized ImageEmbeddingDocumentRetriever with model: {self.model_name_or_path}")
    
    def load_model(self) -> None:
        """Load CLIP model and processor."""
        try:
            logger.info(f"Loading CLIP model: {self.model_name_or_path}")
            
            if self.is_sentence_transformer:
                # Use SentenceTransformer for sentence-transformers models
                if not SBERT_AVAILABLE:
                    raise ImportError("sentence-transformers is required for this model")
                
                logger.info("Loading as SentenceTransformer model")
                self.sentence_transformer_model = SentenceTransformer(self.model_name_or_path)
                
                # Move to device
                device = self.config.device
                if device == "auto":
                    device = "cuda" if torch.cuda.is_available() else "cpu"
                
                self.sentence_transformer_model = self.sentence_transformer_model.to(device)
                
                # For sentence-transformers CLIP models, embedding dimension is typically 512
                self.embedding_dim = 512  # This will be verified during first encoding
                
            else:
                # Use transformers CLIP models for OpenAI and other standard CLIP models
                self.clip_model = CLIPModel.from_pretrained(self.model_name_or_path)
                self.clip_processor = CLIPProcessor.from_pretrained(self.model_name_or_path)
                self.clip_tokenizer = CLIPTokenizer.from_pretrained(self.model_name_or_path)
                
                # Move to device
                device = self.config.device
                if device == "auto":
                    device = "cuda" if torch.cuda.is_available() else "cpu"
                
                self.clip_model = self.clip_model.to(device)
                self.clip_model.eval()
                
                # Get embedding dimension
                self.embedding_dim = self.clip_model.config.projection_dim
            
            self.is_loaded = True
            logger.info(f"CLIP model loaded successfully. Embedding dimension: {self.embedding_dim}")
            
        except Exception as e:
            logger.error(f"Failed to load CLIP model {self.model_name_or_path}: {str(e)}")
            raise
    
    def _load_and_preprocess_image(self, image_path: str) -> Optional[Image.Image]:
        """
        Load and preprocess image for CLIP encoding.
        
        Args:
            image_path: Path to image file
            
        Returns:
            PIL Image object or None if loading fails
        """
        try:
            image_path = Path(image_path)
            
            # Check if file exists
            if not image_path.exists():
                logger.warning(f"Image file not found: {image_path}")
                return None
            
            # Check file format
            if image_path.suffix.lower() not in self.supported_formats:
                logger.warning(f"Unsupported image format: {image_path.suffix}")
                return None
            
            # Check file size
            file_size_mb = image_path.stat().st_size / (1024 * 1024)
            if file_size_mb > self.max_image_size_mb:
                logger.warning(f"Image too large ({file_size_mb:.1f}MB): {image_path}")
                return None
            
            # Load image
            image = Image.open(image_path)
            
            # Convert to RGB if necessary
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            return image
            
        except Exception as e:
            logger.warning(f"Failed to load image {image_path}: {e}")
            return None
    
    def encode_queries(self, 
                      queries: List[Dict[str, str]], 
                      **kwargs) -> Dict[str, np.ndarray]:
        """
        Encode text queries using CLIP text encoder.
        
        Args:
            queries: List of query dictionaries with 'text' field
            **kwargs: Additional encoding parameters
            
        Returns:
            Dictionary mapping query_id to text embedding
        """
        if not self.is_loaded:
            raise RuntimeError("Model must be loaded before encoding queries")
        
        query_embeddings = {}
        batch_size = kwargs.get('batch_size', self.config.batch_size)
        
        # Process queries in batches
        for i in range(0, len(queries), batch_size):
            batch_queries = queries[i:i+batch_size]
            
            # Extract text content
            texts = []
            query_ids = []
            for query in batch_queries:
                query_id = query['query_id']
                text = query.get('text', '')
                
                if not text:
                    logger.warning(f"Empty text for query {query_id}")
                    text = ""
                
                texts.append(text)
                query_ids.append(query_id)
            
            # Encode texts
            if texts:
                if self.is_sentence_transformer:
                    # Use SentenceTransformer encoding for text
                    with torch.no_grad():
                        embeddings = self.sentence_transformer_model.encode(
                            texts,
                            batch_size=batch_size,
                            show_progress_bar=False,
                            convert_to_numpy=True,
                            normalize_embeddings=self.normalize_embeddings
                        )
                        
                        # Update embedding dimension if not set
                        if self.embedding_dim is None:
                            self.embedding_dim = embeddings.shape[1]
                        
                        for query_id, embedding in zip(query_ids, embeddings):
                            query_embeddings[query_id] = embedding
                else:
                    # Use transformers CLIP encoding
                    with torch.no_grad():
                        inputs = self.clip_tokenizer(
                            texts, 
                            padding=True, 
                            truncation=True, 
                            max_length=self.config.max_length,
                            return_tensors="pt"
                        ).to(self.clip_model.device)
                        
                        text_features = self.clip_model.get_text_features(**inputs)
                        
                        # Normalize if requested
                        if self.normalize_embeddings:
                            text_features = F.normalize(text_features, p=2, dim=1)
                        
                        # Convert to numpy and store
                        embeddings = text_features.cpu().numpy()
                        for query_id, embedding in zip(query_ids, embeddings):
                            query_embeddings[query_id] = embedding
        
        logger.info(f"Encoded {len(query_embeddings)} text queries")
        return query_embeddings
    
    def encode_documents(self, 
                        corpus: Dict[str, Dict[str, Any]], 
                        **kwargs) -> Dict[str, np.ndarray]:
        """
        Encode document images using CLIP vision encoder.
        
        Args:
            corpus: Dictionary of documents with image_path field
            **kwargs: Additional encoding parameters
            
        Returns:
            Dictionary mapping doc_id to image embedding
        """
        if not self.is_loaded:
            raise RuntimeError("Model must be loaded before encoding documents")
        
        show_progress = kwargs.get('show_progress', False)
        batch_size = kwargs.get('batch_size', self.config.batch_size)
        
        doc_embeddings = {}
        processed_count = 0
        failed_count = 0
        
        # Process documents in batches
        doc_items = list(corpus.items())
        
        # Setup progress bar if requested
        progress_bar = None
        if show_progress:
            try:
                from tqdm import tqdm
                total_batches = (len(doc_items) + batch_size - 1) // batch_size
                progress_bar = tqdm(total=total_batches, desc="Encoding images", unit="batch")
            except ImportError:
                pass
        
        for i in range(0, len(doc_items), batch_size):
            batch_items = doc_items[i:i+batch_size]
            
            if self.is_sentence_transformer:
                # For sentence-transformers, use image paths directly
                image_paths = []
                valid_doc_ids = []
                
                for doc_id, doc_data in batch_items:
                    image_path = doc_data.get(self.image_path_field)
                    
                    if not image_path:
                        logger.warning(f"No image path found for document {doc_id}")
                        failed_count += 1
                        continue
                    
                    # Check if image can be loaded (basic validation)
                    image = self._load_and_preprocess_image(image_path)
                    if image is not None:
                        image_paths.append(image_path)
                        valid_doc_ids.append(doc_id)
                    else:
                        failed_count += 1
                
                # Encode batch with sentence-transformers
                if image_paths:
                    try:
                        # Use sentence-transformers for image encoding
                        embeddings = self.sentence_transformer_model.encode(
                            image_paths,
                            batch_size=len(image_paths),
                            convert_to_numpy=True,
                            normalize_embeddings=self.normalize_embeddings,
                            show_progress_bar=False
                        )
                        
                        # Store embeddings
                        for doc_id, embedding in zip(valid_doc_ids, embeddings):
                            doc_embeddings[doc_id] = embedding
                            processed_count += 1
                    except Exception as e:
                        logger.error(f"Failed to encode image batch with sentence-transformers: {e}")
                        failed_count += len(image_paths)
            else:
                # For transformers CLIP, load PIL Images
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
                        failed_count += 1
                
                # Encode batch with transformers CLIP
                if images:
                    try:
                        with torch.no_grad():
                            inputs = self.clip_processor(
                                images=images,
                                return_tensors="pt"
                            ).to(self.clip_model.device)
                            
                            image_features = self.clip_model.get_image_features(**inputs)
                            
                            # Normalize if requested
                            if self.normalize_embeddings:
                                image_features = F.normalize(image_features, p=2, dim=1)
                            
                            # Store embeddings
                            embeddings = image_features.cpu().numpy()
                            for doc_id, embedding in zip(valid_doc_ids, embeddings):
                                doc_embeddings[doc_id] = embedding
                                processed_count += 1
                    except Exception as e:
                        logger.error(f"Failed to encode image batch with transformers CLIP: {e}")
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
            
            # Create FAISS index
            if self.normalize_embeddings:
                self.ann_index = faiss.IndexFlatIP(self.embedding_dim)  # Inner product for normalized vectors
            else:
                self.ann_index = faiss.IndexFlatL2(self.embedding_dim)  # L2 distance
                
            self.ann_index.add(embeddings_array.astype(np.float32))
            logger.info(f"Built FAISS index with {embeddings_array.shape[0]} embeddings")
            
        elif self.ann_backend == 'hnswlib' and self.hnswlib_available:
            import hnswlib
            
            # Create HNSWLIB index
            if self.normalize_embeddings:
                space = 'cosine'
            else:
                space = 'l2'
            
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
        doc_embeddings_array = np.array([self.doc_embeddings[doc_id] for doc_id in self.doc_ids_list])
        
        # Compute similarities
        if self.normalize_embeddings:
            similarities = np.dot(query_embeddings, doc_embeddings_array.T)
        else:
            similarities = 1 / (1 + np.linalg.norm(
                query_embeddings[:, np.newaxis] - doc_embeddings_array[np.newaxis, :], axis=2
            ))
        
        results = []
        for i, query_similarities in enumerate(similarities):
            # Get top-k
            top_indices = np.argsort(query_similarities)[::-1][:top_k]
            query_results = []
            
            for idx in top_indices:
                doc_id = self.doc_ids_list[idx]
                score = float(query_similarities[idx])
                query_results.append((doc_id, score))
            
            results.append(query_results)
        
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
                idx, dist = self.ann_index.knn_query(query_emb, k=top_k)
                all_indices.append(idx)
                # Convert distance to similarity
                if self.normalize_embeddings:
                    all_scores.append(1 - dist)  # Cosine distance to similarity
                else:
                    all_scores.append(1 / (1 + dist))  # L2 distance to similarity
            return np.array(all_scores), np.array(all_indices)
    
    def predict(self, 
                queries: List[Dict[str, str]], 
                corpus: Dict[str, Dict[str, Any]], 
                top_k: int = 1000,
                **kwargs) -> Dict[str, Dict[str, float]]:
        """
        Perform image-based retrieval for text queries.
        
        Args:
            queries: List of text queries
            corpus: Dictionary of documents with images
            top_k: Number of top documents to return
            **kwargs: Additional parameters
            
        Returns:
            Dictionary mapping query_id to {doc_id: score}
        """
        # Call parent validation
        super().predict(queries, corpus, top_k, **kwargs)
        
        # Ensure corpus is indexed
        if not self._corpus_indexed:
            logger.info("Corpus not indexed, indexing now...")
            self.index_corpus(corpus, **kwargs)
        
        # Check if corpus has changed
        if set(self.doc_ids_list) != set(corpus.keys()):
            logger.warning("Corpus has changed since indexing, re-indexing...")
            self.index_corpus(corpus, **kwargs)
        
        # Filter documents with images
        image_corpus = {
            doc_id: doc for doc_id, doc in corpus.items()
            if self.image_path_field in doc and doc[self.image_path_field]
        }
        
        if not image_corpus:
            logger.warning("No documents with images found in corpus")
            return {q['query_id']: {} for q in queries}
        
        # Encode queries
        query_embeddings_dict = self.encode_queries(queries)
        
        # Prepare embeddings array for search
        query_ids = [q['query_id'] for q in queries]
        query_embeddings = np.array([query_embeddings_dict[qid] for qid in query_ids])
        
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
        
        logger.info(f"Image retrieval completed for {len(queries)} queries")
        return results
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information."""
        return {
            "model_name": self.name,
            "model_type": "Vision",
            "model_path": self.model_name_or_path,
            "architecture": "CLIP",
            "embedding_dim": self.embedding_dim,
            "corpus_indexed": self._corpus_indexed,
            "use_ann_index": self.use_ann_index,
            "ann_backend": self.ann_backend if self.use_ann_index else None,
            "normalize_embeddings": self.normalize_embeddings,
            "supported_formats": self.supported_formats,
            "image_size": self.image_size
        }


# Future: MultimodalDocumentRetriever combining OCR + Image embeddings
class MultimodalDocumentRetriever(BaseRetrievalModel):
    """
    Multimodal document retriever combining OCR text and image embeddings.
    
    This is a placeholder for future implementation that would combine
    both OCR-based text retrieval and image-based retrieval for enhanced
    performance on document image datasets.
    
    Potential approaches:
    - Late fusion: Combine scores from separate OCR and image retrievers
    - Early fusion: Concatenate OCR and image embeddings
    - Attention-based fusion: Learn to weight OCR vs image features
    - Cross-modal attention: Allow text and image features to interact
    """
    
    def __init__(self, model_config: Dict[str, Any], **kwargs):
        super().__init__(model_config, **kwargs)
        raise NotImplementedError("MultimodalDocumentRetriever is planned for future implementation")
    
    def load_model(self) -> None:
        raise NotImplementedError("MultimodalDocumentRetriever is planned for future implementation")
    
    def predict(self, queries, corpus, top_k=1000, **kwargs):
        raise NotImplementedError("MultimodalDocumentRetriever is planned for future implementation")
