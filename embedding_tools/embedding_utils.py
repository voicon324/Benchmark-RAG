"""
Utility functions để sử dụng OpenAI embeddings trong NewAIBench.
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import logging

logger = logging.getLogger(__name__)


class EmbeddingLoader:
    """Class để load và quản lý embeddings đã tạo."""
    
    def __init__(self, embeddings_dir: str, dataset_name: str):
        """
        Initialize embedding loader.
        
        Args:
            embeddings_dir: Thư mục chứa embeddings
            dataset_name: Tên dataset
        """
        self.embeddings_dir = Path(embeddings_dir)
        self.dataset_name = dataset_name
        self.dataset_dir = self.embeddings_dir / dataset_name
        
        # Kiểm tra dataset tồn tại
        if not self.dataset_dir.exists():
            raise FileNotFoundError(f"Embedding dataset not found: {self.dataset_dir}")
        
        # Load metadata
        self.metadata = self._load_metadata()
        
    def _load_metadata(self) -> Dict[str, Any]:
        """Load embedding metadata."""
        metadata_file = self.dataset_dir / "embedding_metadata.json"
        if metadata_file.exists():
            with open(metadata_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        return {}
    
    def load_corpus_embeddings(self) -> Dict[str, np.ndarray]:
        """
        Load tất cả corpus embeddings.
        
        Returns:
            Dict mapping doc_id -> embedding
        """
        corpus_dir = self.dataset_dir / "corpus"
        if not corpus_dir.exists():
            raise FileNotFoundError(f"Corpus embeddings not found: {corpus_dir}")
        
        embeddings = {}
        for file_path in corpus_dir.glob("*.npy"):
            doc_id = file_path.stem
            embedding = np.load(file_path)
            embeddings[doc_id] = embedding
            
        logger.info(f"Loaded {len(embeddings)} corpus embeddings")
        return embeddings
    
    def load_query_embeddings(self) -> Dict[str, np.ndarray]:
        """
        Load tất cả query embeddings.
        
        Returns:
            Dict mapping query_id -> embedding
        """
        queries_dir = self.dataset_dir / "queries"
        if not queries_dir.exists():
            raise FileNotFoundError(f"Query embeddings not found: {queries_dir}")
        
        embeddings = {}
        for file_path in queries_dir.glob("*.npy"):
            query_id = file_path.stem
            embedding = np.load(file_path)
            embeddings[query_id] = embedding
            
        logger.info(f"Loaded {len(embeddings)} query embeddings")
        return embeddings
    
    def load_single_corpus_embedding(self, doc_id: str) -> np.ndarray:
        """Load embedding cho một document."""
        file_path = self.dataset_dir / "corpus" / f"{doc_id}.npy"
        if not file_path.exists():
            raise FileNotFoundError(f"Corpus embedding not found: {file_path}")
        return np.load(file_path)
    
    def load_single_query_embedding(self, query_id: str) -> np.ndarray:
        """Load embedding cho một query."""
        file_path = self.dataset_dir / "queries" / f"{query_id}.npy"
        if not file_path.exists():
            raise FileNotFoundError(f"Query embedding not found: {file_path}")
        return np.load(file_path)
    
    def get_embedding_info(self) -> Dict[str, Any]:
        """Get thông tin về embeddings."""
        corpus_dir = self.dataset_dir / "corpus"
        queries_dir = self.dataset_dir / "queries"
        
        info = {
            "dataset_name": self.dataset_name,
            "num_corpus_embeddings": len(list(corpus_dir.glob("*.npy"))) if corpus_dir.exists() else 0,
            "num_query_embeddings": len(list(queries_dir.glob("*.npy"))) if queries_dir.exists() else 0,
            "metadata": self.metadata
        }
        
        # Get sample embedding shape
        if corpus_dir.exists():
            sample_files = list(corpus_dir.glob("*.npy"))
            if sample_files:
                sample_embedding = np.load(sample_files[0])
                info["embedding_shape"] = sample_embedding.shape
                info["embedding_dtype"] = str(sample_embedding.dtype)
        
        return info


class SimilarityCalculator:
    """Class để tính similarity giữa embeddings."""
    
    @staticmethod
    def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        """Tính cosine similarity giữa 2 vectors."""
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    
    @staticmethod
    def batch_cosine_similarity(query_emb: np.ndarray, corpus_embs: np.ndarray) -> np.ndarray:
        """
        Tính cosine similarity giữa 1 query và nhiều corpus documents.
        
        Args:
            query_emb: Query embedding shape (embedding_dim,)
            corpus_embs: Corpus embeddings shape (num_docs, embedding_dim)
            
        Returns:
            Similarities shape (num_docs,)
        """
        # Normalize embeddings
        query_norm = query_emb / np.linalg.norm(query_emb)
        corpus_norms = corpus_embs / np.linalg.norm(corpus_embs, axis=1, keepdims=True)
        
        # Compute cosine similarities
        similarities = np.dot(corpus_norms, query_norm)
        return similarities
    
    @staticmethod
    def euclidean_distance(a: np.ndarray, b: np.ndarray) -> float:
        """Tính Euclidean distance giữa 2 vectors."""
        return np.linalg.norm(a - b)
    
    @staticmethod
    def dot_product_similarity(a: np.ndarray, b: np.ndarray) -> float:
        """Tính dot product similarity giữa 2 vectors."""
        return np.dot(a, b)


class EmbeddingRetriever:
    """Class để thực hiện retrieval sử dụng embeddings."""
    
    def __init__(self, embedding_loader: EmbeddingLoader):
        """
        Initialize retriever.
        
        Args:
            embedding_loader: EmbeddingLoader instance
        """
        self.loader = embedding_loader
        self.corpus_embeddings = None
        self.corpus_ids = None
        self.similarity_calc = SimilarityCalculator()
        
    def index_corpus(self):
        """Load và index corpus embeddings."""
        logger.info("Indexing corpus embeddings...")
        corpus_dict = self.loader.load_corpus_embeddings()
        
        # Convert to arrays for efficient computation
        self.corpus_ids = list(corpus_dict.keys())
        self.corpus_embeddings = np.array([corpus_dict[doc_id] for doc_id in self.corpus_ids])
        
        logger.info(f"Indexed {len(self.corpus_ids)} documents")
    
    def retrieve(self, query_id: str, top_k: int = 10) -> List[Tuple[str, float]]:
        """
        Retrieve top-k documents cho một query.
        
        Args:
            query_id: ID của query
            top_k: Số lượng documents cần retrieve
            
        Returns:
            List of (doc_id, similarity_score) sorted by score descending
        """
        if self.corpus_embeddings is None:
            self.index_corpus()
        
        # Load query embedding
        query_emb = self.loader.load_single_query_embedding(query_id)
        
        # Compute similarities
        similarities = self.similarity_calc.batch_cosine_similarity(query_emb, self.corpus_embeddings)
        
        # Get top-k results
        top_indices = np.argsort(similarities)[::-1][:top_k]
        results = [(self.corpus_ids[idx], float(similarities[idx])) for idx in top_indices]
        
        return results
    
    def batch_retrieve(self, query_ids: List[str], top_k: int = 10) -> Dict[str, List[Tuple[str, float]]]:
        """
        Retrieve cho nhiều queries.
        
        Args:
            query_ids: List các query IDs
            top_k: Số lượng documents cần retrieve cho mỗi query
            
        Returns:
            Dict mapping query_id -> list of (doc_id, similarity_score)
        """
        if self.corpus_embeddings is None:
            self.index_corpus()
        
        results = {}
        for query_id in query_ids:
            results[query_id] = self.retrieve(query_id, top_k)
            
        return results


def create_embedding_based_retriever(embeddings_dir: str, dataset_name: str) -> EmbeddingRetriever:
    """
    Factory function để tạo embedding-based retriever.
    
    Args:
        embeddings_dir: Thư mục chứa embeddings
        dataset_name: Tên dataset
        
    Returns:
        EmbeddingRetriever instance
    """
    loader = EmbeddingLoader(embeddings_dir, dataset_name)
    retriever = EmbeddingRetriever(loader)
    return retriever


def analyze_embeddings(embeddings_dir: str, dataset_name: str):
    """
    Phân tích và in thống kê về embeddings.
    
    Args:
        embeddings_dir: Thư mục chứa embeddings
        dataset_name: Tên dataset
    """
    loader = EmbeddingLoader(embeddings_dir, dataset_name)
    info = loader.get_embedding_info()
    
    print(f"📊 Embedding Analysis for {dataset_name}")
    print("=" * 50)
    print(f"📁 Dataset: {info['dataset_name']}")
    print(f"📄 Corpus embeddings: {info['num_corpus_embeddings']}")
    print(f"❓ Query embeddings: {info['num_query_embeddings']}")
    
    if 'embedding_shape' in info:
        print(f"🔢 Embedding shape: {info['embedding_shape']}")
        print(f"🔢 Embedding dtype: {info['embedding_dtype']}")
    
    if info['metadata']:
        print(f"🏷️  Model: {info['metadata'].get('embedding_model', 'Unknown')}")
        print(f"📅 Created: {info['metadata'].get('created_at', 'Unknown')}")
        if 'dimensions' in info['metadata'] and info['metadata']['dimensions']:
            print(f"📏 Dimensions: {info['metadata']['dimensions']}")
    
    # Sample similarity analysis
    if info['num_corpus_embeddings'] > 0 and info['num_query_embeddings'] > 0:
        print("\n🔍 Sample Similarity Analysis:")
        try:
            # Load some sample embeddings
            corpus_dict = loader.load_corpus_embeddings()
            query_dict = loader.load_query_embeddings()
            
            # Get first few items
            sample_corpus = list(corpus_dict.items())[:3]
            sample_queries = list(query_dict.items())[:3]
            
            calc = SimilarityCalculator()
            
            for query_id, query_emb in sample_queries:
                print(f"\n❓ Query {query_id}:")
                for doc_id, doc_emb in sample_corpus:
                    sim = calc.cosine_similarity(query_emb, doc_emb)
                    print(f"   📄 Doc {doc_id}: {sim:.4f}")
                    
        except Exception as e:
            print(f"   ⚠️  Could not compute sample similarities: {e}")


if __name__ == "__main__":
    # Demo usage
    import sys
    
    if len(sys.argv) < 3:
        print("Usage: python embedding_utils.py <embeddings_dir> <dataset_name>")
        print("Example: python embedding_utils.py ./embeddings tydiqa_goldp_vietnamese")
        sys.exit(1)
    
    embeddings_dir = sys.argv[1]
    dataset_name = sys.argv[2]
    
    # Analyze embeddings
    analyze_embeddings(embeddings_dir, dataset_name)
    
    # Demo retrieval
    print(f"\n🔍 Demo Retrieval for {dataset_name}")
    print("=" * 50)
    
    try:
        retriever = create_embedding_based_retriever(embeddings_dir, dataset_name)
        
        # Load some queries
        loader = EmbeddingLoader(embeddings_dir, dataset_name)
        query_dict = loader.load_query_embeddings()
        
        # Demo retrieval for first query
        if query_dict:
            sample_query_id = list(query_dict.keys())[0]
            results = retriever.retrieve(sample_query_id, top_k=5)
            
            print(f"❓ Query: {sample_query_id}")
            print("📄 Top 5 results:")
            for i, (doc_id, score) in enumerate(results, 1):
                print(f"   {i}. Doc {doc_id}: {score:.4f}")
    
    except Exception as e:
        print(f"⚠️  Demo retrieval failed: {e}")
