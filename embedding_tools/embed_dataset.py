#!/usr/bin/env python3
"""
Script để embedding dataset sử dụng OpenAI API.
Sẽ load corpus và queries từ dataset và tạo embeddings cho từng văn bản.
Mỗi embedding sẽ được lưu thành một file riêng biệt.
"""

import os
import json
import asyncio
import aiohttp
import argparse
from pathlib import Path
from typing import List, Dict, Any, Optional
import time
from tqdm import tqdm
import logging
from dataclasses import dataclass
import pickle
import numpy as np

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class EmbeddingConfig:
    """Configuration for embedding process."""
    api_key: str
    model: str = "text-embedding-3-small"  # Hoặc "text-embedding-3-large"
    batch_size: int = 100  # Số lượng text xử lý đồng thời
    max_retries: int = 3
    retry_delay: float = 1.0
    dimensions: Optional[int] = None  # Để None sẽ sử dụng dimension mặc định của model
    

class OpenAIEmbedder:
    """Class để tạo embeddings sử dụng OpenAI API."""
    
    def __init__(self, config: EmbeddingConfig):
        self.config = config
        self.session = None
        
        # API endpoints
        self.embedding_url = "https://api.openai.com/v1/embeddings"
        
        # Headers cho API requests
        self.headers = {
            "Authorization": f"Bearer {config.api_key}",
            "Content-Type": "application/json"
        }
    
    async def __aenter__(self):
        """Async context manager entry."""
        self.session = aiohttp.ClientSession()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        if self.session:
            await self.session.close()
    
    async def create_embedding(self, text: str) -> List[float]:
        """Tạo embedding cho một văn bản."""
        payload = {
            "model": self.config.model,
            "input": text,
            "encoding_format": "float"
        }
        
        if self.config.dimensions:
            payload["dimensions"] = self.config.dimensions
        
        for attempt in range(self.config.max_retries):
            try:
                async with self.session.post(
                    self.embedding_url,
                    headers=self.headers,
                    json=payload
                ) as response:
                    if response.status == 200:
                        result = await response.json()
                        return result["data"][0]["embedding"]
                    elif response.status == 429:  # Rate limit
                        wait_time = self.config.retry_delay * (2 ** attempt)
                        logger.warning(f"Rate limit hit, waiting {wait_time}s...")
                        await asyncio.sleep(wait_time)
                    else:
                        error_text = await response.text()
                        logger.error(f"API error {response.status}: {error_text}")
                        await asyncio.sleep(self.config.retry_delay)
                        
            except Exception as e:
                logger.error(f"Request failed (attempt {attempt + 1}): {e}")
                if attempt < self.config.max_retries - 1:
                    await asyncio.sleep(self.config.retry_delay * (2 ** attempt))
        
        raise Exception(f"Failed to create embedding after {self.config.max_retries} attempts")
    
    async def create_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
        """Tạo embeddings cho một batch các văn bản."""
        # Với OpenAI API, ta có thể gửi nhiều text cùng lúc
        payload = {
            "model": self.config.model,
            "input": texts,
            "encoding_format": "float"
        }
        
        if self.config.dimensions:
            payload["dimensions"] = self.config.dimensions
        
        for attempt in range(self.config.max_retries):
            try:
                async with self.session.post(
                    self.embedding_url,
                    headers=self.headers,
                    json=payload
                ) as response:
                    if response.status == 200:
                        result = await response.json()
                        return [item["embedding"] for item in result["data"]]
                    elif response.status == 429:  # Rate limit
                        wait_time = self.config.retry_delay * (2 ** attempt)
                        logger.warning(f"Rate limit hit, waiting {wait_time}s...")
                        await asyncio.sleep(wait_time)
                    else:
                        error_text = await response.text()
                        logger.error(f"API error {response.status}: {error_text}")
                        await asyncio.sleep(self.config.retry_delay)
                        
            except Exception as e:
                logger.error(f"Batch request failed (attempt {attempt + 1}): {e}")
                if attempt < self.config.max_retries - 1:
                    await asyncio.sleep(self.config.retry_delay * (2 ** attempt))
        
        raise Exception(f"Failed to create embeddings batch after {self.config.max_retries} attempts")


class DatasetEmbedder:
    """Class chính để embedding dataset."""
    
    def __init__(self, config: EmbeddingConfig):
        self.config = config
        self.embedder = OpenAIEmbedder(config)
    
    def load_jsonl(self, file_path: Path) -> List[Dict[str, Any]]:
        """Load dữ liệu từ file JSONL."""
        data = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    data.append(json.loads(line))
        return data
    
    def save_embedding(self, embedding: List[float], file_path: Path):
        """Lưu embedding vào file."""
        # Tạo thư mục nếu chưa tồn tại
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Lưu dưới dạng numpy array để tiết kiệm dung lượng
        np.save(file_path, np.array(embedding, dtype=np.float32))
    
    def load_embedding(self, file_path: Path) -> List[float]:
        """Load embedding từ file."""
        return np.load(file_path).tolist()
    
    async def embed_corpus(self, dataset_path: Path, output_dir: Path, force_recompute: bool = False):
        """Embedding corpus dataset."""
        corpus_file = dataset_path / "corpus.jsonl"
        if not corpus_file.exists():
            logger.error(f"Corpus file not found: {corpus_file}")
            return
        
        # Load corpus data
        logger.info(f"Loading corpus from {corpus_file}")
        corpus_data = self.load_jsonl(corpus_file)
        logger.info(f"Loaded {len(corpus_data)} documents")
        
        # Tạo thư mục output cho corpus embeddings
        corpus_output_dir = output_dir / "corpus"
        corpus_output_dir.mkdir(parents=True, exist_ok=True)
        
        # Embedding từng document
        async with self.embedder:
            # Xử lý theo batch để tối ưu tốc độ
            batch_size = min(self.config.batch_size, 100)  # OpenAI giới hạn khoảng 100 input per request
            
            for i in tqdm(range(0, len(corpus_data), batch_size), desc="Embedding corpus"):
                batch = corpus_data[i:i + batch_size]
                batch_texts = []
                batch_ids = []
                batch_files = []
                
                # Chuẩn bị batch
                for doc in batch:
                    doc_id = doc["doc_id"]
                    text = doc["text"]
                    
                    # File path để lưu embedding
                    embedding_file = corpus_output_dir / f"{doc_id}.npy"
                    
                    # Skip nếu file đã tồn tại và không force recompute
                    if embedding_file.exists() and not force_recompute:
                        continue
                    
                    batch_texts.append(text)
                    batch_ids.append(doc_id)
                    batch_files.append(embedding_file)
                
                # Skip batch nếu tất cả files đã tồn tại
                if not batch_texts:
                    continue
                
                try:
                    # Tạo embeddings cho batch
                    embeddings = await self.embedder.create_embeddings_batch(batch_texts)
                    
                    # Lưu từng embedding
                    for doc_id, embedding, file_path in zip(batch_ids, embeddings, batch_files):
                        self.save_embedding(embedding, file_path)
                        logger.debug(f"Saved embedding for document {doc_id}")
                
                except Exception as e:
                    logger.error(f"Failed to process corpus batch {i//batch_size + 1}: {e}")
                    # Fallback: xử lý từng document riêng lẻ
                    for doc in batch:
                        doc_id = doc["doc_id"]
                        text = doc["text"]
                        embedding_file = corpus_output_dir / f"{doc_id}.npy"
                        
                        if embedding_file.exists() and not force_recompute:
                            continue
                        
                        try:
                            embedding = await self.embedder.create_embedding(text)
                            self.save_embedding(embedding, embedding_file)
                            logger.debug(f"Saved embedding for document {doc_id}")
                        except Exception as e2:
                            logger.error(f"Failed to embed document {doc_id}: {e2}")
                
                # Thêm delay nhỏ để tránh rate limit
                await asyncio.sleep(0.1)
        
        logger.info(f"Completed corpus embedding. Results saved to {corpus_output_dir}")
    
    async def embed_queries(self, dataset_path: Path, output_dir: Path, force_recompute: bool = False):
        """Embedding queries dataset."""
        queries_file = dataset_path / "queries.jsonl"
        if not queries_file.exists():
            logger.error(f"Queries file not found: {queries_file}")
            return
        
        # Load queries data
        logger.info(f"Loading queries from {queries_file}")
        queries_data = self.load_jsonl(queries_file)
        logger.info(f"Loaded {len(queries_data)} queries")
        
        # Tạo thư mục output cho query embeddings
        queries_output_dir = output_dir / "queries"
        queries_output_dir.mkdir(parents=True, exist_ok=True)
        
        # Embedding từng query
        async with self.embedder:
            # Xử lý theo batch
            batch_size = min(self.config.batch_size, 100)
            
            for i in tqdm(range(0, len(queries_data), batch_size), desc="Embedding queries"):
                batch = queries_data[i:i + batch_size]
                batch_texts = []
                batch_ids = []
                batch_files = []
                
                # Chuẩn bị batch
                for query in batch:
                    query_id = query["query_id"]
                    text = query["text"]
                    
                    # File path để lưu embedding
                    embedding_file = queries_output_dir / f"{query_id}.npy"
                    
                    # Skip nếu file đã tồn tại và không force recompute
                    if embedding_file.exists() and not force_recompute:
                        continue
                    
                    batch_texts.append(text)
                    batch_ids.append(query_id)
                    batch_files.append(embedding_file)
                
                # Skip batch nếu tất cả files đã tồn tại
                if not batch_texts:
                    continue
                
                try:
                    # Tạo embeddings cho batch
                    embeddings = await self.embedder.create_embeddings_batch(batch_texts)
                    
                    # Lưu từng embedding
                    for query_id, embedding, file_path in zip(batch_ids, embeddings, batch_files):
                        self.save_embedding(embedding, file_path)
                        logger.debug(f"Saved embedding for query {query_id}")
                
                except Exception as e:
                    logger.error(f"Failed to process queries batch {i//batch_size + 1}: {e}")
                    # Fallback: xử lý từng query riêng lẻ
                    for query in batch:
                        query_id = query["query_id"]
                        text = query["text"]
                        embedding_file = queries_output_dir / f"{query_id}.npy"
                        
                        if embedding_file.exists() and not force_recompute:
                            continue
                        
                        try:
                            embedding = await self.embedder.create_embedding(text)
                            self.save_embedding(embedding, embedding_file)
                            logger.debug(f"Saved embedding for query {query_id}")
                        except Exception as e2:
                            logger.error(f"Failed to embed query {query_id}: {e2}")
                
                # Thêm delay nhỏ để tránh rate limit
                await asyncio.sleep(0.1)
        
        logger.info(f"Completed queries embedding. Results saved to {queries_output_dir}")
    
    async def embed_dataset(self, dataset_path: Path, output_dir: Path, force_recompute: bool = False):
        """Embedding toàn bộ dataset (corpus + queries)."""
        logger.info(f"Starting embedding for dataset: {dataset_path.name}")
        
        # Tạo thư mục theo model name
        model_output_dir = output_dir / self.config.model / dataset_path.name
        model_output_dir.mkdir(parents=True, exist_ok=True)
        
        # Embedding corpus
        await self.embed_corpus(dataset_path, model_output_dir, force_recompute)
        
        # Embedding queries
        await self.embed_queries(dataset_path, model_output_dir, force_recompute)
        
        # Lưu metadata
        metadata = {
            "dataset_name": dataset_path.name,
            "embedding_model": self.config.model,
            "dimensions": self.config.dimensions,
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "api_version": "2024-02-01"
        }
        
        metadata_file = model_output_dir / "embedding_metadata.json"
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Completed embedding for dataset: {dataset_path.name}")


def main():
    parser = argparse.ArgumentParser(description="Embedding dataset using OpenAI API")
    parser.add_argument("--dataset", help="Specific dataset to embed (e.g., tydiqa_goldp_vietnamese)")
    parser.add_argument("--data-dir", default="../data", help="Directory containing datasets")
    parser.add_argument("--output-dir", default="../embeddings", help="Output directory for embeddings")
    parser.add_argument("--model", default="text-embedding-3-small", 
                       choices=["text-embedding-3-small", "text-embedding-3-large"],
                       help="OpenAI embedding model to use")
    parser.add_argument("--dimensions", type=int, help="Embedding dimensions (optional)")
    parser.add_argument("--batch-size", type=int, default=50, help="Batch size for processing")
    parser.add_argument("--force-recompute", action="store_true", 
                       help="Force recompute embeddings even if files exist")
    parser.add_argument("--max-retries", type=int, default=3, help="Max retries for API calls")
    
    args = parser.parse_args()
    
    # Đọc OpenAI API key từ environment variable
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        logger.error("OPENAI_API_KEY environment variable is not set")
        logger.error("Please set it with: export OPENAI_API_KEY='your-api-key-here'")
        return
    
    # Tạo config
    config = EmbeddingConfig(
        api_key=api_key,
        model=args.model,
        dimensions=args.dimensions,
        batch_size=args.batch_size,
        max_retries=args.max_retries
    )
    
    # Tạo embedder
    embedder = DatasetEmbedder(config)
    
    # Paths
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    
    async def run_embedding():
        if args.dataset:
            # Embed specific dataset
            dataset_path = data_dir / args.dataset
            if not dataset_path.exists():
                logger.error(f"Dataset not found: {dataset_path}")
                return
            
            await embedder.embed_dataset(dataset_path, output_dir, args.force_recompute)
        else:
            # Embed all datasets
            datasets = []
            for item in data_dir.iterdir():
                if item.is_dir() and (item / "corpus.jsonl").exists():
                    datasets.append(item)
            
            if not datasets:
                logger.error("No valid datasets found in data directory")
                return
            
            logger.info(f"Found {len(datasets)} datasets to embed")
            
            for dataset_path in datasets:
                try:
                    await embedder.embed_dataset(dataset_path, output_dir, args.force_recompute)
                except Exception as e:
                    logger.error(f"Failed to embed dataset {dataset_path.name}: {e}")
    
    # Chạy embedding
    asyncio.run(run_embedding())


if __name__ == "__main__":
    main()
