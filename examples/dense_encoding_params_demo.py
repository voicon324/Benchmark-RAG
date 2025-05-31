#!/usr/bin/env python3
"""
Demo script showing how to use additional encoding parameters with dense retrieval models.

This script demonstrates:
1. Basic usage without encoding parameters
2. Using encoding parameters from config
3. Passing additional encoding parameters at runtime
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from newaibench.models.dense import DenseTextRetriever
from newaibench.core.config import ModelConfig
import numpy as np


def demo_basic_usage():
    """Demo basic usage without encoding parameters."""
    print("=== Demo 1: Basic Usage (No Encoding Parameters) ===")
    
    config_dict = {
        "name": "basic_demo",
        "type": "dense",
        "model_name_or_path": "all-MiniLM-L6-v2",
        "parameters": {
            "model_architecture": "sentence_transformer",
            "normalize_embeddings": True,
            "max_seq_length": 512
        }
    }
    
    model_config = ModelConfig(**config_dict)
    model = DenseTextRetriever(config_dict)
    model.load_model()
    
    # Test texts
    queries = ["What is machine learning?", "How does AI work?"]
    documents = ["Machine learning is a subset of AI", "Artificial intelligence uses algorithms"]
    
    # Encode without special parameters
    query_embeddings = model.encode_texts(queries, is_query=True)
    doc_embeddings = model.encode_texts(documents, is_query=False)
    
    print(f"Query embeddings shape: {query_embeddings.shape}")
    print(f"Document embeddings shape: {doc_embeddings.shape}")
    print()


def demo_config_parameters():
    """Demo using encoding parameters from config."""
    print("=== Demo 2: Using Encoding Parameters from Config ===")
    
    config_dict = {
        "name": "config_params_demo",
        "type": "dense", 
        "model_name_or_path": "all-MiniLM-L6-v2",
        "parameters": {
            "model_architecture": "sentence_transformer",
            "normalize_embeddings": True,
            "max_seq_length": 512,
            # Query encoding parameters
            "query_encode_params": {
                "task": "retrieval.query",
                "prompt_name": "query"
            },
            # Document encoding parameters
            "document_encode_params": {
                "task": "retrieval.passage", 
                "prompt_name": "passage"
            }
        }
    }
    
    model_config = ModelConfig(**config_dict)
    model = DenseTextRetriever(config_dict)
    model.load_model()
    
    # Test texts
    queries = ["What is machine learning?", "How does AI work?"]
    documents = ["Machine learning is a subset of AI", "Artificial intelligence uses algorithms"]
    
    # Encode with config parameters (automatically applied)
    query_embeddings = model.encode_texts(queries, is_query=True)
    doc_embeddings = model.encode_texts(documents, is_query=False)
    
    print(f"Query embeddings shape: {query_embeddings.shape}")
    print(f"Document embeddings shape: {doc_embeddings.shape}")
    print(f"Query encode params used: {model.query_encode_params}")
    print(f"Document encode params used: {model.document_encode_params}")
    print()


def demo_runtime_parameters():
    """Demo passing additional encoding parameters at runtime."""
    print("=== Demo 3: Additional Runtime Encoding Parameters ===")
    
    config_dict = {
        "name": "runtime_params_demo",
        "type": "dense",
        "model_name_or_path": "all-MiniLM-L6-v2", 
        "parameters": {
            "model_architecture": "sentence_transformer",
            "normalize_embeddings": True,
            "max_seq_length": 512,
            # Base config parameters
            "query_encode_params": {
                "task": "retrieval.query"
            }
        }
    }
    
    model_config = ModelConfig(**config_dict)
    model = DenseTextRetriever(config_dict)
    model.load_model()
    
    # Test texts
    queries = ["What is machine learning?", "How does AI work?"]
    documents = ["Machine learning is a subset of AI", "Artificial intelligence uses algorithms"]
    
    # Encode with additional runtime parameters
    additional_query_params = {
        "prompt_name": "search_query",
        "instruction": "Represent this query for semantic search:"
    }
    
    additional_doc_params = {
        "prompt_name": "document", 
        "instruction": "Represent this document for retrieval:"
    }
    
    query_embeddings = model.encode_texts(
        queries, 
        is_query=True,
        additional_encode_params=additional_query_params
    )
    
    doc_embeddings = model.encode_texts(
        documents,
        is_query=False, 
        additional_encode_params=additional_doc_params
    )
    
    print(f"Query embeddings shape: {query_embeddings.shape}")
    print(f"Document embeddings shape: {doc_embeddings.shape}")
    print(f"Additional query params: {additional_query_params}")
    print(f"Additional doc params: {additional_doc_params}")
    print()


def demo_encode_queries_documents():
    """Demo using encode_queries and encode_documents methods with parameters."""
    print("=== Demo 4: Using encode_queries/encode_documents with Parameters ===")
    
    config_dict = {
        "name": "encode_methods_demo",
        "type": "dense",
        "model_name_or_path": "all-MiniLM-L6-v2",
        "parameters": {
            "model_architecture": "sentence_transformer",
            "normalize_embeddings": True,
            "query_encode_params": {
                "task": "retrieval.query",
                "prompt_name": "query"
            },
            "document_encode_params": {
                "task": "retrieval.passage"
            }
        }
    }
    
    model = DenseTextRetriever(config_dict)
    model.load_model()
    
    # Prepare data in expected format
    queries = [
        {"query_id": "q1", "text": "What is machine learning?"},
        {"query_id": "q2", "text": "How does AI work?"}
    ]
    
    documents = {
        "doc1": {"text": "Machine learning is a subset of AI"},
        "doc2": {"text": "Artificial intelligence uses algorithms"}
    }
    
    # Encode using the methods with additional parameters
    query_embeddings = model.encode_queries(
        queries, 
        additional_encode_params={"prompt_name": "search"}
    )
    
    doc_embeddings = model.encode_documents(
        documents,
        additional_encode_params={"prompt_name": "passage"}
    )
    
    print(f"Query embeddings: {list(query_embeddings.keys())}")
    print(f"Document embeddings: {list(doc_embeddings.keys())}")
    print(f"Query embedding shape: {next(iter(query_embeddings.values())).shape}")
    print(f"Document embedding shape: {next(iter(doc_embeddings.values())).shape}")
    print()


if __name__ == "__main__":
    print("Dense Model Encoding Parameters Demo")
    print("=" * 50)
    
    try:
        demo_basic_usage()
        demo_config_parameters() 
        demo_runtime_parameters()
        demo_encode_queries_documents()
        
        print("✅ All demos completed successfully!")
        
    except Exception as e:
        print(f"❌ Demo failed with error: {e}")
        import traceback
        traceback.print_exc()
