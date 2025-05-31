#!/usr/bin/env python3
"""
Quick test script for the new encoding parameters feature.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from newaibench.models.dense import DenseTextRetriever

def test_encode_params():
    """Test the new encoding parameters feature."""
    print("Testing Dense Model Encoding Parameters Feature")
    print("=" * 50)
    
    # Test configuration with encoding parameters
    config_dict = {
        "name": "test_model",
        "type": "dense",
        "model_name_or_path": "all-MiniLM-L6-v2",
        "parameters": {
            "model_architecture": "sentence_transformer",
            "normalize_embeddings": True,
            "max_seq_length": 256,
            "batch_size": 2,
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
    
    print("1. Creating model with encoding parameters...")
    model = DenseTextRetriever(config_dict)
    
    print("2. Loading model...")
    model.load_model()
    
    print("3. Testing query encoding with config parameters...")
    queries = ["What is machine learning?", "How does AI work?"]
    query_embeddings = model.encode_texts(queries, is_query=True, show_progress=False)
    print(f"   Query embeddings shape: {query_embeddings.shape}")
    
    print("4. Testing document encoding with config parameters...")
    documents = ["Machine learning is a subset of AI", "AI uses algorithms to solve problems"]
    doc_embeddings = model.encode_texts(documents, is_query=False, show_progress=False)
    print(f"   Document embeddings shape: {doc_embeddings.shape}")
    
    print("5. Testing with additional runtime parameters...")
    additional_params = {"prompt_name": "search"}
    query_embeddings_2 = model.encode_texts(
        queries, 
        is_query=True, 
        show_progress=False,
        additional_encode_params=additional_params
    )
    print(f"   Query embeddings with additional params shape: {query_embeddings_2.shape}")
    
    print("6. Testing encode_queries method...")
    query_dicts = [
        {"query_id": "q1", "text": "What is machine learning?"},
        {"query_id": "q2", "text": "How does AI work?"}
    ]
    query_embeddings_dict = model.encode_queries(
        query_dicts, 
        additional_encode_params={"prompt_name": "search_query"}
    )
    print(f"   Query embeddings dict keys: {list(query_embeddings_dict.keys())}")
    
    print("7. Testing encode_documents method...")
    doc_dicts = {
        "doc1": {"text": "Machine learning is a subset of AI"},
        "doc2": {"text": "AI uses algorithms to solve problems"}
    }
    doc_embeddings_dict = model.encode_documents(
        doc_dicts,
        additional_encode_params={"prompt_name": "document"}
    )
    print(f"   Document embeddings dict keys: {list(doc_embeddings_dict.keys())}")
    
    print("\n✅ All tests completed successfully!")
    print(f"   Config query params: {model.query_encode_params}")
    print(f"   Config document params: {model.document_encode_params}")

if __name__ == "__main__":
    try:
        test_encode_params()
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
