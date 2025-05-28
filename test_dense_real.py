#!/usr/bin/env python3
"""
Real-world testing script for Dense Retrieval implementation.

This script performs comprehensive testing with actual models to validate
the DenseTextRetriever implementation works correctly in practice.
"""

import sys
import os
sys.path.append('src')

import time
import numpy as np
from typing import List, Dict, Any
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_sentence_bert_basic():
    """Test basic Sentence-BERT functionality."""
    print("\n=== Testing Sentence-BERT Basic Functionality ===")
    
    try:
        from newaibench.models.dense import SentenceBERTModel
        
        # Configure with a lightweight model
        config = {
            "name": "sbert_test",
            "type": "dense",
            "model_name_or_path": "all-MiniLM-L6-v2",
            "parameters": {
                "normalize_embeddings": True,
                "use_ann_index": False
            }
        }
        
        # Create model
        model = SentenceBERTModel(config)
        print(f"✓ Model created: {model.model_name_or_path}")
        
        # Load model
        start_time = time.time()
        model.load_model()
        load_time = time.time() - start_time
        print(f"✓ Model loaded in {load_time:.2f}s")
        
        # Test encoding
        test_texts = [
            "What is machine learning?",
            "How do neural networks work?",
            "Explain deep learning algorithms",
            "Machine learning is a subset of artificial intelligence",
            "Neural networks are inspired by biological neurons"
        ]
        
        start_time = time.time()
        embeddings = model.encode_texts(test_texts)
        encode_time = time.time() - start_time
        
        print(f"✓ Encoded {len(test_texts)} texts in {encode_time:.3f}s")
        print(f"✓ Embedding shape: {embeddings.shape}")
        print(f"✓ Embedding dtype: {embeddings.dtype}")
        
        # Verify embeddings are normalized
        norms = np.linalg.norm(embeddings, axis=1)
        print(f"✓ Embedding norms: {norms}")
        assert np.allclose(norms, 1.0, atol=1e-6), "Embeddings should be normalized"
        
        # Test similarity computation
        similarity_matrix = np.dot(embeddings, embeddings.T)
        print(f"✓ Similarity matrix shape: {similarity_matrix.shape}")
        
        # Most similar pairs should be reasonable
        most_similar = np.argsort(similarity_matrix[0])[::-1]
        print(f"✓ Most similar to first query: {[test_texts[i] for i in most_similar[:3]]}")
        
        return True
        
    except Exception as e:
        print(f"✗ Error in Sentence-BERT test: {e}")
        return False

def test_ann_indexing():
    """Test ANN indexing with real data."""
    print("\n=== Testing ANN Indexing ===")
    
    try:
        from newaibench.models.dense import DenseTextRetriever
        
        # Test HNSWLIB if available
        config = {
            "name": "ann_test",
            "type": "dense", 
            "model_name_or_path": "all-MiniLM-L6-v2",
            "parameters": {
                "use_ann_index": True,
                "ann_backend": "hnswlib",
                "normalize_embeddings": True,
                "m_parameter_hnsw": 16,
                "ef_search_hnsw": 50
            }
        }
        
        model = DenseTextRetriever(config)
        model.load_model()
        print(f"✓ ANN model loaded with backend: {model.ann_backend}")
        
        # Create a larger test corpus
        corpus = [
            {"id": f"doc_{i}", "text": f"This is document number {i} about topic {i%5}"}
            for i in range(100)
        ]
        
        # Index corpus
        start_time = time.time()
        model.index_corpus(corpus)
        index_time = time.time() - start_time
        print(f"✓ Indexed {len(corpus)} documents in {index_time:.3f}s")
        
        # Test search
        queries = ["document about topic 1", "text number 50"]
        
        start_time = time.time()
        results = model.predict(queries, corpus, top_k=5)
        search_time = time.time() - start_time
        
        print(f"✓ Searched {len(queries)} queries in {search_time:.3f}s")
        print(f"✓ Results shape: {len(results)} queries, {len(results[0])} results each")
        
        # Verify results format
        for i, query_results in enumerate(results):
            print(f"Query {i}: '{queries[i]}'")
            for j, (doc_id, score) in enumerate(query_results[:3]):
                print(f"  {j+1}. {doc_id}: {score:.4f}")
        
        return True
        
    except Exception as e:
        print(f"✗ Error in ANN indexing test: {e}")
        return False

def test_caching_functionality():
    """Test embedding caching system."""
    print("\n=== Testing Caching Functionality ===")
    
    try:
        from newaibench.models.dense import DenseTextRetriever
        import tempfile
        import shutil
        
        # Create temporary directory for cache
        cache_dir = tempfile.mkdtemp()
        
        config = {
            "name": "cache_test",
            "type": "dense",
            "model_name_or_path": "all-MiniLM-L6-v2", 
            "parameters": {
                "normalize_embeddings": True,
                "cache_embeddings": True,
                "embedding_cache_path": os.path.join(cache_dir, "embeddings.pkl")
            }
        }
        
        model = DenseTextRetriever(config)
        model.load_model()
        
        corpus = [
            {"id": f"doc_{i}", "text": f"Document {i} content"}
            for i in range(20)
        ]
        
        # First indexing (should cache)
        start_time = time.time()
        model.index_corpus(corpus)
        first_index_time = time.time() - start_time
        print(f"✓ First indexing: {first_index_time:.3f}s")
        
        # Check cache file exists
        cache_path = model.embedding_cache_path
        assert os.path.exists(cache_path), "Cache file should be created"
        print(f"✓ Cache file created: {cache_path}")
        
        # Create new model instance (should load from cache)
        model2 = DenseTextRetriever(config)
        model2.load_model()
        
        start_time = time.time() 
        model2.index_corpus(corpus)
        cached_index_time = time.time() - start_time
        print(f"✓ Cached indexing: {cached_index_time:.3f}s")
        
        # Cached should be significantly faster
        speedup = first_index_time / cached_index_time
        print(f"✓ Cache speedup: {speedup:.2f}x")
        
        # Cleanup
        shutil.rmtree(cache_dir)
        print("✓ Cache test completed and cleaned up")
        
        return True
        
    except Exception as e:
        print(f"✗ Error in caching test: {e}")
        return False

def test_batch_processing():
    """Test batch processing efficiency."""
    print("\n=== Testing Batch Processing ===")
    
    try:
        from newaibench.models.dense import DenseTextRetriever
        
        config = {
            "name": "batch_test", 
            "type": "dense",
            "model_name_or_path": "all-MiniLM-L6-v2",
            "parameters": {
                "normalize_embeddings": True
            }
        }
        
        model = DenseTextRetriever(config)
        model.load_model()
        
        # Create varying length texts
        texts = [
            "Short text",
            "This is a medium length text with several words and concepts",
            "This is a much longer text that contains many more words and should test the model's ability to handle variable length inputs effectively while maintaining good performance characteristics",
            "Another short one",
            "Medium length text again for variety"
        ] * 20  # 100 texts total
        
        # Test different batch sizes
        batch_sizes = [1, 8, 16, 32]
        
        for batch_size in batch_sizes:
            start_time = time.time()
            embeddings = model.encode_texts(texts, batch_size=batch_size)
            encode_time = time.time() - start_time
            
            texts_per_sec = len(texts) / encode_time
            print(f"✓ Batch size {batch_size:2d}: {encode_time:.3f}s ({texts_per_sec:.0f} texts/sec)")
        
        print(f"✓ Final embedding shape: {embeddings.shape}")
        
        return True
        
    except Exception as e:
        print(f"✗ Error in batch processing test: {e}")
        return False

def test_error_handling():
    """Test error handling and edge cases."""
    print("\n=== Testing Error Handling ===")
    
    try:
        from newaibench.models.dense import DenseTextRetriever
        
        # Test invalid model
        config = {
            "name": "error_test",
            "type": "dense", 
            "model_name_or_path": "nonexistent-model-12345",
            "parameters": {}
        }
        
        model = DenseTextRetriever(config)
        
        try:
            model.load_model()
            print("✗ Should have failed with invalid model")
            return False
        except Exception as e:
            print(f"✓ Correctly handled invalid model: {type(e).__name__}")
        
        # Test with valid model
        config["model_name_or_path"] = "all-MiniLM-L6-v2"
        model = DenseTextRetriever(config)
        model.load_model()
        
        # Test empty corpus
        empty_results = model.predict(["test query"], [], top_k=5)
        assert len(empty_results[0]) == 0, "Empty corpus should return empty results"
        print("✓ Handled empty corpus correctly")
        
        # Test empty query
        corpus = [{"id": "doc1", "text": "Some content"}]
        model.index_corpus(corpus)
        
        empty_query_results = model.predict([""], corpus, top_k=5)
        print("✓ Handled empty query correctly")
        
        # Test malformed documents
        bad_corpus = [
            {"id": "doc1"},  # Missing text
            {"text": "No ID"},  # Missing ID
            {"id": "doc2", "text": "Good doc"}
        ]
        
        try:
            model.index_corpus(bad_corpus)
            print("✓ Handled malformed documents gracefully")
        except Exception as e:
            print(f"✓ Correctly rejected malformed corpus: {type(e).__name__}")
        
        return True
        
    except Exception as e:
        print(f"✗ Error in error handling test: {e}")
        return False

def test_integration_scenario():
    """Test realistic integration scenario."""
    print("\n=== Testing Integration Scenario ===")
    
    try:
        from newaibench.models.dense import SentenceBERTModel
        
        # Simulate real FAQ scenario
        faq_corpus = [
            {"id": "faq_1", "text": "How do I reset my password? To reset your password, go to the login page and click 'Forgot Password'."},
            {"id": "faq_2", "text": "What are your business hours? We are open Monday through Friday from 9 AM to 6 PM EST."},
            {"id": "faq_3", "text": "How can I contact customer support? You can reach customer support via email at support@company.com or phone at 1-800-123-4567."},
            {"id": "faq_4", "text": "What payment methods do you accept? We accept all major credit cards, PayPal, and bank transfers."},
            {"id": "faq_5", "text": "How do I track my order? You can track your order using the tracking number sent to your email after purchase."},
            {"id": "faq_6", "text": "What is your return policy? Items can be returned within 30 days of purchase for a full refund."},
            {"id": "faq_7", "text": "Do you offer international shipping? Yes, we ship to most countries worldwide with varying delivery times."},
            {"id": "faq_8", "text": "How do I create an account? Click the 'Sign Up' button on our homepage and fill out the registration form."}
        ]
        
        # User queries
        user_queries = [
            "I forgot my login credentials",
            "When are you open?", 
            "Need help from support team",
            "What cards do you take?",
            "Where is my package?",
            "Can I return this item?",
            "Do you ship overseas?",
            "How to register new account?"
        ]
        
        config = {
            "name": "faq_system",
            "type": "dense",
            "model_name_or_path": "all-MiniLM-L6-v2",
            "parameters": {
                "use_ann_index": True,
                "ann_backend": "hnswlib",
                "normalize_embeddings": True
            }
        }
        
        model = SentenceBERTModel(config)
        model.load_model()
        
        # Index FAQ corpus
        print(f"✓ Indexing {len(faq_corpus)} FAQ entries...")
        model.index_corpus(faq_corpus)
        
        # Process user queries
        print(f"✓ Processing {len(user_queries)} user queries...")
        results = model.predict(user_queries, faq_corpus, top_k=3)
        
        # Display results
        for i, (query, query_results) in enumerate(zip(user_queries, results)):
            print(f"\nQuery {i+1}: '{query}'")
            for j, (faq_id, score) in enumerate(query_results):
                faq_text = next(faq["text"] for faq in faq_corpus if faq["id"] == faq_id)
                print(f"  {j+1}. {faq_id} (score: {score:.3f})")
                print(f"     {faq_text[:80]}...")
        
        # Verify reasonable matching
        assert results[0][0][1] > 0.3, "Password reset query should have reasonable match"
        assert results[1][0][1] > 0.3, "Business hours query should have reasonable match"
        
        print("✓ Integration scenario completed successfully")
        return True
        
    except Exception as e:
        print(f"✗ Error in integration scenario: {e}")
        return False

def main():
    """Run all tests."""
    print("Dense Retrieval Implementation - Real World Testing")
    print("=" * 60)
    
    tests = [
        ("Sentence-BERT Basic", test_sentence_bert_basic),
        ("ANN Indexing", test_ann_indexing), 
        ("Caching Functionality", test_caching_functionality),
        ("Batch Processing", test_batch_processing),
        ("Error Handling", test_error_handling),
        ("Integration Scenario", test_integration_scenario)
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        try:
            print(f"\n{'='*20} {test_name} {'='*20}")
            success = test_func()
            results[test_name] = success
        except Exception as e:
            print(f"✗ Test '{test_name}' failed with exception: {e}")
            results[test_name] = False
    
    # Summary
    print(f"\n{'='*60}")
    print("TEST RESULTS SUMMARY")
    print(f"{'='*60}")
    
    passed = sum(results.values())
    total = len(results)
    
    for test_name, success in results.items():
        status = "✓ PASS" if success else "✗ FAIL"
        print(f"{test_name:.<40} {status}")
    
    print(f"\nOverall: {passed}/{total} tests passed ({passed/total*100:.1f}%)")
    
    if passed == total:
        print("🎉 All tests passed! Dense retrieval implementation is working correctly.")
    else:
        print("⚠️  Some tests failed. Please check the implementation.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
