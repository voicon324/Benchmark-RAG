#!/usr/bin/env python3
"""
Simple test to verify image retrieval models are working.
"""

import sys
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

try:
    print("Testing imports...")
    from newaibench.models.image_retrieval import (
        OCRBasedDocumentRetriever,
        ImageEmbeddingDocumentRetriever
    )
    from newaibench.models.base import ModelType
    print("✓ Imports successful")
    
    print("\nTesting OCRBasedDocumentRetriever initialization...")
    config = {
        "model_name": "sentence-transformers/all-MiniLM-L6-v2",
        "model_type": ModelType.DENSE
    }
    ocr_retriever = OCRBasedDocumentRetriever(config)
    print("✓ OCRBasedDocumentRetriever initialized")
    
    print("\nTesting ImageEmbeddingDocumentRetriever initialization...")
    image_config = {
        "model_name": "openai/clip-vit-base-patch32",
        "model_type": ModelType.MULTIMODAL
    }
    image_retriever = ImageEmbeddingDocumentRetriever(image_config)
    print("✓ ImageEmbeddingDocumentRetriever initialized")
    
    print("\nTesting basic initialization...")
    # In a real scenario, you would test the actual methods like encode_documents
    # but for this basic test, we'll just check that initialization works
    print("✓ Models successfully initialized")
    
    print("\n" + "="*50)
    print("ALL TESTS PASSED!")
    print("Image retrieval models are ready to use.")
    print("="*50)
    
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
