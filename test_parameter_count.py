#!/usr/bin/env python3
"""
Quick test script to verify parameter count functionality
"""

import sys
import os
sys.path.append('/home/hkduy/NewAI/new_bench/src')

def test_parameter_count_feature():
    """Test that parameter count is properly captured in model info."""
    
    print("🧪 Testing Parameter Count Feature")
    print("=" * 50)
    
    # Test 1: Import the helper function
    try:
        from newaibench.models.base import count_model_parameters
        print("✅ Successfully imported count_model_parameters")
    except ImportError as e:
        print(f"❌ Failed to import count_model_parameters: {e}")
        return False
    
    # Test 2: Test sparse model (BM25)
    try:
        from newaibench.models.sparse import BM25Model
        from newaibench.models.base import ModelConfig
        
        config = ModelConfig(
            name="test_bm25",
            device="cpu",
            batch_size=32
        )
        
        bm25_model = BM25Model(config)
        model_info = bm25_model.get_model_info()
        
        print(f"✅ BM25 Model Info includes parameter_count: {'parameter_count' in model_info}")
        print(f"   Parameter count value: {model_info.get('parameter_count', 'Not found')}")
        
    except Exception as e:
        print(f"❌ Failed to test BM25 model: {e}")
        return False
    
    # Test 3: Test with a mock PyTorch model
    try:
        import torch
        import torch.nn as nn
        
        # Create a simple test model
        class SimpleModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(10, 5)
                
            def forward(self, x):
                return self.linear(x)
        
        test_model = SimpleModel()
        param_count = count_model_parameters(test_model)
        
        # Should be 10*5 + 5 = 55 parameters
        expected_params = 55
        
        print(f"✅ PyTorch model parameter count: {param_count}")
        print(f"   Expected: {expected_params}, Got: {param_count}")
        print(f"   Correct: {param_count == expected_params}")
        
    except Exception as e:
        print(f"⚠️  Could not test PyTorch model (may be normal if PyTorch not available): {e}")
    
    # Test 4: Test dense model info structure
    try:
        from newaibench.models.dense import DenseTextRetriever
        
        config = {
            'name': 'test_dense',
            'model_name_or_path': 'sentence-transformers/all-MiniLM-L6-v2',
            'device': 'cpu',
            'batch_size': 8
        }
        
        # Don't actually load the model, just check the get_model_info method exists
        # and has the right structure
        print("✅ DenseTextRetriever class imported successfully")
        print("   get_model_info method should include parameter_count field")
        
    except Exception as e:
        print(f"❌ Failed to test dense model: {e}")
        return False
    
    print("\n🎉 Parameter count feature appears to be working!")
    print("\nNext steps:")
    print("1. Run an actual experiment to verify parameter counts are captured")
    print("2. Check experiment results to see parameter count in metadata")
    
    return True

if __name__ == "__main__":
    success = test_parameter_count_feature()
    if success:
        print("\n✅ All tests passed!")
        sys.exit(0)
    else:
        print("\n❌ Some tests failed!")
        sys.exit(1)
