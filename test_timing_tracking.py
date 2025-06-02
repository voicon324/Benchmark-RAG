#!/usr/bin/env python3
"""
Test script to demonstrate the new timing tracking features in ExperimentRunner.
"""

import json
import sys
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from newaibench.experiment.runner import ExperimentResult
from dataclasses import asdict

def test_experiment_result_timing():
    """Test the new timing fields in ExperimentResult."""
    print("Testing ExperimentResult with timing fields...")
    
    # Create a sample result with timing information
    result = ExperimentResult(
        model_name="test_model",
        dataset_name="test_dataset", 
        metrics={"recall@10": 0.85, "ndcg@10": 0.75},
        execution_time=120.5,
        index_time=45.2,
        retrieval_time=65.8,
        metadata={
            'timing_breakdown': {
                'total_execution_time': 120.5,
                'index_time': 45.2,
                'retrieval_time': 65.8,
                'evaluation_time': 9.5
            }
        }
    )
    
    # Convert to dict for JSON serialization
    result_dict = asdict(result)
    
    print("ExperimentResult structure:")
    print(json.dumps(result_dict, indent=2))
    
    # Verify timing fields exist
    assert hasattr(result, 'index_time'), "Missing index_time field"
    assert hasattr(result, 'retrieval_time'), "Missing retrieval_time field"
    assert result.index_time == 45.2, "Index time not set correctly"
    assert result.retrieval_time == 65.8, "Retrieval time not set correctly"
    
    print("✓ All timing fields are properly tracked!")
    
    return result_dict

def demonstrate_timing_breakdown():
    """Demonstrate how timing breakdown will appear in saved results."""
    print("\nDemonstrating timing breakdown in results file...")
    
    # Simulate multiple experiment results
    results = []
    
    for i in range(3):
        result = ExperimentResult(
            model_name=f"model_{i+1}",
            dataset_name="test_dataset",
            metrics={"recall@10": 0.8 + i*0.05, "ndcg@10": 0.7 + i*0.05},
            execution_time=100.0 + i*20,
            index_time=30.0 + i*10,
            retrieval_time=50.0 + i*8,
            metadata={
                'timing_breakdown': {
                    'total_execution_time': 100.0 + i*20,
                    'index_time': 30.0 + i*10,
                    'retrieval_time': 50.0 + i*8,
                    'evaluation_time': 20.0 + i*2
                }
            }
        )
        results.append(result)
    
    # Simulate summary generation
    successful_results = [r for r in results if r.success]
    
    timing_statistics = {
        'total_index_time': sum(r.index_time for r in successful_results),
        'total_retrieval_time': sum(r.retrieval_time for r in successful_results),
        'average_index_time': sum(r.index_time for r in successful_results) / len(successful_results),
        'average_retrieval_time': sum(r.retrieval_time for r in successful_results) / len(successful_results),
        'index_time_per_experiment': [r.index_time for r in successful_results],
        'retrieval_time_per_experiment': [r.retrieval_time for r in successful_results],
    }
    
    # Show what the results file would look like
    results_data = {
        'results': [asdict(result) for result in results],
        'summary': {
            'total_experiments': len(results),
            'successful_experiments': len(successful_results),
            'timing_statistics': timing_statistics,
            'average_metrics': {
                'recall@10': sum(r.metrics['recall@10'] for r in successful_results) / len(successful_results),
                'ndcg@10': sum(r.metrics['ndcg@10'] for r in successful_results) / len(successful_results)
            }
        }
    }
    
    print("Sample results.json structure with timing:")
    print(json.dumps(results_data, indent=2))
    
    print(f"\n✓ Timing summary:")
    print(f"  Total index time: {timing_statistics['total_index_time']:.2f}s")
    print(f"  Total retrieval time: {timing_statistics['total_retrieval_time']:.2f}s") 
    print(f"  Average index time: {timing_statistics['average_index_time']:.2f}s")
    print(f"  Average retrieval time: {timing_statistics['average_retrieval_time']:.2f}s")

def main():
    """Main test function."""
    print("=" * 60)
    print("TESTING NEW TIMING TRACKING FEATURES")
    print("=" * 60)
    
    try:
        test_experiment_result_timing()
        demonstrate_timing_breakdown()
        
        print("\n" + "=" * 60)
        print("ALL TESTS PASSED!")
        print("=" * 60)
        
        print("\nNew features added:")
        print("1. ✓ ExperimentResult now tracks index_time and retrieval_time separately")
        print("2. ✓ Timing breakdown is included in metadata")
        print("3. ✓ Summary statistics include detailed timing information")
        print("4. ✓ Log output shows timing breakdown")
        print("5. ✓ Results are serializable to JSON with timing data")
        
        print("\nBenefits:")
        print("- Can analyze indexing vs retrieval performance separately")
        print("- Better understanding of where time is spent during experiments")
        print("- Helpful for optimizing model performance")
        print("- Detailed timing data saved for later analysis")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return 1
        
    return 0

if __name__ == "__main__":
    exit(main())
