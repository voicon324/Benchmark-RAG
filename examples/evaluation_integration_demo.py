"""
Integration example demonstrating how to use NewAIBench evaluation module
with dataset loaders and models for complete IR evaluation pipeline.
"""

import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple

# Import NewAIBench components
from newaibench import (
    TextDatasetLoader,
    SparseRetrievalModel, 
    Evaluator,
    BatchEvaluator,
    EvaluationConfig,
    quick_evaluate
)


def create_sample_dataset():
    """Create a sample IR dataset for demonstration."""
    # Sample queries
    queries = {
        'q1': 'machine learning algorithms',
        'q2': 'deep neural networks',
        'q3': 'natural language processing'
    }
    
    # Sample documents corpus
    documents = {
        'doc1': 'Machine learning is a subset of artificial intelligence that focuses on algorithms.',
        'doc2': 'Deep learning uses neural networks with multiple layers to model complex patterns.',
        'doc3': 'Natural language processing enables computers to understand human language.',
        'doc4': 'Supervised learning algorithms learn from labeled training data.',
        'doc5': 'Unsupervised learning finds hidden patterns in data without labels.',
        'doc6': 'Reinforcement learning agents learn through interaction with environment.'
    }
    
    # Sample relevance judgments (qrels)
    qrels = {
        'q1': {'doc1': 3, 'doc4': 2, 'doc5': 2, 'doc6': 1, 'doc2': 0, 'doc3': 0},
        'q2': {'doc2': 3, 'doc1': 1, 'doc4': 1, 'doc3': 0, 'doc5': 0, 'doc6': 0},
        'q3': {'doc3': 3, 'doc1': 1, 'doc2': 1, 'doc4': 0, 'doc5': 0, 'doc6': 0}
    }
    
    return queries, documents, qrels


def demonstrate_basic_evaluation():
    """Demonstrate basic evaluation with quick_evaluate function."""
    print("=== Basic Evaluation Demo ===")
    
    queries, documents, qrels = create_sample_dataset()
    
    # Simulate model results (normally these would come from your retrieval model)
    model_results = {
        'q1': [('doc1', 0.95), ('doc4', 0.85), ('doc2', 0.75), ('doc5', 0.65), ('doc6', 0.55), ('doc3', 0.45)],
        'q2': [('doc2', 0.92), ('doc1', 0.78), ('doc3', 0.68), ('doc4', 0.58), ('doc5', 0.48), ('doc6', 0.38)],
        'q3': [('doc3', 0.90), ('doc1', 0.80), ('doc2', 0.70), ('doc4', 0.60), ('doc5', 0.50), ('doc6', 0.40)]
    }
    
    # Quick evaluation with default settings
    metrics = quick_evaluate(qrels, model_results, k_values=[1, 5, 10])
    
    print("Quick Evaluation Results:")
    for metric, value in metrics.items():
        print(f"  {metric}: {value:.4f}")


def demonstrate_advanced_evaluation():
    """Demonstrate advanced evaluation with custom configuration."""
    print("\n=== Advanced Evaluation Demo ===")
    
    queries, documents, qrels = create_sample_dataset()
    
    # Simulate results from multiple models for comparison
    model_a_results = {
        'q1': [('doc1', 0.95), ('doc4', 0.85), ('doc5', 0.75), ('doc6', 0.65), ('doc2', 0.55), ('doc3', 0.45)],
        'q2': [('doc2', 0.92), ('doc1', 0.82), ('doc4', 0.72), ('doc3', 0.62), ('doc5', 0.52), ('doc6', 0.42)],
        'q3': [('doc3', 0.90), ('doc1', 0.80), ('doc2', 0.70), ('doc4', 0.60), ('doc5', 0.50), ('doc6', 0.40)]
    }
    
    model_b_results = {
        'q1': [('doc4', 0.88), ('doc1', 0.87), ('doc5', 0.78), ('doc6', 0.68), ('doc3', 0.58), ('doc2', 0.48)],
        'q2': [('doc2', 0.89), ('doc4', 0.79), ('doc1', 0.75), ('doc3', 0.65), ('doc6', 0.55), ('doc5', 0.45)],
        'q3': [('doc3', 0.93), ('doc2', 0.83), ('doc1', 0.73), ('doc5', 0.63), ('doc4', 0.53), ('doc6', 0.43)]
    }
    
    # Configure evaluation with detailed settings
    config = EvaluationConfig(
        k_values=[1, 3, 5, 10],
        relevance_threshold=1,
        include_per_query=True,
        statistical_tests=False
    )
    
    # Create evaluator
    evaluator = Evaluator(config)
    
    # Evaluate both models
    print("Model A Results:")
    results_a = evaluator.evaluate(qrels, model_a_results)
    evaluator.print_results(results_a)
    
    print("\nModel B Results:")
    results_b = evaluator.evaluate(qrels, model_b_results)
    evaluator.print_results(results_b)
    
    # Compare models
    print("\nModel Comparison:")
    comparison = evaluator.compare_models({
        'Model A': results_a,
        'Model B': results_b
    })
    
    for metric, values in comparison.items():
        print(f"{metric}:")
        for model, value in values.items():
            print(f"  {model}: {value:.4f}")


def demonstrate_batch_evaluation():
    """Demonstrate batch evaluation with multiple datasets."""
    print("\n=== Batch Evaluation Demo ===")
    
    # Create multiple evaluation scenarios
    scenarios = []
    
    for i in range(3):
        queries, documents, qrels = create_sample_dataset()
        
        # Simulate different model performance per scenario
        noise_factor = 0.1 * i
        model_results = {}
        
        for qid in queries:
            # Add noise to simulate different conditions
            base_scores = [0.9, 0.8, 0.7, 0.6, 0.5, 0.4]
            noisy_scores = [max(0.1, score - noise_factor * np.random.random()) 
                          for score in base_scores]
            
            docs = list(qrels[qid].keys())
            model_results[qid] = [(docs[j], noisy_scores[j]) for j in range(len(docs))]
        
        scenarios.append({
            'name': f'Scenario_{i+1}',
            'qrels': qrels,
            'results': model_results
        })
    
    # Batch evaluation
    config = EvaluationConfig(k_values=[5, 10])
    batch_evaluator = BatchEvaluator(config)
    
    # Run batch evaluation
    batch_results = batch_evaluator.evaluate_batch([
        (scenario['qrels'], scenario['results'], scenario['name']) 
        for scenario in scenarios
    ])
    
    # Print batch results
    print("Batch Evaluation Results:")
    batch_evaluator.print_batch_results(batch_results)
    
    # Statistical summary
    print("\nStatistical Summary:")
    summary = batch_evaluator.statistical_summary(batch_results)
    for metric, stats in summary.items():
        print(f"{metric}: mean={stats['mean']:.4f}, std={stats['std']:.4f}")


def demonstrate_real_world_integration():
    """Demonstrate integration with actual model and dataset components."""
    print("\n=== Real-world Integration Demo ===")
    
    # This would normally load from actual dataset files
    queries, documents, qrels = create_sample_dataset()
    
    # Initialize a sparse retrieval model (BM25)
    try:
        model = SparseRetrievalModel()
        model.index_documents(documents)
        
        # Run retrieval for each query
        model_results = {}
        for qid, query_text in queries.items():
            results = model.search(query_text, top_k=10)
            model_results[qid] = [(hit.doc_id, hit.score) for hit in results]
        
        # Evaluate the model
        config = EvaluationConfig(k_values=[1, 5, 10])
        evaluator = Evaluator(config)
        
        results = evaluator.evaluate(qrels, model_results)
        
        print("Real Model Evaluation Results:")
        evaluator.print_results(results)
        
        # Save results
        output_path = Path("evaluation_results.json")
        evaluator.save_results(results, output_path)
        print(f"Results saved to {output_path}")
        
    except Exception as e:
        print(f"Model integration demo failed: {e}")
        print("This is expected if model dependencies are not installed")


def demonstrate_error_handling():
    """Demonstrate error handling and edge cases."""
    print("\n=== Error Handling Demo ===")
    
    # Test with mismatched query IDs
    qrels = {'q1': {'doc1': 1}}
    results = {'q2': [('doc1', 0.5)]}  # Different query ID
    
    try:
        metrics = quick_evaluate(qrels, results)
        print("No error raised for mismatched query IDs")
    except Exception as e:
        print(f"Handled mismatched query IDs: {type(e).__name__}")
    
    # Test with empty results
    qrels = {'q1': {'doc1': 1}}
    results = {'q1': []}
    
    metrics = quick_evaluate(qrels, results)
    print(f"Empty results handled gracefully: nDCG@10 = {metrics.get('ndcg@10', 0):.4f}")
    
    # Test with invalid relevance scores
    qrels = {'q1': {'doc1': -1}}  # Invalid negative relevance
    results = {'q1': [('doc1', 0.5)]}
    
    try:
        metrics = quick_evaluate(qrels, results)
        print("Negative relevance scores handled")
    except Exception as e:
        print(f"Handled invalid relevance: {type(e).__name__}")


if __name__ == "__main__":
    """Run all demonstration examples."""
    print("NewAIBench Evaluation Module Integration Examples")
    print("=" * 50)
    
    # Run all demonstrations
    demonstrate_basic_evaluation()
    demonstrate_advanced_evaluation() 
    demonstrate_batch_evaluation()
    demonstrate_real_world_integration()
    demonstrate_error_handling()
    
    print("\n" + "=" * 50)
    print("Integration examples completed!")
    print("\nNext steps:")
    print("1. Integrate with your dataset loaders")
    print("2. Connect with your retrieval models")
    print("3. Run cross-validation tests")
    print("4. Optimize for your specific use case")
