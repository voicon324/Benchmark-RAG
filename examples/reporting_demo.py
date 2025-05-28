#!/usr/bin/env python3
"""
Demo script for NewAIBench Reporting System.

This script demonstrates the complete functionality of the reporting system
including storage, aggregation, and report generation.
"""

import json
import os
import sys
import tempfile
from datetime import datetime, timedelta
from pathlib import Path
import random
import pandas as pd

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from newaibench.reporting import (
    ResultsStorage, FileSystemStorage, SQLiteStorage,
    ResultsAggregator, AggregationConfig,
    ReportGenerator, ReportConfig
)
from newaibench.reporting.integration import setup_default_reporting


def create_sample_data(storage: ResultsStorage, num_experiments: int = 3):
    """Create sample benchmark data for demonstration."""
    print("Creating sample benchmark data...")
    
    # Sample models and datasets
    models = ["gpt-4", "claude-3-sonnet", "gemini-pro", "llama-2-70b", "mistral-7b"]
    datasets = ["hellaswag", "arc-challenge", "winogrande", "truthfulqa", "mmlu"]
    experiments = [f"llm_eval_round_{i+1}" for i in range(num_experiments)]
    
    # Sample metrics with realistic variations
    base_metrics = {
        "gpt-4": {"accuracy": 0.85, "f1_score": 0.82, "precision": 0.88, "recall": 0.78},
        "claude-3-sonnet": {"accuracy": 0.83, "f1_score": 0.80, "precision": 0.86, "recall": 0.76},
        "gemini-pro": {"accuracy": 0.81, "f1_score": 0.78, "precision": 0.84, "recall": 0.74},
        "llama-2-70b": {"accuracy": 0.79, "f1_score": 0.76, "precision": 0.82, "recall": 0.72},
        "mistral-7b": {"accuracy": 0.75, "f1_score": 0.72, "precision": 0.78, "recall": 0.68}
    }
    
    dataset_difficulty = {
        "hellaswag": 1.0,
        "arc-challenge": 0.8,
        "winogrande": 0.9,
        "truthfulqa": 0.7,
        "mmlu": 0.85
    }
    
    total_results = 0
    
    for exp_idx, experiment in enumerate(experiments):
        print(f"  Creating data for {experiment}...")
        
        for model in models:
            for dataset in datasets:
                # Add some time variation
                timestamp = datetime.now() - timedelta(
                    days=exp_idx * 7 + random.randint(0, 6),
                    hours=random.randint(0, 23),
                    minutes=random.randint(0, 59)
                )
                
                # Vary metrics based on dataset difficulty and add noise
                difficulty = dataset_difficulty[dataset]
                base = base_metrics[model]
                
                metrics = {}
                for metric, base_value in base.items():
                    # Apply difficulty factor and add random noise
                    adjusted_value = base_value * difficulty * random.uniform(0.95, 1.05)
                    metrics[metric] = round(max(0, min(1, adjusted_value)), 4)
                
                # Add some additional metrics
                metrics.update({
                    "inference_time": round(random.uniform(0.5, 3.0), 3),
                    "memory_usage": round(random.uniform(2.0, 8.0), 2),
                    "tokens_per_second": round(random.uniform(50, 200), 1)
                })
                
                # Store result
                storage.store_result(
                    experiment_name=experiment,
                    model=model,
                    dataset=dataset,
                    metrics=metrics,
                    metadata={
                        "timestamp": timestamp.isoformat(),
                        "config": {
                            "temperature": round(random.uniform(0.1, 1.0), 1),
                            "max_tokens": random.choice([100, 200, 500, 1000]),
                            "batch_size": random.choice([1, 4, 8, 16])
                        },
                        "hardware": {
                            "gpu": random.choice(["A100", "V100", "RTX4090", "H100"]),
                            "memory": f"{random.choice([16, 32, 64, 80])}GB"
                        }
                    }
                )
                total_results += 1
    
    print(f"Created {total_results} sample results across {len(experiments)} experiments")
    return total_results


def demo_storage_operations(storage: ResultsStorage):
    """Demonstrate storage operations."""
    print("\n=== Storage Operations Demo ===")
    
    # List experiments
    experiments = storage.list_experiments()
    print(f"Found {len(experiments)} experiments: {experiments}")
    
    # Get results for specific experiment
    if experiments:
        exp_name = experiments[0]
        results = storage.get_results(experiment_name=exp_name)
        print(f"Experiment '{exp_name}' has {len(results)} results")
        
        if results:
            print(f"Sample result metrics: {results[0].metrics}")


def demo_aggregation(storage: ResultsStorage):
    """Demonstrate aggregation functionality."""
    print("\n=== Aggregation Demo ===")
    
    aggregator = ResultsAggregator(storage)
    
    # Basic aggregation config
    config = AggregationConfig(
        metrics=["accuracy", "f1_score", "inference_time"]
    )
    
    # Get comparison table
    print("1. Model Comparison Table:")
    comparison_df = aggregator.get_comparison_table(config)
    print(comparison_df.head(10).to_string(index=False))
    
    # Summary statistics
    print("\n2. Summary Statistics:")
    stats_df = aggregator.get_summary_statistics(config)
    print(stats_df.round(4).to_string())
    
    # Best models
    print("\n3. Best Models by Accuracy:")
    best_models = aggregator.find_best_models(config, "accuracy")
    for model, score in list(best_models.items())[:5]:
        print(f"  {model}: {score:.4f}")
    
    # Performance trends (if multiple experiments)
    print("\n4. Performance Trends:")
    trends = aggregator.analyze_performance_trends(config)
    for model, trend_data in list(trends.items())[:3]:
        print(f"  {model}:")
        for metric, values in trend_data.items():
            if values and len(values) > 1:
                trend_symbol = "↑" if values[-1] > values[0] else "↓" if values[-1] < values[0] else "→"
                print(f"    {metric}: {trend_symbol} ({values[0]:.4f} → {values[-1]:.4f})")


def demo_reporting(storage: ResultsStorage, output_dir: str):
    """Demonstrate report generation."""
    print(f"\n=== Report Generation Demo ===")
    
    aggregator = ResultsAggregator(storage)
    reporter = ReportGenerator(storage, aggregator)
    
    # Setup configurations
    agg_config = AggregationConfig(
        metrics=["accuracy", "f1_score", "precision", "recall"]
    )
    
    report_config = ReportConfig(
        title="LLM Benchmark Comparison Report",
        sort_by="accuracy",
        sort_ascending=False,
        top_k=15,
        metrics_precision=4
    )
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print("Generating reports...")
    
    # Generate all formats
    generated_files = reporter.generate_full_report(
        output_dir=output_path,
        formats=["csv", "markdown", "latex"],
        config=report_config,
        aggregation_config=agg_config,
        include_visualizations=True
    )
    
    print("Generated files:")
    for format_type, files in generated_files.items():
        print(f"  {format_type}:")
        for file_path in files:
            print(f"    - {file_path}")
    
    return generated_files


def demo_cli_simulation(storage_path: str):
    """Demonstrate CLI functionality through Python."""
    print(f"\n=== CLI Simulation Demo ===")
    
    # Import CLI functions
    from newaibench.reporting.cli import (
        setup_storage, cmd_list_experiments, cmd_list_results, 
        cmd_aggregate, cmd_generate_report
    )
    
    # Create mock args for CLI functions
    class MockArgs:
        def __init__(self, **kwargs):
            for key, value in kwargs.items():
                setattr(self, key, value)
    
    print("1. Listing experiments via CLI...")
    args = MockArgs(storage_type="dual", storage_path=storage_path)
    try:
        cmd_list_experiments(args)
    except Exception as e:
        print(f"CLI demo error: {e}")
    
    print("\n2. Aggregation via CLI...")
    args = MockArgs(
        storage_type="dual", 
        storage_path=storage_path,
        operation="statistics",
        experiments=None,
        models=["gpt-4", "claude-3-sonnet"],
        datasets=None,
        metrics=["accuracy", "f1_score"],
        metric=None
    )
    try:
        cmd_aggregate(args)
    except Exception as e:
        print(f"CLI demo error: {e}")


def demo_integration():
    """Demonstrate integration with experiment runner."""
    print(f"\n=== Integration Demo ===")
    
    # Create a mock experiment runner
    class MockExperimentRunner:
        def __init__(self):
            self.name = "MockRunner"
        
        def evaluate(self, model, dataset, output_dir=None):
            """Mock evaluation method."""
            # Simulate evaluation result
            class MockResult:
                def __init__(self):
                    self.metrics = {
                        "accuracy": random.uniform(0.7, 0.9),
                        "f1_score": random.uniform(0.65, 0.85)
                    }
                    self.config = {"temperature": 0.7}
            
            return MockResult()
    
    # Setup integration
    from newaibench.reporting.integration import ReportingIntegration
    
    temp_dir = tempfile.mkdtemp()
    integration = setup_default_reporting(temp_dir)
    
    # Mock runner
    runner = MockExperimentRunner()
    
    # Setup auto-capture
    integration.setup_auto_capture(runner, "integration_test")
    
    # Mock models and datasets
    class MockModel:
        def __init__(self, name):
            self.name = name
    
    class MockDataset:
        def __init__(self, name):
            self.name = name
    
    # Run some mock evaluations
    print("Running mock evaluations with auto-capture...")
    for i in range(3):
        model = MockModel(f"test_model_{i+1}")
        dataset = MockDataset(f"test_dataset_{i+1}")
        result = runner.evaluate(model, dataset)
        print(f"  Evaluated {model.name} on {dataset.name}: accuracy={result.metrics['accuracy']:.4f}")
    
    # Check captured results
    experiments = integration.storage.list_experiments()
    print(f"Auto-captured experiments: {experiments}")
    
    # Cleanup
    import shutil
    shutil.rmtree(temp_dir)


def main():
    """Main demo function."""
    print("NewAIBench Reporting System Demo")
    print("=" * 50)
    
    # Create temporary directory for demo
    with tempfile.TemporaryDirectory() as temp_dir:
        print(f"Using temporary directory: {temp_dir}")
        
        # Setup reporting system
        print("\nSetting up reporting system...")
        integration = setup_default_reporting(temp_dir)
        storage = integration.storage
        
        # Create sample data
        num_results = create_sample_data(storage, num_experiments=3)
        
        # Demo storage operations
        demo_storage_operations(storage)
        
        # Demo aggregation
        demo_aggregation(storage)
        
        # Demo reporting
        reports_dir = Path(temp_dir) / "reports"
        generated_files = demo_reporting(storage, str(reports_dir))
        
        # Demo CLI simulation
        demo_cli_simulation(temp_dir)
        
        # Show some generated report content
        print(f"\n=== Sample Report Content ===")
        if "markdown" in generated_files and generated_files["markdown"]:
            md_file = generated_files["markdown"][0]
            if Path(md_file).exists():
                with open(md_file, 'r') as f:
                    content = f.read()
                    # Show first 50 lines
                    lines = content.split('\n')
                    print('\n'.join(lines[:50]))
                    if len(lines) > 50:
                        print(f"\n... (showing first 50 lines of {len(lines)} total)")
        
        print(f"\n=== Demo Summary ===")
        print(f"- Created {num_results} sample benchmark results")
        print(f"- Demonstrated storage, aggregation, and reporting")
        print(f"- Generated {sum(len(files) for files in generated_files.values())} output files")
        print(f"- All files saved to: {reports_dir}")
        
        # Keep reports accessible (copy to permanent location)
        permanent_dir = Path("./demo_reports")
        if reports_dir.exists():
            import shutil
            if permanent_dir.exists():
                shutil.rmtree(permanent_dir)
            shutil.copytree(reports_dir, permanent_dir)
            print(f"- Reports copied to permanent location: {permanent_dir}")
    
    # Demo integration separately (needs clean environment)
    demo_integration()
    
    print("\nDemo completed successfully!")
    print("Check the generated reports in ./demo_reports/")


if __name__ == "__main__":
    main()
