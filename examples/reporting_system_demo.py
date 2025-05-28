"""
Demo script showing how to use the NewAIBench reporting system.

This script demonstrates:
1. Storing experiment results
2. Aggregating and analyzing results
3. Generating reports in multiple formats
4. Using the CLI interface
"""

import sys
import json
import tempfile
from datetime import datetime, timedelta
from pathlib import Path

# Add source to path
sys.path.append('/home/hkduy/NewAI/new_bench/src')

from newaibench.reporting.storage import ResultsStorage, ExperimentMetadata, RunResult, EvaluationResults
from newaibench.reporting.aggregator import ResultsAggregator, AggregationConfig, create_aggregation_config
from newaibench.reporting.reporter import ReportGenerator, ReportConfig
from newaibench.reporting.integration import ReportingIntegration


def create_sample_experiment_data(storage: ResultsStorage, num_experiments: int = 3):
    """Create sample experiment data for demonstration."""
    
    print(f"🔄 Creating {num_experiments} sample experiments...")
    
    models = ["BERT-base", "RoBERTa-large", "GPT-3.5", "T5-base", "DistilBERT"]
    datasets = ["MS-MARCO", "Natural-Questions", "SQuAD-2.0", "TREC-DL"]
    
    for exp_id in range(num_experiments):
        experiment_name = f"experiment_{exp_id + 1}"
        print(f"  Creating {experiment_name}...")
        
        # Create experiment metadata
        experiment_metadata = ExperimentMetadata(
            experiment_id=experiment_name,
            experiment_name=experiment_name,
            description=f"Sample experiment {exp_id + 1} for benchmarking various models",
            created_at=datetime.now() - timedelta(days=exp_id * 7),
            author="demo_user",
            tags=["benchmark", "demo", f"experiment_{exp_id + 1}"]
        )
        storage.store_experiment_metadata(experiment_metadata)
        
        for model in models[:3]:  # Use first 3 models
            for dataset in datasets[:2]:  # Use first 2 datasets
                # Create multiple runs per model-dataset combination
                for run_id in range(2):
                    
                    # Simulate realistic timestamps
                    base_time = datetime.now() - timedelta(days=exp_id * 7 + run_id)
                    
                    # Create unique run ID
                    import uuid
                    run_id_str = str(uuid.uuid4())
                    
                    # Simulate realistic metrics based on model and dataset
                    base_ndcg = 0.65
                    if "BERT" in model:
                        base_ndcg = 0.72
                    elif "RoBERTa" in model:
                        base_ndcg = 0.75
                    elif "GPT" in model:
                        base_ndcg = 0.78
                    elif "T5" in model:
                        base_ndcg = 0.70
                    
                    # Dataset difficulty factor
                    if dataset == "MS-MARCO":
                        base_ndcg *= 0.95
                    elif dataset == "Natural-Questions":
                        base_ndcg *= 0.90
                    elif dataset == "SQuAD-2.0":
                        base_ndcg *= 1.05
                    
                    # Add some random variation
                    import random
                    random.seed(42 + hash(model + dataset + str(run_id) + str(exp_id)))
                    variation = random.uniform(-0.08, 0.08)
                    
                    ndcg_10 = max(0.1, min(1.0, base_ndcg + variation))
                    map_10 = ndcg_10 * random.uniform(0.85, 0.95)
                    recall_10 = ndcg_10 * random.uniform(1.05, 1.15)
                    precision_10 = ndcg_10 * random.uniform(0.80, 0.90)
                    
                    metrics = {
                        "ndcg@10": round(ndcg_10, 4),
                        "map@10": round(map_10, 4),
                        "recall@10": round(min(1.0, recall_10), 4),
                        "precision@10": round(precision_10, 4),
                        "mrr": round(ndcg_10 * 0.9, 4)
                    }
                    
                    # Simulate execution time
                    execution_time = random.uniform(80, 300)
                    
                    # Randomly simulate some failures
                    success = True
                    if random.random() < 0.05:  # 5% failure rate
                        success = False
                        metrics = {}
                    
                    run_result = RunResult(
                        run_id=run_id_str,
                        experiment_id=experiment_name,
                        model_name=model,
                        dataset_name=dataset,
                        metrics=metrics,
                        execution_time=execution_time,
                        created_at=base_time,
                        config={
                            "learning_rate": 0.001 if run_id == 0 else 0.0005,
                            "batch_size": 32,
                            "epochs": 5,
                            "run_id": run_id
                        },
                        metadata={
                            "model_size": "110M" if "base" in model else "340M",
                            "dataset_size": "8.8M" if dataset == "MS-MARCO" else "307K",
                            "notes": f"Run {run_id + 1} for {model} on {dataset}"
                        },
                        success=success
                    )
                    
                    storage.store_run_result(run_result)
    
    print(f"✅ Created sample data for {num_experiments} experiments")


def demonstrate_storage_operations(storage: ResultsStorage):
    """Demonstrate basic storage operations."""
    
    print("\n📁 STORAGE OPERATIONS DEMO")
    print("=" * 50)
    
    # List experiments
    experiments = storage.list_experiments()
    print(f"📊 Found {len(experiments)} experiments:")
    for exp in experiments:
        print(f"  - {exp}")
    
    # Get details for first experiment
    if experiments:
        first_exp = experiments[0]
        exp_results = storage.get_experiment_results(first_exp)
        exp_metadata = storage.get_experiment_metadata(first_exp)
        
        print(f"\n📈 Details for '{first_exp}':")
        print(f"  - Total runs: {len(exp_results)}")
        if exp_metadata:
            print(f"  - Created: {exp_metadata.created_at}")
            print(f"  - Description: {exp_metadata.description}")
        
        # Show some run details
        completed_runs = [r for r in exp_results if r.success]
        if completed_runs:
            sample_run = completed_runs[0]
            print(f"  - Sample run: {sample_run.model_name} on {sample_run.dataset_name}")
            print(f"    Metrics: {sample_run.metrics}")


def demonstrate_aggregation_analysis(aggregator: ResultsAggregator):
    """Demonstrate aggregation and analysis features."""
    
    print("\n📊 AGGREGATION & ANALYSIS DEMO")
    print("=" * 50)
    
    # Basic aggregation config
    config = create_aggregation_config(
        metrics=["ndcg@10", "map@10", "recall@10"],
        include_failed=False
    )
    
    # Time series analysis
    print("⏰ Time Series Analysis:")
    time_analysis = aggregator.get_time_series_analysis(config)
    print(f"  - Total runs analyzed: {time_analysis.get('total_runs', 0)}")
    if 'date_range' in time_analysis:
        print(f"  - Date range: {time_analysis['date_range']['start']} to {time_analysis['date_range']['end']}")
    
    # Try pandas-based operations (if available)
    try:
        print("\n📈 Comparison Table (top 5 model-dataset combinations):")
        comparison_df = aggregator.get_comparison_table(config)
        if comparison_df is not None and not comparison_df.empty:
            top_performers = comparison_df.nlargest(5, 'ndcg@10')
            for idx, row in top_performers.iterrows():
                print(f"  {idx+1}. {row['model']} on {row['dataset']}: NDCG@10 = {row['ndcg@10']:.4f}")
        else:
            print("  (No pandas data available)")
    except ImportError:
        print("  (Pandas not available - skipping DataFrame operations)")
    
    # Export aggregated results
    print("\n💾 Exporting aggregated results...")
    output_dir = Path(tempfile.gettempdir()) / "newaibenh_demo"
    output_dir.mkdir(exist_ok=True)
    
    json_file = output_dir / "aggregated_results.json"
    aggregator.export_aggregated_results(config, json_file, format='json')
    print(f"  ✅ Exported to: {json_file}")


def demonstrate_report_generation(integration: ReportingIntegration):
    """Demonstrate report generation."""
    
    print("\n📄 REPORT GENERATION DEMO")
    print("=" * 50)
    
    output_dir = Path(tempfile.gettempdir()) / "newaibenh_reports"
    output_dir.mkdir(exist_ok=True)
    
    # Get list of experiments
    experiments = integration.storage.list_experiments()
    if not experiments:
        print("❌ No experiments found for reporting")
        return
    
    # Generate report for first experiment
    first_exp = experiments[0]
    print(f"📊 Generating report for experiment: {first_exp}")
    
    try:
        output_files = integration.generate_experiment_report(
            experiment_name=first_exp,
            output_dir=str(output_dir),
            formats=['markdown', 'csv']
        )
        
        print("✅ Generated report files:")
        for file_path in output_files:
            file_size = Path(file_path).stat().st_size
            print(f"  - {file_path} ({file_size} bytes)")
            
    except Exception as e:
        print(f"⚠️  Report generation error: {e}")
    
    # Generate comparative report across all experiments
    print(f"\n📈 Generating comparative analysis across all experiments...")
    
    try:
        # Create aggregation config for all experiments
        config = AggregationConfig(
            experiments=experiments,
            metrics=["ndcg@10", "map@10", "recall@10"],
            include_failed=False
        )
        
        # Generate comprehensive report
        report_config = ReportConfig(
            title="NewAIBench Comprehensive Analysis",
            include_metadata=True,
            include_statistics=True,
            include_comparison=True
        )
        
        # Generate comprehensive report using the integration's report generator
        try:
            csv_path = output_dir / "comprehensive_report.csv"
            integration.report_generator.generate_csv_report(
                output_path=str(csv_path),
                config=report_config,
                aggregation_config=config
            )
            print(f"✅ Generated comprehensive CSV report: {csv_path}")
        except Exception as csv_error:
            print(f"⚠️  CSV report error: {csv_error}")
        
        try:
            md_path = output_dir / "comprehensive_report.md"
            integration.report_generator.generate_markdown_report(
                output_path=str(md_path),
                config=report_config,
                aggregation_config=config
            )
            print(f"✅ Generated comprehensive Markdown report: {md_path}")
        except Exception as md_error:
            print(f"⚠️  Markdown report error: {md_error}")
            
    except Exception as e:
        print(f"⚠️  Comprehensive report error: {e}")


def demonstrate_cli_usage():
    """Demonstrate CLI usage examples."""
    
    print("\n💻 CLI USAGE EXAMPLES")
    print("=" * 50)
    
    print("Here are some example CLI commands you can run:")
    print("")
    
    # Storage path for examples
    storage_path = "/tmp/newaibenh_demo"
    
    examples = [
        {
            "description": "List all experiments",
            "command": f"python -m newaibench.reporting.cli list --storage-path {storage_path}"
        },
        {
            "description": "Generate report for specific experiment",
            "command": f"python -m newaibench.reporting.cli report experiment_1 --output-dir ./reports --format markdown csv"
        },
        {
            "description": "Aggregate results with filters",
            "command": f"python -m newaibench.reporting.cli aggregate --models BERT-base RoBERTa-large --metrics ndcg@10 map@10 --output ./aggregated.json"
        },
        {
            "description": "Import results from JSON file",
            "command": f"python -m newaibench.reporting.cli import results.json --experiment-name imported_experiment"
        }
    ]
    
    for i, example in enumerate(examples, 1):
        print(f"{i}. {example['description']}:")
        print(f"   {example['command']}")
        print()


def main():
    """Run the complete demonstration."""
    
    print("🚀 NewAIBench Reporting System Demo")
    print("=" * 60)
    print()
    
    # Create temporary storage
    temp_dir = tempfile.mkdtemp(prefix="newaibenh_demo_")
    print(f"📁 Using temporary storage: {temp_dir}")
    
    try:
        # Initialize components
        storage = ResultsStorage(temp_dir)
        aggregator = ResultsAggregator(storage)
        integration = ReportingIntegration(storage)
        
        # Create sample data
        create_sample_experiment_data(storage, num_experiments=3)
        
        # Demonstrate different features
        demonstrate_storage_operations(storage)
        demonstrate_aggregation_analysis(aggregator)
        demonstrate_report_generation(integration)
        demonstrate_cli_usage()
        
        print("\n" + "=" * 60)
        print("✅ Demo completed successfully!")
        print(f"📁 Demo files available in: {temp_dir}")
        print("🗑️  Temporary files will be cleaned up on exit")
        
    except Exception as e:
        print(f"\n❌ Demo failed with error: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Clean up (optional - comment out to inspect files)
        import shutil
        try:
            shutil.rmtree(temp_dir, ignore_errors=True)
            print("🧹 Cleaned up temporary files")
        except:
            pass


if __name__ == "__main__":
    main()
