#!/usr/bin/env python3
"""
NewAIBench Experiment Runner CLI

Command-line interface for running benchmark experiments with NewAIBench framework.
Supports both configuration files and command-line arguments.
"""

import argparse
import sys
import json
import os
from pathlib import Path
from typing import Dict, Any, List

# Add the src directory to the Python path
src_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "src")
sys.path.insert(0, src_path)

from newaibench.experiment import (
    ExperimentRunner,
    ExperimentConfig,
    ModelConfiguration,
    DatasetConfiguration,
    EvaluationConfiguration,
    OutputConfiguration,
    load_experiment_config,
    save_experiment_config,
    create_experiment_template,
    validate_device_availability,
    setup_experiment_logging
)


def create_parser() -> argparse.ArgumentParser:
    """Create command-line argument parser."""
    parser = argparse.ArgumentParser(
        description="NewAIBench Experiment Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run from configuration file
  python run_experiment.py --config experiments/text_comparison.yaml
  
  # Quick run with CLI arguments
  python run_experiment.py --model bm25 --dataset /data/msmarco --output ./results
  
  # Create configuration template
  python run_experiment.py --create-template basic --output-config template.yaml
  
  # Run multiple models on same dataset
  python run_experiment.py --models bm25,dense --model-paths "",sentence-transformers/all-MiniLM-L6-v2 --dataset /data/test --output ./results
        """
    )
    
    # Configuration file mode
    parser.add_argument(
        '--config', '-c',
        type=str,
        help='Path to experiment configuration file (YAML or JSON)'
    )
    
    # Template creation
    parser.add_argument(
        '--create-template',
        choices=['basic', 'text_comparison', 'image_experiment'],
        help='Create a configuration template'
    )
    
    parser.add_argument(
        '--output-config',
        type=str,
        help='Output path for created configuration template'
    )
    
    # Quick configuration via CLI
    parser.add_argument(
        '--models',
        type=str,
        help='Comma-separated list of model types (bm25, dense, image_retrieval)'
    )
    
    parser.add_argument(
        '--model-paths',
        type=str,
        help='Comma-separated list of model paths/names (empty string for models that don\'t need paths)'
    )
    
    parser.add_argument(
        '--datasets',
        type=str,
        help='Comma-separated list of dataset paths'
    )
    
    parser.add_argument(
        '--dataset-types',
        type=str,
        help='Comma-separated list of dataset types (text, image). If not specified, will be auto-detected.'
    )
    
    parser.add_argument(
        '--output', '-o',
        type=str,
        help='Output directory for experiment results'
    )
    
    parser.add_argument(
        '--experiment-name',
        type=str,
        help='Name for the experiment run'
    )
    
    # Evaluation options
    parser.add_argument(
        '--metrics',
        type=str,
        default='ndcg,map,recall',
        help='Comma-separated list of metrics to compute (default: ndcg,map,recall)'
    )
    
    parser.add_argument(
        '--k-values',
        type=str,
        default='1,5,10,100',
        help='Comma-separated list of k values (default: 1,5,10,100)'
    )
    
    parser.add_argument(
        '--top-k',
        type=int,
        default=1000,
        help='Number of top documents to retrieve (default: 1000)'
    )
    
    # Model parameters
    parser.add_argument(
        '--device',
        choices=['cpu', 'cuda', 'auto'],
        default='auto',
        help='Device to run models on (default: auto)'
    )
    
    parser.add_argument(
        '--batch-size',
        type=int,
        default=32,
        help='Batch size for dense models (default: 32)'
    )
    
    # BM25 specific parameters
    parser.add_argument(
        '--bm25-k1',
        type=float,
        default=1.2,
        help='BM25 k1 parameter (default: 1.2)'
    )
    
    parser.add_argument(
        '--bm25-b',
        type=float,
        default=0.75,
        help='BM25 b parameter (default: 0.75)'
    )
    
    # Other options
    parser.add_argument(
        '--log-level',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        default='INFO',
        help='Logging level (default: INFO)'
    )
    
    parser.add_argument(
        '--save-run-files',
        action='store_true',
        help='Save run files for each experiment'
    )
    
    parser.add_argument(
        '--overwrite',
        action='store_true',
        help='Overwrite existing experiment results'
    )
    
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Show what would be run without actually running experiments'
    )
    
    return parser


def auto_detect_dataset_type(dataset_path: str) -> str:
    """Auto-detect dataset type based on directory contents.
    
    Args:
        dataset_path: Path to dataset directory
        
    Returns:
        'text' or 'image'
    """
    dataset_dir = Path(dataset_path)
    
    if not dataset_dir.exists():
        return 'text'  # Default to text
    
    # Check for image files
    image_extensions = {'.jpg', '.jpeg', '.png', '.tiff', '.tif', '.pdf'}
    for file_path in dataset_dir.rglob('*'):
        if file_path.suffix.lower() in image_extensions:
            return 'image'
    
    # Default to text
    return 'text'


def create_config_from_args(args) -> ExperimentConfig:
    """Create ExperimentConfig from command-line arguments.
    
    Args:
        args: Parsed command-line arguments
        
    Returns:
        ExperimentConfig instance
    """
    # Parse model specifications
    model_types = args.models.split(',')
    model_paths = args.model_paths.split(',') if args.model_paths else [''] * len(model_types)
    
    if len(model_paths) != len(model_types):
        raise ValueError(f"Number of model paths ({len(model_paths)}) must match number of model types ({len(model_types)})")
    
    # Create model configurations
    models = []
    for i, (model_type, model_path) in enumerate(zip(model_types, model_paths)):
        model_type = model_type.strip()
        model_path = model_path.strip()
        
        # Model parameters based on type
        parameters = {}
        if model_type == 'sparse':
            parameters = {
                'k1': args.bm25_k1,
                'b': args.bm25_b
            }
        elif model_type == 'dense':
            parameters = {
                'normalize_embeddings': True
            }
        
        model_config = ModelConfiguration(
            name=f"{model_type}_{i+1}" if len(model_types) > 1 else model_type,
            type=model_type,
            model_name_or_path=model_path,
            parameters=parameters,
            device=validate_device_availability(args.device),
            batch_size=args.batch_size
        )
        models.append(model_config)
    
    # Parse dataset specifications
    dataset_paths = args.datasets.split(',')
    dataset_types = args.dataset_types.split(',') if args.dataset_types else None
    
    # Auto-detect dataset types if not provided
    if dataset_types is None:
        dataset_types = [auto_detect_dataset_type(path.strip()) for path in dataset_paths]
    
    if len(dataset_types) != len(dataset_paths):
        raise ValueError(f"Number of dataset types ({len(dataset_types)}) must match number of dataset paths ({len(dataset_paths)})")
    
    # Create dataset configurations
    datasets = []
    for i, (dataset_path, dataset_type) in enumerate(zip(dataset_paths, dataset_types)):
        dataset_path = dataset_path.strip()
        dataset_type = dataset_type.strip()
        
        # Config overrides based on type
        config_overrides = {}
        if dataset_type == 'image':
            config_overrides = {
                'ocr_enabled': True,
                'cache_enabled': True
            }
        else:
            config_overrides = {
                'cache_enabled': True
            }
        
        dataset_config = DatasetConfiguration(
            name=f"dataset_{i+1}" if len(dataset_paths) > 1 else Path(dataset_path).name,
            type=dataset_type,
            data_dir=dataset_path,
            config_overrides=config_overrides
        )
        datasets.append(dataset_config)
    
    # Create evaluation configuration
    evaluation = EvaluationConfiguration(
        metrics=args.metrics.split(','),
        k_values=[int(k.strip()) for k in args.k_values.split(',')],
        top_k=args.top_k,
        save_run_file=args.save_run_files
    )
    
    # Create output configuration
    experiment_name = args.experiment_name or f"experiment_{len(models)}models_{len(datasets)}datasets"
    
    output = OutputConfiguration(
        output_dir=args.output,
        experiment_name=experiment_name,
        log_level=args.log_level,
        overwrite=args.overwrite
    )
    
    return ExperimentConfig(
        models=models,
        datasets=datasets,
        evaluation=evaluation,
        output=output,
        description=f"Experiment with {len(models)} models and {len(datasets)} datasets"
    )


def print_experiment_plan(config: ExperimentConfig):
    """Print experiment execution plan.
    
    Args:
        config: Experiment configuration
    """
    print("\n" + "="*60)
    print("EXPERIMENT EXECUTION PLAN")
    print("="*60)
    
    print(f"Experiment: {config.output.experiment_name}")
    if config.description:
        print(f"Description: {config.description}")
    print(f"Output directory: {config.output.output_dir}")
    
    print(f"\nModels ({len(config.models)}):")
    for i, model in enumerate(config.models, 1):
        print(f"  {i}. {model.name} ({model.type})")
        if model.model_name_or_path:
            print(f"     Path: {model.model_name_or_path}")
        if model.parameters:
            print(f"     Parameters: {model.parameters}")
    
    print(f"\nDatasets ({len(config.datasets)}):")
    for i, dataset in enumerate(config.datasets, 1):
        print(f"  {i}. {dataset.name} ({dataset.type})")
        print(f"     Path: {dataset.data_dir}")
        if dataset.config_overrides:
            print(f"     Config: {dataset.config_overrides}")
    
    print(f"\nEvaluation:")
    print(f"  Metrics: {', '.join(config.evaluation.metrics)}")
    print(f"  K values: {config.evaluation.k_values}")
    print(f"  Top-k retrieval: {config.evaluation.top_k}")
    
    total_experiments = len(config.models) * len(config.datasets)
    print(f"\nTotal experiments to run: {total_experiments}")
    print("="*60)


def main():
    """Main entry point for the CLI."""
    parser = create_parser()
    args = parser.parse_args()
    
    # Handle template creation
    if args.create_template:
        template = create_experiment_template(args.create_template)
        output_path = args.output_config or f"{args.create_template}_template.yaml"
        
        # Convert to ExperimentConfig and save
        config = ExperimentConfig.from_dict(template)
        save_experiment_config(config, output_path)
        print(f"Created {args.create_template} template: {output_path}")
        return
    
    # Determine configuration source
    if args.config:
        # Load from configuration file
        try:
            config = load_experiment_config(args.config)
            print(f"Loaded configuration from: {args.config}")
        except Exception as e:
            print(f"Error loading configuration: {e}", file=sys.stderr)
            sys.exit(1)
    
    elif args.models and args.datasets and args.output:
        # Create from CLI arguments
        try:
            config = create_config_from_args(args)
            print("Created configuration from command-line arguments")
        except Exception as e:
            print(f"Error creating configuration: {e}", file=sys.stderr)
            sys.exit(1)
    
    else:
        print("Error: Must provide either --config file or --models, --datasets, and --output", file=sys.stderr)
        parser.print_help()
        sys.exit(1)
    
    # Setup logging
    setup_experiment_logging(config.output.log_level)
    
    # Show experiment plan
    print_experiment_plan(config)
    
    # Handle dry run
    if args.dry_run:
        print("\nDry run completed. No experiments were executed.")
        return
    
    # Confirm execution
    try:
        response = input("\nProceed with experiment execution? (y/N): ").strip().lower()
        if response not in ['y', 'yes']:
            print("Experiment cancelled.")
            return
    except KeyboardInterrupt:
        print("\nExperiment cancelled.")
        return
    
    # Run experiments
    try:
        runner = ExperimentRunner(config)
        results = runner.run()
        
        # Print final summary
        successful = sum(1 for r in results if r.success)
        failed = sum(1 for r in results if not r.success)
        
        print(f"\n" + "="*60)
        print("EXPERIMENT COMPLETED")
        print("="*60)
        print(f"Total experiments: {len(results)}")
        print(f"Successful: {successful}")
        print(f"Failed: {failed}")
        print(f"Results saved to: {runner.experiment_dir}")
        
        if failed > 0:
            print(f"\n{failed} experiments failed. Check logs for details.")
            sys.exit(1)
        
    except KeyboardInterrupt:
        print("\nExperiment interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"Experiment failed: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
