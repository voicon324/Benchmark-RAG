
"""
Command-line interface for NewAIBench reporting system.

This module provides CLI commands for generating reports, aggregating results,
and managing benchmark data.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Optional, List, Dict, Any

from .storage import ResultsStorage
from .aggregator import ResultsAggregator, AggregationConfig
from .reporter import ReportGenerator, ReportConfig


def setup_storage(storage_type: str, storage_path: str) -> ResultsStorage:
    """Setup storage backend based on type."""
    # Since we only have filesystem storage now, just use that
    return ResultsStorage(storage_path)


def cmd_import_results(args):
    """Import results from existing NewAIBench output directories."""
    storage = setup_storage(args.storage_type, args.storage_path)
    
    imported_count = 0
    for results_dir in args.results_dirs:
        results_path = Path(results_dir)
        if not results_path.exists():
            print(f"Warning: {results_dir} does not exist, skipping...")
            continue
        
        # Look for experiment directories
        for exp_dir in results_path.iterdir():
            if exp_dir.is_dir():
                # Look for evaluation.json files in subdirectories
                for run_dir in exp_dir.iterdir():
                    if run_dir.is_dir():
                        eval_file = run_dir / 'evaluation.json'
                        if eval_file.exists():
                            try:
                                with open(eval_file, 'r') as f:
                                    eval_data = json.load(f)
                                
                                # Extract experiment info from path
                                experiment_name = exp_dir.name
                                run_parts = run_dir.name.split('_')
                                if len(run_parts) >= 3:
                                    model = run_parts[0]
                                    dataset = run_parts[1]
                                    timestamp = '_'.join(run_parts[2:])
                                else:
                                    model = run_dir.name
                                    dataset = "unknown"
                                    timestamp = "unknown"
                                
                                # Store result
                                storage.store_result(
                                    experiment_name=experiment_name,
                                    model=model,
                                    dataset=dataset,
                                    metrics=eval_data.get('metrics', {}),
                                    metadata={
                                        'timestamp': timestamp,
                                        'config': eval_data.get('config', {}),
                                        'source_path': str(eval_file)
                                    }
                                )
                                imported_count += 1
                                
                            except Exception as e:
                                print(f"Error importing {eval_file}: {e}")
    
    print(f"Successfully imported {imported_count} results")


def cmd_list_experiments(args):
    """List all experiments in storage."""
    storage = setup_storage(args.storage_type, args.storage_path)
    experiments = storage.list_experiments()
    
    if not experiments:
        print("No experiments found")
        return
    
    print(f"Found {len(experiments)} experiments:")
    for exp in experiments:
        print(f"  - {exp}")


def cmd_list_results(args):
    """List results with optional filtering."""
    storage = setup_storage(args.storage_type, args.storage_path)
    
    config = AggregationConfig(
        experiments=args.experiments,
        models=args.models,
        datasets=args.datasets
    )
    
    aggregator = ResultsAggregator(storage)
    results_df = aggregator.get_comparison_table(config)
    
    if results_df.empty:
        print("No results found matching criteria")
        return
    
    print(f"Found {len(results_df)} results:")
    print(results_df.to_string(index=False))


def cmd_generate_report(args):
    """Generate benchmark report."""
    storage = setup_storage(args.storage_type, args.storage_path)
    aggregator = ResultsAggregator(storage)
    reporter = ReportGenerator(storage, aggregator)
    
    # Setup configurations
    aggregation_config = AggregationConfig(
        experiments=args.experiments,
        models=args.models,
        datasets=args.datasets,
        metrics=args.metrics
    )
    
    report_config = ReportConfig(
        title=args.title,
        include_metadata=not args.no_metadata,
        include_statistics=not args.no_statistics,
        include_comparison=not args.no_comparison,
        metrics_precision=args.precision,
        sort_by=args.sort_by,
        sort_ascending=args.sort_ascending,
        top_k=args.top_k
    )
    
    # Generate reports
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if args.format == 'all':
        generated_files = reporter.generate_full_report(
            output_dir,
            formats=['csv', 'markdown', 'latex'],
            config=report_config,
            aggregation_config=aggregation_config,
            include_visualizations=args.include_visualizations
        )
        
        print("Generated files:")
        for format_type, files in generated_files.items():
            print(f"  {format_type}:")
            for file_path in files:
                print(f"    - {file_path}")
    
    else:
        if args.format == 'csv':
            output_file = reporter.generate_csv_report(
                output_dir / 'report.csv',
                report_config,
                aggregation_config
            )
        elif args.format == 'markdown':
            output_file = reporter.generate_markdown_report(
                output_dir / 'report.md',
                report_config,
                aggregation_config
            )
        elif args.format == 'latex':
            output_file = reporter.generate_latex_report(
                output_dir / 'report.tex',
                report_config,
                aggregation_config
            )
        else:
            raise ValueError(f"Unknown format: {args.format}")
        
        print(f"Generated report: {output_file}")
        
        # Generate visualizations if requested
        if args.include_visualizations:
            try:
                viz_files = reporter.generate_visualization(
                    output_dir / 'visualizations',
                    report_config,
                    aggregation_config
                )
                print("Generated visualizations:")
                for file_path in viz_files:
                    print(f"  - {file_path}")
            except ImportError as e:
                print(f"Warning: {e}")


def cmd_aggregate(args):
    """Aggregate and analyze results."""
    storage = setup_storage(args.storage_type, args.storage_path)
    aggregator = ResultsAggregator(storage)
    
    config = AggregationConfig(
        experiments=args.experiments,
        models=args.models,
        datasets=args.datasets,
        metrics=args.metrics
    )
    
    if args.operation == 'statistics':
        stats = aggregator.get_summary_statistics(config)
        print("Summary Statistics:")
        print(stats.to_string())
    
    elif args.operation == 'comparison':
        comparison = aggregator.get_comparison_table(config)
        print("Model Comparison:")
        print(comparison.to_string(index=False))
    
    elif args.operation == 'best':
        if not args.metric:
            print("Error: --metric required for 'best' operation")
            return
        
        best_models = aggregator.find_best_models(config, args.metric)
        print(f"Best models for {args.metric}:")
        for model, score in best_models.items():
            print(f"  {model}: {score:.4f}")
    
    elif args.operation == 'trends':
        trends = aggregator.analyze_performance_trends(config)
        print("Performance Trends:")
        for model, trend_data in trends.items():
            print(f"  {model}:")
            for metric, values in trend_data.items():
                if values:
                    trend = "↑" if values[-1] > values[0] else "↓" if values[-1] < values[0] else "→"
                    print(f"    {metric}: {trend} ({values[0]:.4f} → {values[-1]:.4f})")


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="NewAIBench Reporting CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    # Global arguments
    parser.add_argument(
        '--storage-type',
        choices=['filesystem', 'sqlite', 'dual'],
        default='dual',
        help='Storage backend type'
    )
    parser.add_argument(
        '--storage-path',
        default='./results',
        help='Path to storage directory or database'
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Import command
    import_parser = subparsers.add_parser('import', help='Import existing results')
    import_parser.add_argument(
        'results_dirs',
        nargs='+',
        help='Directories containing NewAIBench results'
    )
    import_parser.set_defaults(func=cmd_import_results)
    
    # List experiments command
    list_exp_parser = subparsers.add_parser('list-experiments', help='List all experiments')
    list_exp_parser.set_defaults(func=cmd_list_experiments)
    
    # List results command
    list_parser = subparsers.add_parser('list', help='List results with filtering')
    list_parser.add_argument('--experiments', nargs='+', help='Filter by experiments')
    list_parser.add_argument('--models', nargs='+', help='Filter by models')
    list_parser.add_argument('--datasets', nargs='+', help='Filter by datasets')
    list_parser.set_defaults(func=cmd_list_results)
    
    # Generate report command
    report_parser = subparsers.add_parser('report', help='Generate benchmark report')
    report_parser.add_argument(
        '--format',
        choices=['csv', 'markdown', 'latex', 'all'],
        default='markdown',
        help='Output format'
    )
    report_parser.add_argument('--output-dir', default='./reports', help='Output directory')
    report_parser.add_argument('--title', default='NewAIBench Results Report', help='Report title')
    report_parser.add_argument('--experiments', nargs='+', help='Filter by experiments')
    report_parser.add_argument('--models', nargs='+', help='Filter by models')
    report_parser.add_argument('--datasets', nargs='+', help='Filter by datasets')
    report_parser.add_argument('--metrics', nargs='+', help='Filter by metrics')
    report_parser.add_argument('--sort-by', help='Sort results by metric')
    report_parser.add_argument('--sort-ascending', action='store_true', help='Sort in ascending order')
    report_parser.add_argument('--top-k', type=int, help='Show only top K results')
    report_parser.add_argument('--precision', type=int, default=4, help='Decimal precision for metrics')
    report_parser.add_argument('--no-metadata', action='store_true', help='Exclude metadata section')
    report_parser.add_argument('--no-statistics', action='store_true', help='Exclude statistics section')
    report_parser.add_argument('--no-comparison', action='store_true', help='Exclude comparison section')
    report_parser.add_argument('--include-visualizations', action='store_true', help='Generate charts')
    report_parser.set_defaults(func=cmd_generate_report)
    
    # Aggregate command
    agg_parser = subparsers.add_parser('aggregate', help='Aggregate and analyze results')
    agg_parser.add_argument(
        'operation',
        choices=['statistics', 'comparison', 'best', 'trends'],
        help='Aggregation operation'
    )
    agg_parser.add_argument('--experiments', nargs='+', help='Filter by experiments')
    agg_parser.add_argument('--models', nargs='+', help='Filter by models')
    agg_parser.add_argument('--datasets', nargs='+', help='Filter by datasets')
    agg_parser.add_argument('--metrics', nargs='+', help='Filter by metrics')
    agg_parser.add_argument('--metric', help='Specific metric for best/trends analysis')
    agg_parser.set_defaults(func=cmd_aggregate)
    
    # Parse and execute
    args = parser.parse_args()
    
    if not hasattr(args, 'func'):
        parser.print_help()
        return
    
    try:
        args.func(args)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
