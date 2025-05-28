
"""
Reporting and export module for NewAIBench results.

This module provides functionality to generate reports in various formats
(CSV, Markdown, LaTeX) and create visualizations from benchmark results.
"""

import csv
import json
import os
import warnings
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, asdict
import pandas as pd

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    VISUALIZATION_AVAILABLE = True
except ImportError:
    VISUALIZATION_AVAILABLE = False
    warnings.warn(
        "Matplotlib/Seaborn not available. Visualization features will be disabled. "
        "Install with: pip install matplotlib seaborn"
    )

from .aggregator import ResultsAggregator, AggregationConfig
from .storage import ResultsStorage


@dataclass
class ReportConfig:
    """Configuration for report generation."""
    title: str = "NewAIBench Results Report"
    include_metadata: bool = True
    include_statistics: bool = True
    include_comparison: bool = True
    metrics_precision: int = 4
    sort_by: Optional[str] = None
    sort_ascending: bool = False
    top_k: Optional[int] = None


class ReportGenerator:
    """Generates benchmark reports in various formats."""
    
    def __init__(self, storage: ResultsStorage, aggregator: ResultsAggregator):
        """
        Initialize report generator.
        
        Args:
            storage: Storage backend for accessing results
            aggregator: Results aggregator for data processing
        """
        self.storage = storage
        self.aggregator = aggregator
    
    def generate_csv_report(
        self,
        output_path: Union[str, Path],
        config: Optional[ReportConfig] = None,
        aggregation_config: Optional[AggregationConfig] = None
    ) -> str:
        """
        Generate CSV report from benchmark results.
        
        Args:
            output_path: Path to save CSV file
            config: Report configuration
            aggregation_config: Aggregation configuration
            
        Returns:
            Path to generated CSV file
        """
        config = config or ReportConfig()
        aggregation_config = aggregation_config or AggregationConfig()
        
        # Get aggregated results
        results_df = self.aggregator.get_comparison_table(aggregation_config)
        
        if results_df.empty:
            raise ValueError("No results found for given configuration")
        
        # Apply sorting and filtering
        if config.sort_by and config.sort_by in results_df.columns:
            results_df = results_df.sort_values(
                config.sort_by, 
                ascending=config.sort_ascending
            )
        
        if config.top_k:
            results_df = results_df.head(config.top_k)
        
        # Round numeric columns
        numeric_cols = results_df.select_dtypes(include=['float64', 'float32']).columns
        results_df[numeric_cols] = results_df[numeric_cols].round(config.metrics_precision)
        
        # Save to CSV
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        results_df.to_csv(output_path, index=False)
        
        return str(output_path)
    
    def generate_markdown_report(
        self,
        output_path: Union[str, Path],
        config: Optional[ReportConfig] = None,
        aggregation_config: Optional[AggregationConfig] = None
    ) -> str:
        """
        Generate Markdown report from benchmark results.
        
        Args:
            output_path: Path to save Markdown file
            config: Report configuration
            aggregation_config: Aggregation configuration
            
        Returns:
            Path to generated Markdown file
        """
        config = config or ReportConfig()
        aggregation_config = aggregation_config or AggregationConfig()
        
        # Get aggregated results
        results_df = self.aggregator.get_comparison_table(aggregation_config)
        
        if results_df.empty:
            raise ValueError("No results found for given configuration")
        
        # Apply sorting and filtering
        if config.sort_by and config.sort_by in results_df.columns:
            results_df = results_df.sort_values(
                config.sort_by, 
                ascending=config.sort_ascending
            )
        
        if config.top_k:
            results_df = results_df.head(config.top_k)
        
        # Round numeric columns
        numeric_cols = results_df.select_dtypes(include=['float64', 'float32']).columns
        results_df[numeric_cols] = results_df[numeric_cols].round(config.metrics_precision)
        
        # Generate Markdown content
        content = []
        
        # Title and metadata
        content.append(f"# {config.title}")
        content.append("")
        
        if config.include_metadata:
            content.append("## Report Metadata")
            content.append(f"- **Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            content.append(f"- **Total Results**: {len(results_df)}")
            if aggregation_config.datasets:
                content.append(f"- **Datasets**: {', '.join(aggregation_config.datasets)}")
            if aggregation_config.models:
                content.append(f"- **Models**: {', '.join(aggregation_config.models)}")
            content.append("")
        
        # Statistics summary
        if config.include_statistics and not results_df.empty:
            content.append("## Summary Statistics")
            stats_df = results_df[numeric_cols].describe()
            content.append(stats_df.round(config.metrics_precision).to_markdown())
            content.append("")
        
        # Main results table
        content.append("## Results")
        content.append(results_df.to_markdown(index=False))
        content.append("")
        
        # Best models per metric
        if config.include_comparison and not results_df.empty:
            content.append("## Best Performing Models")
            for metric in numeric_cols:
                if metric in results_df.columns:
                    best_idx = results_df[metric].idxmax()
                    best_model = results_df.loc[best_idx, 'model']
                    best_value = results_df.loc[best_idx, metric]
                    content.append(f"- **{metric}**: {best_model} ({best_value:.{config.metrics_precision}f})")
            content.append("")
        
        # Save to file
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(content))
        
        return str(output_path)
    
    def generate_latex_report(
        self,
        output_path: Union[str, Path],
        config: Optional[ReportConfig] = None,
        aggregation_config: Optional[AggregationConfig] = None
    ) -> str:
        """
        Generate LaTeX report from benchmark results.
        
        Args:
            output_path: Path to save LaTeX file
            config: Report configuration
            aggregation_config: Aggregation configuration
            
        Returns:
            Path to generated LaTeX file
        """
        config = config or ReportConfig()
        aggregation_config = aggregation_config or AggregationConfig()
        
        # Get aggregated results
        results_df = self.aggregator.get_comparison_table(aggregation_config)
        
        if results_df.empty:
            raise ValueError("No results found for given configuration")
        
        # Apply sorting and filtering
        if config.sort_by and config.sort_by in results_df.columns:
            results_df = results_df.sort_values(
                config.sort_by, 
                ascending=config.sort_ascending
            )
        
        if config.top_k:
            results_df = results_df.head(config.top_k)
        
        # Round numeric columns
        numeric_cols = results_df.select_dtypes(include=['float64', 'float32']).columns
        results_df[numeric_cols] = results_df[numeric_cols].round(config.metrics_precision)
        
        # Generate LaTeX content
        content = []
        
        # Document structure
        content.append("\\documentclass{article}")
        content.append("\\usepackage{booktabs}")
        content.append("\\usepackage{longtable}")
        content.append("\\usepackage{geometry}")
        content.append("\\geometry{margin=1in}")
        content.append("\\title{" + config.title.replace('_', '\\_') + "}")
        content.append("\\date{" + datetime.now().strftime('%Y-%m-%d') + "}")
        content.append("\\begin{document}")
        content.append("\\maketitle")
        content.append("")
        
        # Metadata section
        if config.include_metadata:
            content.append("\\section{Report Metadata}")
            content.append("\\begin{itemize}")
            content.append(f"\\item Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            content.append(f"\\item Total Results: {len(results_df)}")
            if aggregation_config.datasets:
                datasets_str = ', '.join(aggregation_config.datasets).replace('_', '\\_')
                content.append(f"\\item Datasets: {datasets_str}")
            if aggregation_config.models:
                models_str = ', '.join(aggregation_config.models).replace('_', '\\_')
                content.append(f"\\item Models: {models_str}")
            content.append("\\end{itemize}")
            content.append("")
        
        # Statistics summary
        if config.include_statistics and not results_df.empty:
            content.append("\\section{Summary Statistics}")
            stats_df = results_df[numeric_cols].describe()
            latex_table = stats_df.round(config.metrics_precision).to_latex(
                escape=False,
                column_format='l' + 'c' * len(stats_df.columns)
            )
            content.append(latex_table)
            content.append("")
        
        # Main results table
        content.append("\\section{Results}")
        
        # Handle long tables
        if len(results_df) > 20:
            content.append("\\begin{longtable}{" + "l" + "c" * (len(results_df.columns) - 1) + "}")
            content.append("\\toprule")
        else:
            content.append("\\begin{table}[htbp]")
            content.append("\\centering")
            content.append("\\begin{tabular}{" + "l" + "c" * (len(results_df.columns) - 1) + "}")
            content.append("\\toprule")
        
        # Table headers
        headers = [col.replace('_', '\\_') for col in results_df.columns]
        content.append(" & ".join(headers) + " \\\\")
        content.append("\\midrule")
        
        # Table rows
        for _, row in results_df.iterrows():
            row_values = []
            for val in row:
                if isinstance(val, str):
                    row_values.append(val.replace('_', '\\_'))
                else:
                    row_values.append(str(val))
            content.append(" & ".join(row_values) + " \\\\")
        
        # Close table
        content.append("\\bottomrule")
        if len(results_df) > 20:
            content.append("\\end{longtable}")
        else:
            content.append("\\end{tabular}")
            content.append("\\caption{Benchmark Results}")
            content.append("\\end{table}")
        content.append("")
        
        # Best models section
        if config.include_comparison and not results_df.empty:
            content.append("\\section{Best Performing Models}")
            content.append("\\begin{itemize}")
            for metric in numeric_cols:
                if metric in results_df.columns:
                    best_idx = results_df[metric].idxmax()
                    best_model = results_df.loc[best_idx, 'model'].replace('_', '\\_')
                    best_value = results_df.loc[best_idx, metric]
                    content.append(f"\\item \\textbf{{{metric.replace('_', '\\_')}}}: {best_model} ({best_value:.{config.metrics_precision}f})")
            content.append("\\end{itemize}")
            content.append("")
        
        content.append("\\end{document}")
        
        # Save to file
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(content))
        
        return str(output_path)
    
    def generate_visualization(
        self,
        output_dir: Union[str, Path],
        config: Optional[ReportConfig] = None,
        aggregation_config: Optional[AggregationConfig] = None,
        chart_types: Optional[List[str]] = None
    ) -> List[str]:
        """
        Generate visualization charts from benchmark results.
        
        Args:
            output_dir: Directory to save charts
            config: Report configuration
            aggregation_config: Aggregation configuration
            chart_types: Types of charts to generate ('bar', 'heatmap', 'box', 'scatter')
            
        Returns:
            List of paths to generated chart files
        """
        if not VISUALIZATION_AVAILABLE:
            raise ImportError(
                "Matplotlib/Seaborn not available. Install with: pip install matplotlib seaborn"
            )
        
        config = config or ReportConfig()
        aggregation_config = aggregation_config or AggregationConfig()
        chart_types = chart_types or ['bar', 'heatmap']
        
        # Get aggregated results
        results_df = self.aggregator.get_comparison_table(aggregation_config)
        
        if results_df.empty:
            raise ValueError("No results found for given configuration")
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        numeric_cols = results_df.select_dtypes(include=['float64', 'float32']).columns
        generated_files = []
        
        # Set style
        plt.style.use('seaborn-v0_8' if hasattr(plt.style, 'seaborn-v0_8') else 'default')
        
        # Bar chart for model comparison
        if 'bar' in chart_types and 'model' in results_df.columns:
            for metric in numeric_cols:
                fig, ax = plt.subplots(figsize=(12, 8))
                
                # Sort by metric value
                plot_df = results_df.sort_values(metric, ascending=False)
                if config.top_k:
                    plot_df = plot_df.head(config.top_k)
                
                bars = ax.bar(range(len(plot_df)), plot_df[metric])
                ax.set_xticks(range(len(plot_df)))
                ax.set_xticklabels(plot_df['model'], rotation=45, ha='right')
                ax.set_ylabel(metric)
                ax.set_title(f'{metric} by Model')
                
                # Add value labels on bars
                for i, bar in enumerate(bars):
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height,
                           f'{height:.{config.metrics_precision}f}',
                           ha='center', va='bottom')
                
                plt.tight_layout()
                filename = output_dir / f'{metric}_by_model.png'
                plt.savefig(filename, dpi=300, bbox_inches='tight')
                plt.close()
                generated_files.append(str(filename))
        
        # Heatmap for model-dataset comparison
        if 'heatmap' in chart_types and 'model' in results_df.columns and 'dataset' in results_df.columns:
            for metric in numeric_cols:
                # Create pivot table
                pivot_df = results_df.pivot_table(
                    values=metric,
                    index='model',
                    columns='dataset',
                    aggfunc='mean'
                )
                
                if not pivot_df.empty:
                    fig, ax = plt.subplots(figsize=(10, 8))
                    sns.heatmap(
                        pivot_df,
                        annot=True,
                        fmt=f'.{config.metrics_precision}f',
                        cmap='viridis',
                        ax=ax
                    )
                    ax.set_title(f'{metric} Heatmap: Models vs Datasets')
                    plt.tight_layout()
                    filename = output_dir / f'{metric}_heatmap.png'
                    plt.savefig(filename, dpi=300, bbox_inches='tight')
                    plt.close()
                    generated_files.append(str(filename))
        
        # Box plot for metric distribution
        if 'box' in chart_types:
            if len(numeric_cols) > 1:
                fig, ax = plt.subplots(figsize=(12, 8))
                results_df[numeric_cols].boxplot(ax=ax)
                ax.set_title('Metric Distribution')
                ax.set_ylabel('Values')
                plt.xticks(rotation=45)
                plt.tight_layout()
                filename = output_dir / 'metrics_boxplot.png'
                plt.savefig(filename, dpi=300, bbox_inches='tight')
                plt.close()
                generated_files.append(str(filename))
        
        # Scatter plot for metric correlation
        if 'scatter' in chart_types and len(numeric_cols) >= 2:
            metrics_list = list(numeric_cols)
            for i in range(len(metrics_list)):
                for j in range(i + 1, len(metrics_list)):
                    metric_x, metric_y = metrics_list[i], metrics_list[j]
                    
                    fig, ax = plt.subplots(figsize=(10, 8))
                    scatter = ax.scatter(
                        results_df[metric_x],
                        results_df[metric_y],
                        alpha=0.7,
                        s=60
                    )
                    
                    # Add model labels
                    if 'model' in results_df.columns:
                        for idx, row in results_df.iterrows():
                            ax.annotate(
                                row['model'][:10],  # Truncate long names
                                (row[metric_x], row[metric_y]),
                                xytext=(5, 5),
                                textcoords='offset points',
                                fontsize=8,
                                alpha=0.7
                            )
                    
                    ax.set_xlabel(metric_x)
                    ax.set_ylabel(metric_y)
                    ax.set_title(f'{metric_x} vs {metric_y}')
                    plt.tight_layout()
                    filename = output_dir / f'{metric_x}_vs_{metric_y}_scatter.png'
                    plt.savefig(filename, dpi=300, bbox_inches='tight')
                    plt.close()
                    generated_files.append(str(filename))
        
        return generated_files
    
    def generate_full_report(
        self,
        output_dir: Union[str, Path],
        formats: Optional[List[str]] = None,
        config: Optional[ReportConfig] = None,
        aggregation_config: Optional[AggregationConfig] = None,
        include_visualizations: bool = True
    ) -> Dict[str, List[str]]:
        """
        Generate complete report in multiple formats.
        
        Args:
            output_dir: Directory to save all report files
            formats: List of formats to generate ('csv', 'markdown', 'latex')
            config: Report configuration
            aggregation_config: Aggregation configuration
            include_visualizations: Whether to generate charts
            
        Returns:
            Dictionary mapping format to list of generated files
        """
        formats = formats or ['csv', 'markdown', 'latex']
        config = config or ReportConfig()
        aggregation_config = aggregation_config or AggregationConfig()
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        generated_files = {}
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Generate reports in requested formats
        if 'csv' in formats:
            csv_file = self.generate_csv_report(
                output_dir / f'report_{timestamp}.csv',
                config,
                aggregation_config
            )
            generated_files['csv'] = [csv_file]
        
        if 'markdown' in formats:
            md_file = self.generate_markdown_report(
                output_dir / f'report_{timestamp}.md',
                config,
                aggregation_config
            )
            generated_files['markdown'] = [md_file]
        
        if 'latex' in formats:
            tex_file = self.generate_latex_report(
                output_dir / f'report_{timestamp}.tex',
                config,
                aggregation_config
            )
            generated_files['latex'] = [tex_file]
        
        # Generate visualizations
        if include_visualizations and VISUALIZATION_AVAILABLE:
            try:
                viz_dir = output_dir / f'visualizations_{timestamp}'
                chart_files = self.generate_visualization(
                    viz_dir,
                    config,
                    aggregation_config
                )
                generated_files['visualizations'] = chart_files
            except Exception as e:
                warnings.warn(f"Failed to generate visualizations: {e}")
        
        return generated_files
