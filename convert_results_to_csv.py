#!/usr/bin/env python3
"""
Convert NewAIBench experiment results JSON to CSV format.

This script reads experiment results from JSON files and converts them to CSV format
with models as rows and recall metrics as columns.
"""

import json
import csv
import os
import argparse
from pathlib import Path
from typing import Dict, List, Any


def load_results_json(file_path: str) -> Dict[str, Any]:
    """Load results from JSON file.
    
    Args:
        file_path: Path to the results.json file
        
    Returns:
        Dictionary containing the results data
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def extract_recall_metrics(results_data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Extract recall metrics from results data.
    
    Args:
        results_data: Results data from JSON file
        
    Returns:
        List of dictionaries containing model name and recall metrics
    """
    extracted_data = []
    
    for result in results_data.get('results', []):
        model_name = result.get('model_name', 'Unknown')
        dataset_name = result.get('dataset_name', 'Unknown')
        metrics = result.get('metrics', {})
        
        # Extract recall metrics
        recall_metrics = {}
        for metric_name, metric_value in metrics.items():
            if metric_name.startswith('recall@'):
                recall_metrics[metric_name] = metric_value
        
        # Add additional info
        row_data = {
            'model_name': model_name,
            'dataset_name': dataset_name,
            **recall_metrics
        }
        
        # Add execution times if available
        if 'execution_time' in result:
            row_data['execution_time'] = result['execution_time']
        if 'index_time' in result:
            row_data['index_time'] = result['index_time']
        if 'retrieval_time' in result:
            row_data['retrieval_time'] = result['retrieval_time']
        
        extracted_data.append(row_data)
    
    return extracted_data


def write_csv_report(data: List[Dict[str, Any]], output_path: str):
    """Write extracted data to CSV file.
    
    Args:
        data: List of dictionaries containing model data
        output_path: Path to output CSV file
    """
    if not data:
        print("No data to write to CSV")
        return
    
    # Get all possible columns
    all_columns = set()
    for row in data:
        all_columns.update(row.keys())
    
    # Sort columns with model_name and dataset_name first, then recall metrics, then others
    recall_columns = sorted([col for col in all_columns if col.startswith('recall@')], 
                           key=lambda x: int(x.split('@')[1]))
    
    other_columns = [col for col in all_columns if not col.startswith('recall@') 
                    and col not in ['model_name', 'dataset_name']]
    
    column_order = ['model_name', 'dataset_name'] + recall_columns + sorted(other_columns)
    
    # Write CSV
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=column_order)
        writer.writeheader()
        writer.writerows(data)
    
    print(f"CSV report saved to: {output_path}")
    print(f"Columns: {column_order}")
    print(f"Number of models: {len(data)}")


def process_results_file(input_path: str, output_dir: str = "report_csv"):
    """Process a single results.json file and convert to CSV.
    
    Args:
        input_path: Path to results.json file
        output_dir: Directory to save CSV reports
    """
    print(f"Processing: {input_path}")
    
    try:
        # Load results data
        results_data = load_results_json(input_path)
        
        # Extract recall metrics
        extracted_data = extract_recall_metrics(results_data)
        
        if not extracted_data:
            print(f"No recall metrics found in {input_path}")
            return
        
        # Generate output filename
        input_path_obj = Path(input_path)
        dataset_name = extracted_data[0].get('dataset_name', 'unknown')
        output_filename = f"{dataset_name}_recall_report.csv"
        output_path = os.path.join(output_dir, output_filename)
        
        # Write CSV report
        write_csv_report(extracted_data, output_path)
        
        # Print summary
        print(f"Successfully processed {len(extracted_data)} models")
        for row in extracted_data:
            print(f"  - {row['model_name']}")
        
    except Exception as e:
        print(f"Error processing {input_path}: {e}")


def process_results_directory(results_dir: str, output_dir: str = "report_csv"):
    """Process all results.json files in a directory structure.
    
    Args:
        results_dir: Directory containing results
        output_dir: Directory to save CSV reports
    """
    results_path = Path(results_dir)
    
    # Find all results.json files
    results_files = list(results_path.rglob("results.json"))
    
    if not results_files:
        print(f"No results.json files found in {results_dir}")
        return
    
    print(f"Found {len(results_files)} results.json files")
    
    for results_file in results_files:
        process_results_file(str(results_file), output_dir)
        print("-" * 50)


def main():
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(
        description="Convert NewAIBench experiment results to CSV format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process a single results.json file
  python convert_results_to_csv.py --input results/tydiqa_goldp_vietnamese/tydiqa_goldp_vietnamese/results.json
  
  # Process all results.json files in a directory
  python convert_results_to_csv.py --input results/ --directory
  
  # Specify custom output directory
  python convert_results_to_csv.py --input results.json --output my_reports/
        """
    )
    
    parser.add_argument(
        '--input', '-i',
        type=str,
        required=True,
        help='Path to results.json file or results directory'
    )
    
    parser.add_argument(
        '--output', '-o',
        type=str,
        default='report_csv',
        help='Output directory for CSV reports (default: report_csv)'
    )
    
    parser.add_argument(
        '--directory', '-d',
        action='store_true',
        help='Process all results.json files in input directory recursively'
    )
    
    args = parser.parse_args()
    
    input_path = args.input
    output_dir = args.output
    
    if not os.path.exists(input_path):
        print(f"Error: Input path '{input_path}' does not exist")
        return
    
    if args.directory or os.path.isdir(input_path):
        process_results_directory(input_path, output_dir)
    else:
        process_results_file(input_path, output_dir)


if __name__ == "__main__":
    main()
