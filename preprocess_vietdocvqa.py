#!/usr/bin/env python3
"""
Preprocessing script for VietDocVQA dataset to convert ocr_text field to text field.
This script ensures compatibility with the TextDatasetLoader which expects 'text', 'contents', or 'content' fields.
"""

import json
import shutil
from pathlib import Path


def preprocess_corpus(input_file: Path, output_file: Path) -> None:
    """Convert ocr_text field to text field in JSONL corpus file."""
    print(f"Processing corpus file: {input_file}")
    
    processed_count = 0
    with open(input_file, 'r', encoding='utf-8') as infile, \
         open(output_file, 'w', encoding='utf-8') as outfile:
        
        for line_num, line in enumerate(infile, 1):
            try:
                line = line.strip()
                if not line:
                    continue
                
                doc = json.loads(line)
                
                # Convert ocr_text to text field
                if "ocr_text" in doc:
                    doc["text"] = doc["ocr_text"]
                    # Keep the original ocr_text field for reference
                    # del doc["ocr_text"]  # Uncomment if you want to remove the original field
                
                # Write the processed document
                json.dump(doc, outfile, ensure_ascii=False)
                outfile.write('\n')
                processed_count += 1
                
            except json.JSONDecodeError as e:
                print(f"Warning: Invalid JSON on line {line_num}: {e}")
                continue
            except Exception as e:
                print(f"Warning: Error processing line {line_num}: {e}")
                continue
    
    print(f"Successfully processed {processed_count} documents")


def main():
    """Main preprocessing function."""
    # Define paths
    data_dir = Path("/home/hkduy/NewAI/new_bench/NewAIBench_VietDocVQAII_with_OCR")
    original_corpus = data_dir / "corpus.jsonl"
    processed_corpus = data_dir / "corpus_processed.jsonl"
    
    # Check if input file exists
    if not original_corpus.exists():
        print(f"Error: Input corpus file not found: {original_corpus}")
        return
    
    # Create backup if processed file already exists
    if processed_corpus.exists():
        backup_file = data_dir / "corpus_processed.jsonl.backup"
        print(f"Backing up existing processed file to: {backup_file}")
        shutil.copy2(processed_corpus, backup_file)
    
    # Process the corpus
    preprocess_corpus(original_corpus, processed_corpus)
    
    # Check if we need to copy other files or create symbolic links
    for file_name in ["queries.jsonl", "qrels.jsonl"]:
        original_file = data_dir / file_name
        if original_file.exists():
            print(f"Found {file_name} - this file should work as-is")
        else:
            print(f"Warning: {file_name} not found in {data_dir}")
    
    print("\nPreprocessing complete!")
    print(f"Processed corpus saved to: {processed_corpus}")
    print("\nTo use the processed corpus, update your experiment configuration to use 'corpus_processed.jsonl' instead of 'corpus.jsonl'")


if __name__ == "__main__":
    main()
