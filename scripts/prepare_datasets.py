#!/usr/bin/env python3
"""
Document Image Dataset Preparation Scripts

Main entry point for preparing various document image datasets for NewAIBench.
Supports DocVQA, Vietnamese Admin Docs, and other image document datasets.
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, Any, Optional

# Add the src directory to the path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from newaibench.datasets.converters.docvqa_converter import DocVQAConverter
from newaibench.datasets.converters.vietnamese_admin_docs_converter import VietnameseAdminDocsConverter
from newaibench.datasets.converters.base_converter import ConversionConfig
from newaibench.utils.ocr_processor import create_ocr_processor

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DatasetPreparer:
    """Main class for dataset preparation workflows."""
    
    SUPPORTED_DATASETS = {
        "docvqa": {
            "name": "DocVQA",
            "converter": DocVQAConverter,
            "description": "Document Visual Question Answering dataset",
            "ocr_required": True
        },
        "vietnamese_admin": {
            "name": "Vietnamese Administrative Documents",
            "converter": VietnameseAdminDocsConverter,
            "description": "Vietnamese administrative documents sample dataset",
            "ocr_required": True
        }
    }
    
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def prepare_dataset(self, 
                       dataset_type: str,
                       input_path: str,
                       output_path: str,
                       ocr_engine: str = "auto",
                       ocr_languages: Optional[list] = None,
                       **kwargs) -> bool:
        """
        Prepare a specific dataset type.
        
        Args:
            dataset_type: Type of dataset to prepare
            input_path: Path to raw dataset
            output_path: Path for processed output
            ocr_engine: OCR engine to use
            ocr_languages: Languages for OCR processing
            **kwargs: Additional configuration options
            
        Returns:
            bool: Success status
        """
        if dataset_type not in self.SUPPORTED_DATASETS:
            self.logger.error(f"Unsupported dataset type: {dataset_type}")
            self.logger.info(f"Supported types: {list(self.SUPPORTED_DATASETS.keys())}")
            return False
        
        dataset_info = self.SUPPORTED_DATASETS[dataset_type]
        self.logger.info(f"Preparing {dataset_info['name']} dataset...")
        
        # Prepare OCR processor if needed
        ocr_processor = None
        if dataset_info["ocr_required"]:
            self.logger.info("Setting up OCR processor...")
            try:
                ocr_processor = create_ocr_processor(
                    engine=ocr_engine,
                    languages=ocr_languages or ["en"],
                    confidence_threshold=kwargs.get("ocr_confidence", 0.5)
                )
                self.logger.info(f"OCR processor ready: {ocr_engine}")
            except Exception as e:
                self.logger.error(f"Failed to initialize OCR processor: {e}")
                return False
        
        # Create dataset configuration
        try:
            config = ConversionConfig(
                source_path=input_path,
                output_path=output_path,
                dataset_name=f"{dataset_type}_prepared",
                format_type="custom",
                ocr_engine=ocr_engine,
                ocr_languages=ocr_languages or ["en"],
                **kwargs
            )
            
            # Initialize converter
            converter_class = dataset_info["converter"]
            converter = converter_class(config)
            
            # Check if format is supported
            if not converter.detect_format():
                self.logger.error(f"Input format not supported for {dataset_type}")
                return False
            
            # Run conversion
            self.logger.info("Starting dataset conversion...")
            success = converter.convert()
            
            if success:
                self.logger.info(f"✓ Dataset preparation completed successfully!")
                self.logger.info(f"Output directory: {output_path}")
                
                # Print statistics if available
                stats = converter.get_stats()
                if stats:
                    self.logger.info("Conversion Statistics:")
                    for key, value in stats.items():
                        self.logger.info(f"  {key}: {value}")
                
                return True
            else:
                self.logger.error("Dataset preparation failed")
                return False
                
        except Exception as e:
            self.logger.error(f"Error during dataset preparation: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def list_supported_datasets(self):
        """Print information about supported datasets."""
        print("\nSupported Document Image Datasets:")
        print("=" * 50)
        
        for dataset_id, info in self.SUPPORTED_DATASETS.items():
            print(f"\nDataset ID: {dataset_id}")
            print(f"Name: {info['name']}")
            print(f"Description: {info['description']}")
            print(f"OCR Required: {'Yes' if info['ocr_required'] else 'No'}")
    
    def prepare_vietnamese_sample_dataset(self, output_path: str) -> bool:
        """
        Create a sample Vietnamese Administrative Documents dataset.
        
        Args:
            output_path: Where to create the sample dataset
            
        Returns:
            bool: Success status
        """
        self.logger.info("Creating Vietnamese Administrative Documents sample dataset...")
        
        try:
            from scripts.create_vietnamese_sample import VietnameseSampleCreator
            
            creator = VietnameseSampleCreator(output_path)
            success = creator.create_sample_dataset()
            
            if success:
                self.logger.info(f"✓ Sample dataset created at: {output_path}")
                return True
            else:
                self.logger.error("Failed to create sample dataset")
                return False
                
        except ImportError:
            self.logger.error("Vietnamese sample creator not available")
            return False
        except Exception as e:
            self.logger.error(f"Error creating sample dataset: {e}")
            return False


def main():
    """Main CLI interface."""
    parser = argparse.ArgumentParser(
        description="NewAIBench Document Image Dataset Preparation Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Prepare DocVQA dataset
  python prepare_datasets.py docvqa /path/to/docvqa /path/to/output

  # Prepare Vietnamese admin docs with PaddleOCR
  python prepare_datasets.py vietnamese_admin /path/to/docs /path/to/output \\
    --ocr-engine paddleocr --ocr-languages vi en

  # Create Vietnamese sample dataset
  python prepare_datasets.py --create-vietnamese-sample /path/to/sample

  # List supported datasets
  python prepare_datasets.py --list-datasets
        """
    )
    
    # Dataset type argument (optional for some commands)
    parser.add_argument(
        "dataset_type",
        nargs='?',
        help="Type of dataset to prepare (docvqa, vietnamese_admin)"
    )
    
    parser.add_argument(
        "input_path",
        nargs='?',
        help="Path to raw dataset directory"
    )
    
    parser.add_argument(
        "output_path",
        nargs='?',
        help="Path for processed dataset output"
    )
    
    # OCR options
    parser.add_argument(
        "--ocr-engine",
        choices=["auto", "tesseract", "easyocr", "paddleocr"],
        default="auto",
        help="OCR engine to use (default: auto)"
    )
    
    parser.add_argument(
        "--ocr-languages",
        nargs="+",
        default=["en"],
        help="Languages for OCR processing (default: en)"
    )
    
    parser.add_argument(
        "--ocr-confidence",
        type=float,
        default=0.5,
        help="OCR confidence threshold (default: 0.5)"
    )
    
    # Processing options
    parser.add_argument(
        "--max-samples",
        type=int,
        help="Maximum number of samples to process"
    )
    
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Number of worker threads for parallel processing"
    )
    
    # Special commands
    parser.add_argument(
        "--list-datasets",
        action="store_true",
        help="List supported dataset types"
    )
    
    parser.add_argument(
        "--create-vietnamese-sample",
        metavar="OUTPUT_PATH",
        help="Create a sample Vietnamese administrative documents dataset"
    )
    
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )
    
    args = parser.parse_args()
    
    # Setup logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Initialize preparer
    preparer = DatasetPreparer()
    
    # Handle special commands
    if args.list_datasets:
        preparer.list_supported_datasets()
        return
    
    if args.create_vietnamese_sample:
        success = preparer.prepare_vietnamese_sample_dataset(args.create_vietnamese_sample)
        sys.exit(0 if success else 1)
    
    # Validate required arguments for dataset preparation
    if not args.dataset_type or not args.input_path or not args.output_path:
        parser.error("dataset_type, input_path, and output_path are required for dataset preparation")
    
    # Prepare dataset
    success = preparer.prepare_dataset(
        dataset_type=args.dataset_type,
        input_path=args.input_path,
        output_path=args.output_path,
        ocr_engine=args.ocr_engine,
        ocr_languages=args.ocr_languages,
        ocr_confidence=args.ocr_confidence,
        max_samples=args.max_samples,
        workers=args.workers
    )
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
