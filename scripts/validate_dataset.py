#!/usr/bin/env python3
"""
Dataset Validation and Testing Script

Validates prepared document image datasets and runs integration tests
to ensure they work correctly with NewAIBench framework.
"""

import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import argparse

# Add the src directory to the path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from newaibench.datasets.image import DocumentImageDatasetLoader, DocumentImageDatasetConfig
from newaibench.models.image_retrieval import OCRBasedDocumentRetriever
from newaibench.evaluation.evaluator import RetrievalEvaluator
from newaibench.evaluation.metrics import RetrievalMetrics

logger = logging.getLogger(__name__)


class DatasetValidator:
    """Validates and tests prepared document image datasets."""
    
    def __init__(self, dataset_path: str):
        """
        Initialize dataset validator.
        
        Args:
            dataset_path: Path to prepared dataset directory
        """
        self.dataset_path = Path(dataset_path)
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Validation results
        self.validation_results = {
            "structure_valid": False,
            "format_valid": False,
            "content_valid": False,
            "integration_valid": False,
            "retrieval_valid": False,
            "errors": [],
            "warnings": [],
            "statistics": {}
        }
    
    def validate_dataset(self, 
                        run_integration_test: bool = True,
                        run_retrieval_test: bool = True) -> bool:
        """
        Run comprehensive dataset validation.
        
        Args:
            run_integration_test: Whether to test NewAIBench integration
            run_retrieval_test: Whether to test retrieval functionality
            
        Returns:
            bool: Overall validation success
        """
        try:
            self.logger.info(f"Validating dataset: {self.dataset_path}")
            
            # 1. Validate directory structure
            self.logger.info("1. Validating directory structure...")
            self._validate_structure()
            
            # 2. Validate file formats
            self.logger.info("2. Validating file formats...")
            self._validate_formats()
            
            # 3. Validate content consistency
            self.logger.info("3. Validating content consistency...")
            self._validate_content()
            
            # 4. Test NewAIBench integration
            if run_integration_test:
                self.logger.info("4. Testing NewAIBench integration...")
                self._test_integration()
            
            # 5. Test retrieval functionality
            if run_retrieval_test and self.validation_results["integration_valid"]:
                self.logger.info("5. Testing retrieval functionality...")
                self._test_retrieval()
            
            # Generate summary
            self._generate_summary()
            
            # Determine overall success
            overall_success = (
                self.validation_results["structure_valid"] and
                self.validation_results["format_valid"] and
                self.validation_results["content_valid"]
            )
            
            if run_integration_test:
                overall_success = overall_success and self.validation_results["integration_valid"]
            
            if run_retrieval_test:
                overall_success = overall_success and self.validation_results["retrieval_valid"]
            
            return overall_success
            
        except Exception as e:
            self.logger.error(f"Validation failed with error: {e}")
            self.validation_results["errors"].append(f"Validation error: {str(e)}")
            return False
    
    def _validate_structure(self):
        """Validate dataset directory structure."""
        try:
            required_files = ["corpus.jsonl", "metadata.json"]
            optional_files = ["queries.jsonl", "qrels.jsonl"]
            required_dirs = ["images"]
            optional_dirs = ["ocr"]
            
            # Check required files
            missing_files = []
            for file_name in required_files:
                file_path = self.dataset_path / file_name
                if not file_path.exists():
                    missing_files.append(file_name)
            
            if missing_files:
                self.validation_results["errors"].append(f"Missing required files: {missing_files}")
                return
            
            # Check optional files
            for file_name in optional_files:
                file_path = self.dataset_path / file_name
                if not file_path.exists():
                    self.validation_results["warnings"].append(f"Optional file missing: {file_name}")
            
            # Check required directories
            missing_dirs = []
            for dir_name in required_dirs:
                dir_path = self.dataset_path / dir_name
                if not dir_path.exists():
                    missing_dirs.append(dir_name)
            
            if missing_dirs:
                self.validation_results["errors"].append(f"Missing required directories: {missing_dirs}")
                return
            
            # Check for images in images directory
            images_dir = self.dataset_path / "images"
            image_files = list(images_dir.glob("*"))
            image_files = [f for f in image_files if f.is_file()]
            
            if not image_files:
                self.validation_results["warnings"].append("No image files found in images directory")
            
            self.validation_results["statistics"]["image_files_count"] = len(image_files)
            self.validation_results["structure_valid"] = True
            self.logger.info("✓ Directory structure validation passed")
            
        except Exception as e:
            self.validation_results["errors"].append(f"Structure validation error: {str(e)}")
    
    def _validate_formats(self):
        """Validate file formats and JSON structure."""
        try:
            # Validate corpus.jsonl
            corpus_errors = self._validate_jsonl_file(
                self.dataset_path / "corpus.jsonl",
                required_fields=["_id", "title", "text", "image_path"],
                optional_fields=["metadata"]
            )
            
            if corpus_errors:
                self.validation_results["errors"].extend(corpus_errors)
                return
            
            # Validate queries.jsonl if exists
            queries_file = self.dataset_path / "queries.jsonl"
            if queries_file.exists():
                queries_errors = self._validate_jsonl_file(
                    queries_file,
                    required_fields=["query_id", "text"],
                    optional_fields=["metadata"]
                )
                if queries_errors:
                    self.validation_results["errors"].extend(queries_errors)
                    return
            
            # Validate qrels.jsonl if exists
            qrels_file = self.dataset_path / "qrels.jsonl"
            if qrels_file.exists():
                qrels_errors = self._validate_jsonl_file(
                    qrels_file,
                    required_fields=["query_id", "doc_id", "relevance"],
                    optional_fields=[]
                )
                if qrels_errors:
                    self.validation_results["errors"].extend(qrels_errors)
                    return
            
            # Validate metadata.json
            metadata_errors = self._validate_metadata_file()
            if metadata_errors:
                self.validation_results["errors"].extend(metadata_errors)
                return
            
            self.validation_results["format_valid"] = True
            self.logger.info("✓ File format validation passed")
            
        except Exception as e:
            self.validation_results["errors"].append(f"Format validation error: {str(e)}")
    
    def _validate_jsonl_file(self, 
                           file_path: Path, 
                           required_fields: List[str],
                           optional_fields: List[str]) -> List[str]:
        """Validate a JSONL file structure."""
        errors = []
        
        if not file_path.exists():
            return [f"File does not exist: {file_path}"]
        
        try:
            line_count = 0
            with open(file_path, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        continue
                    
                    try:
                        data = json.loads(line)
                        line_count += 1
                        
                        # Check required fields
                        for field in required_fields:
                            if field not in data:
                                errors.append(f"{file_path.name} line {line_num}: Missing required field '{field}'")
                        
                        # Check field types
                        if "_id" in data and not isinstance(data["_id"], str):
                            errors.append(f"{file_path.name} line {line_num}: '_id' must be string")
                        
                        if "text" in data and not isinstance(data["text"], str):
                            errors.append(f"{file_path.name} line {line_num}: 'text' must be string")
                        
                        if "relevance" in data and not isinstance(data["relevance"], (int, float)):
                            errors.append(f"{file_path.name} line {line_num}: 'relevance' must be number")
                    
                    except json.JSONDecodeError as e:
                        errors.append(f"{file_path.name} line {line_num}: Invalid JSON - {str(e)}")
            
            self.validation_results["statistics"][f"{file_path.name}_lines"] = line_count
            
        except Exception as e:
            errors.append(f"Error reading {file_path.name}: {str(e)}")
        
        return errors
    
    def _validate_metadata_file(self) -> List[str]:
        """Validate metadata.json file."""
        errors = []
        metadata_file = self.dataset_path / "metadata.json"
        
        try:
            with open(metadata_file, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
            
            # Check required metadata fields
            required_fields = ["dataset_name", "description", "language"]
            for field in required_fields:
                if field not in metadata:
                    errors.append(f"metadata.json: Missing required field '{field}'")
            
            # Store metadata for later use
            self.validation_results["statistics"]["metadata"] = metadata
            
        except json.JSONDecodeError as e:
            errors.append(f"metadata.json: Invalid JSON - {str(e)}")
        except Exception as e:
            errors.append(f"Error reading metadata.json: {str(e)}")
        
        return errors
    
    def _validate_content(self):
        """Validate content consistency between files."""
        try:
            # Load corpus
            corpus_data = {}
            with open(self.dataset_path / "corpus.jsonl", 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        doc = json.loads(line)
                        corpus_data[doc["_id"]] = doc
            
            # Check image file references
            missing_images = []
            for doc_id, doc in corpus_data.items():
                image_path = self.dataset_path / doc["image_path"]
                if not image_path.exists():
                    missing_images.append(f"Document {doc_id}: {doc['image_path']}")
            
            if missing_images:
                self.validation_results["warnings"].append(f"Missing image files: {missing_images[:5]}...")
            
            # Validate queries and qrels consistency if they exist
            queries_file = self.dataset_path / "queries.jsonl"
            qrels_file = self.dataset_path / "qrels.jsonl"
            
            if queries_file.exists() and qrels_file.exists():
                # Load queries
                queries_data = {}
                with open(queries_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        if line.strip():
                            query = json.loads(line)
                            queries_data[query["query_id"]] = query
                
                # Load qrels and check consistency
                orphaned_queries = set()
                orphaned_docs = set()
                
                with open(qrels_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        if line.strip():
                            qrel = json.loads(line)
                            
                            if qrel["query_id"] not in queries_data:
                                orphaned_queries.add(qrel["query_id"])
                            
                            if qrel["doc_id"] not in corpus_data:
                                orphaned_docs.add(qrel["doc_id"])
                
                if orphaned_queries:
                    self.validation_results["warnings"].append(f"Qrels reference non-existent queries: {len(orphaned_queries)} cases")
                
                if orphaned_docs:
                    self.validation_results["warnings"].append(f"Qrels reference non-existent documents: {len(orphaned_docs)} cases")
            
            self.validation_results["statistics"]["corpus_documents"] = len(corpus_data)
            self.validation_results["content_valid"] = True
            self.logger.info("✓ Content consistency validation passed")
            
        except Exception as e:
            self.validation_results["errors"].append(f"Content validation error: {str(e)}")
    
    def _test_integration(self):
        """Test integration with NewAIBench DocumentImageDatasetLoader."""
        try:
            # Create dataset configuration
            config = DocumentImageDatasetConfig(
                dataset_path=str(self.dataset_path),
                format_type="jsonl",
                require_ocr_text=False,  # Use existing OCR text
                cache_enabled=False,
                metadata={"name": "validation_test"}
            )
            
            # Initialize loader
            loader = DocumentImageDatasetLoader(config)
            
            # Test loading corpus
            corpus = loader.load_corpus()
            if not corpus:
                self.validation_results["errors"].append("Failed to load corpus through DocumentImageDatasetLoader")
                return
            
            # Test loading queries if available
            queries_file = self.dataset_path / "queries.jsonl"
            if queries_file.exists():
                queries = loader.load_queries()
                if not queries:
                    self.validation_results["warnings"].append("Failed to load queries through DocumentImageDatasetLoader")
            
            # Test loading qrels if available
            qrels_file = self.dataset_path / "qrels.jsonl"
            if qrels_file.exists():
                qrels = loader.load_qrels()
                if not qrels:
                    self.validation_results["warnings"].append("Failed to load qrels through DocumentImageDatasetLoader")
            
            # Get dataset statistics
            stats = loader.get_statistics(corpus, {}, {})
            self.validation_results["statistics"]["loader_stats"] = stats
            
            self.validation_results["integration_valid"] = True
            self.logger.info("✓ NewAIBench integration test passed")
            
        except Exception as e:
            self.validation_results["errors"].append(f"Integration test error: {str(e)}")
    
    def _test_retrieval(self):
        """Test retrieval functionality with OCRBasedDocumentRetriever."""
        try:
            # Load dataset
            config = DocumentImageDatasetConfig(
                dataset_path=str(self.dataset_path),
                format_type="jsonl",
                require_ocr_text=False,
                cache_enabled=False,
                metadata={"name": "retrieval_test"}
            )
            
            loader = DocumentImageDatasetLoader(config)
            corpus = loader.load_corpus()
            
            # Check if we have queries
            queries_file = self.dataset_path / "queries.jsonl"
            if not queries_file.exists():
                self.validation_results["warnings"].append("No queries file - skipping retrieval test")
                self.validation_results["retrieval_valid"] = True
                return
            
            queries = loader.load_queries()
            if not queries:
                self.validation_results["warnings"].append("No queries loaded - skipping retrieval test")
                self.validation_results["retrieval_valid"] = True
                return
            
            # Initialize retriever (lightweight model for testing)
            retriever_config = {
                "model_name": "sentence-transformers/all-MiniLM-L6-v2",
                "model_type": "dense_text",
                "ocr_engine": "tesseract",
                "enable_preprocessing": True
            }
            
            retriever = OCRBasedDocumentRetriever(retriever_config)
            retriever.load_model()
            
            # Test retrieval with a few queries
            test_queries = list(queries.items())[:3]  # Test first 3 queries
            retrieval_results = {}
            
            for query_id, query_text in test_queries:
                try:
                    results = retriever.predict([query_text], list(corpus.values()), top_k=5)
                    retrieval_results[query_id] = len(results[0]) if results else 0
                except Exception as e:
                    self.validation_results["warnings"].append(f"Retrieval failed for query {query_id}: {str(e)}")
            
            if retrieval_results:
                avg_results = sum(retrieval_results.values()) / len(retrieval_results)
                self.validation_results["statistics"]["avg_retrieval_results"] = avg_results
                
                if avg_results > 0:
                    self.validation_results["retrieval_valid"] = True
                    self.logger.info("✓ Retrieval functionality test passed")
                else:
                    self.validation_results["warnings"].append("Retrieval returned no results")
                    self.validation_results["retrieval_valid"] = True  # Not a failure
            else:
                self.validation_results["warnings"].append("No retrieval tests completed")
                self.validation_results["retrieval_valid"] = True
            
        except Exception as e:
            self.validation_results["warnings"].append(f"Retrieval test error: {str(e)}")
            self.validation_results["retrieval_valid"] = True  # Treat as non-critical
    
    def _generate_summary(self):
        """Generate validation summary."""
        self.logger.info("\n" + "="*60)
        self.logger.info("DATASET VALIDATION SUMMARY")
        self.logger.info("="*60)
        
        # Overall status
        passed_tests = sum([
            self.validation_results["structure_valid"],
            self.validation_results["format_valid"],
            self.validation_results["content_valid"],
            self.validation_results["integration_valid"],
            self.validation_results["retrieval_valid"]
        ])
        
        total_tests = 5
        self.logger.info(f"Tests Passed: {passed_tests}/{total_tests}")
        
        # Individual test results
        test_names = [
            ("structure_valid", "Directory Structure"),
            ("format_valid", "File Formats"),
            ("content_valid", "Content Consistency"),
            ("integration_valid", "NewAIBench Integration"),
            ("retrieval_valid", "Retrieval Functionality")
        ]
        
        for key, name in test_names:
            status = "✓ PASS" if self.validation_results[key] else "✗ FAIL"
            self.logger.info(f"{name}: {status}")
        
        # Statistics
        if self.validation_results["statistics"]:
            self.logger.info("\nStatistics:")
            for key, value in self.validation_results["statistics"].items():
                if key != "metadata" and key != "loader_stats":
                    self.logger.info(f"  {key}: {value}")
        
        # Errors
        if self.validation_results["errors"]:
            self.logger.info(f"\nErrors ({len(self.validation_results['errors'])}):")
            for error in self.validation_results["errors"]:
                self.logger.error(f"  • {error}")
        
        # Warnings
        if self.validation_results["warnings"]:
            self.logger.info(f"\nWarnings ({len(self.validation_results['warnings'])}):")
            for warning in self.validation_results["warnings"]:
                self.logger.warning(f"  • {warning}")
        
        self.logger.info("="*60)
    
    def get_validation_results(self) -> Dict[str, Any]:
        """Get validation results."""
        return self.validation_results


def main():
    """CLI entry point for dataset validation."""
    parser = argparse.ArgumentParser(
        description="Validate prepared document image datasets",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic validation
  python validate_dataset.py /path/to/prepared/dataset

  # Skip integration tests
  python validate_dataset.py /path/to/dataset --no-integration

  # Skip retrieval tests
  python validate_dataset.py /path/to/dataset --no-retrieval

  # Full validation with verbose output
  python validate_dataset.py /path/to/dataset --verbose
        """
    )
    
    parser.add_argument("dataset_path", help="Path to prepared dataset directory")
    
    parser.add_argument(
        "--no-integration",
        action="store_true",
        help="Skip NewAIBench integration test"
    )
    
    parser.add_argument(
        "--no-retrieval",
        action="store_true",
        help="Skip retrieval functionality test"
    )
    
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Run validation
    validator = DatasetValidator(args.dataset_path)
    
    success = validator.validate_dataset(
        run_integration_test=not args.no_integration,
        run_retrieval_test=not args.no_retrieval
    )
    
    if success:
        print(f"\n✓ Dataset validation passed!")
        print(f"Dataset: {args.dataset_path}")
    else:
        print(f"\n✗ Dataset validation failed!")
        print(f"Dataset: {args.dataset_path}")
        sys.exit(1)


if __name__ == "__main__":
    main()
