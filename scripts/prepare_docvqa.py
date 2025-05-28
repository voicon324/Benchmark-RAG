#!/usr/bin/env python3
"""
DocVQA Dataset Preparation Script

Prepares the DocVQA (Document Visual Question Answering) dataset for use with NewAIBench.
Handles document extraction, OCR processing, and query generation from question-answer pairs.
"""

import json
import logging
import shutil
from pathlib import Path
from typing import Dict, List, Any, Optional, Set
import argparse
import sys

# Add the src directory to the path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from newaibench.utils.ocr_processor import create_ocr_processor
from newaibench.datasets.config import DatasetConfig

logger = logging.getLogger(__name__)


class DocVQAPreparator:
    """Handles DocVQA dataset preparation and processing."""
    
    def __init__(self, input_path: str, output_path: str, ocr_engine: str = "auto"):
        """
        Initialize DocVQA preparator.
        
        Args:
            input_path: Path to raw DocVQA dataset
            output_path: Path for processed output
            ocr_engine: OCR engine to use
        """
        self.input_path = Path(input_path)
        self.output_path = Path(output_path)
        self.ocr_engine = ocr_engine
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # OCR processor
        self.ocr_processor = None
        
        # Statistics
        self.stats = {
            "total_questions": 0,
            "unique_documents": 0,
            "processed_documents": 0,
            "ocr_successful": 0,
            "ocr_failed": 0,
            "queries_generated": 0,
            "qrels_generated": 0
        }
    
    def prepare_dataset(self, 
                       max_documents: Optional[int] = None,
                       extract_ocr: bool = True,
                       create_queries: bool = True) -> bool:
        """
        Prepare the complete DocVQA dataset.
        
        Args:
            max_documents: Maximum number of documents to process (for testing)
            extract_ocr: Whether to extract OCR text
            create_queries: Whether to create query files
            
        Returns:
            bool: Success status
        """
        try:
            self.logger.info("Starting DocVQA dataset preparation...")
            
            # Validate input
            if not self._validate_input():
                return False
            
            # Create output structure
            self._create_output_structure()
            
            # Initialize OCR processor if needed
            if extract_ocr:
                self._initialize_ocr_processor()
            
            # Collect all DocVQA data
            self.logger.info("Collecting DocVQA data...")
            docvqa_data = self._collect_docvqa_data()
            
            # Extract unique documents
            self.logger.info("Extracting unique documents...")
            documents = self._extract_unique_documents(docvqa_data)
            
            # Limit documents if specified
            if max_documents and len(documents) > max_documents:
                self.logger.info(f"Limiting to {max_documents} documents for testing")
                documents = dict(list(documents.items())[:max_documents])
            
            # Process images and extract OCR
            if extract_ocr:
                self.logger.info("Processing document images and extracting OCR...")
                self._process_document_images(documents)
            
            # Create corpus file
            self.logger.info("Creating corpus file...")
            self._create_corpus_file(documents)
            
            # Create queries and qrels from Q&A pairs
            if create_queries:
                self.logger.info("Creating queries and relevance judgments...")
                self._create_queries_and_qrels(docvqa_data, documents)
            
            # Create metadata
            self._create_metadata_file()
            
            # Print statistics
            self._print_statistics()
            
            self.logger.info("✓ DocVQA dataset preparation completed successfully!")
            return True
            
        except Exception as e:
            self.logger.error(f"Error during dataset preparation: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def _validate_input(self) -> bool:
        """Validate input dataset structure."""
        if not self.input_path.exists():
            self.logger.error(f"Input path does not exist: {self.input_path}")
            return False
        
        # Check for DocVQA files
        required_patterns = [
            "*.json",  # Question files
            "**/documents/*.png",  # Document images
            "**/documents/*.jpg"   # Document images
        ]
        
        found_files = False
        for pattern in required_patterns:
            if list(self.input_path.glob(pattern)):
                found_files = True
                break
        
        if not found_files:
            self.logger.error("No DocVQA files found in input directory")
            self.logger.info("Expected structure: JSON files with questions and documents/ folder with images")
            return False
        
        return True
    
    def _create_output_structure(self):
        """Create output directory structure."""
        self.output_path.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories
        (self.output_path / "images").mkdir(exist_ok=True)
        (self.output_path / "ocr").mkdir(exist_ok=True)
        
        self.logger.info(f"Created output structure at: {self.output_path}")
    
    def _initialize_ocr_processor(self):
        """Initialize OCR processor."""
        try:
            self.ocr_processor = create_ocr_processor(
                engine=self.ocr_engine,
                languages=["en"],  # DocVQA is primarily English
                confidence_threshold=0.5
            )
            self.logger.info(f"OCR processor initialized: {self.ocr_engine}")
        except Exception as e:
            self.logger.error(f"Failed to initialize OCR processor: {e}")
            raise
    
    def _collect_docvqa_data(self) -> List[Dict]:
        """Collect all DocVQA question-answer data."""
        all_data = []
        
        # Find all JSON files
        json_files = list(self.input_path.rglob("*.json"))
        
        for json_file in json_files:
            # Skip metadata files
            if json_file.name.lower() in ["metadata.json", "info.json", "readme.json"]:
                continue
            
            try:
                self.logger.info(f"Loading {json_file}")
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # Handle different DocVQA formats
                if isinstance(data, dict):
                    if "data" in data:
                        all_data.extend(data["data"])
                    elif "questions" in data:
                        all_data.extend(data["questions"])
                    else:
                        # Single question format
                        all_data.append(data)
                elif isinstance(data, list):
                    all_data.extend(data)
                
            except Exception as e:
                self.logger.warning(f"Error loading {json_file}: {e}")
                continue
        
        self.stats["total_questions"] = len(all_data)
        self.logger.info(f"Collected {len(all_data)} questions from {len(json_files)} files")
        
        return all_data
    
    def _extract_unique_documents(self, docvqa_data: List[Dict]) -> Dict[str, Dict]:
        """Extract unique documents from DocVQA data."""
        documents = {}
        seen_docs = set()
        
        for item in docvqa_data:
            # Extract document ID
            doc_id = None
            for key in ["docId", "doc_id", "document_id", "ucsf_document_id"]:
                if key in item:
                    doc_id = str(item[key])
                    break
            
            if not doc_id or doc_id in seen_docs:
                continue
            
            # Extract image path
            image_path = ""
            for key in ["image", "image_path", "document_image"]:
                if key in item:
                    image_path = item[key]
                    break
            
            if not image_path:
                # Try to construct image path from doc_id
                image_path = f"documents/{doc_id}.png"
            
            # Find actual image file
            actual_image_path = self._find_document_image(image_path, doc_id)
            
            if actual_image_path:
                documents[doc_id] = {
                    "doc_id": doc_id,
                    "image_path": str(actual_image_path),
                    "title": f"Document {doc_id}",
                    "document_type": item.get("document_type", "document"),
                    "questions": [],  # Will be populated later
                    "metadata": {
                        "source": "docvqa",
                        "original_doc_id": doc_id
                    }
                }
                seen_docs.add(doc_id)
        
        self.stats["unique_documents"] = len(documents)
        self.logger.info(f"Found {len(documents)} unique documents")
        
        return documents
    
    def _find_document_image(self, image_path: str, doc_id: str) -> Optional[Path]:
        """Find the actual document image file."""
        # Try direct path
        direct_path = self.input_path / image_path
        if direct_path.exists():
            return direct_path
        
        # Try common patterns
        patterns = [
            f"**/documents/{doc_id}.*",
            f"**/images/{doc_id}.*",
            f"**/{doc_id}.*",
            f"**/documents/*{doc_id}*.*",
            f"**/images/*{doc_id}*.*"
        ]
        
        for pattern in patterns:
            matches = list(self.input_path.glob(pattern))
            if matches:
                # Filter for image extensions
                for match in matches:
                    if match.suffix.lower() in ['.png', '.jpg', '.jpeg', '.tiff', '.pdf']:
                        return match
        
        self.logger.warning(f"Could not find image for document {doc_id}")
        return None
    
    def _process_document_images(self, documents: Dict[str, Dict]):
        """Process document images and extract OCR text."""
        if not self.ocr_processor:
            self.logger.warning("OCR processor not initialized, skipping OCR extraction")
            return
        
        for doc_id, doc_info in documents.items():
            try:
                image_path = Path(doc_info["image_path"])
                
                # Copy image to output directory
                output_image_path = self.output_path / "images" / f"{doc_id}{image_path.suffix}"
                shutil.copy2(image_path, output_image_path)
                doc_info["output_image_path"] = str(output_image_path.relative_to(self.output_path))
                
                # Extract OCR text
                self.logger.debug(f"Processing OCR for {doc_id}")
                ocr_result = self.ocr_processor.process_image(image_path)
                
                if ocr_result["text"]:
                    doc_info["ocr_text"] = ocr_result["text"]
                    doc_info["ocr_confidence"] = ocr_result.get("confidence", 0)
                    
                    # Save OCR text to file
                    ocr_file = self.output_path / "ocr" / f"{doc_id}.txt"
                    with open(ocr_file, 'w', encoding='utf-8') as f:
                        f.write(ocr_result["text"])
                    
                    self.stats["ocr_successful"] += 1
                else:
                    doc_info["ocr_text"] = ""
                    doc_info["ocr_confidence"] = 0
                    self.stats["ocr_failed"] += 1
                
                self.stats["processed_documents"] += 1
                
                if self.stats["processed_documents"] % 10 == 0:
                    self.logger.info(f"Processed {self.stats['processed_documents']} documents...")
                
            except Exception as e:
                self.logger.error(f"Error processing document {doc_id}: {e}")
                self.stats["ocr_failed"] += 1
                continue
    
    def _create_corpus_file(self, documents: Dict[str, Dict]):
        """Create the corpus.jsonl file."""
        corpus_file = self.output_path / "corpus.jsonl"
        
        with open(corpus_file, 'w', encoding='utf-8') as f:
            for doc_id, doc_info in documents.items():
                corpus_entry = {
                    "_id": doc_id,
                    "title": doc_info["title"],
                    "text": doc_info.get("ocr_text", ""),
                    "image_path": doc_info.get("output_image_path", ""),
                    "metadata": {
                        **doc_info["metadata"],
                        "document_type": doc_info.get("document_type", "document"),
                        "ocr_confidence": doc_info.get("ocr_confidence", 0),
                        "has_ocr": bool(doc_info.get("ocr_text", "").strip())
                    }
                }
                
                f.write(json.dumps(corpus_entry, ensure_ascii=False) + '\n')
        
        self.logger.info(f"Created corpus file with {len(documents)} documents")
    
    def _create_queries_and_qrels(self, docvqa_data: List[Dict], documents: Dict[str, Dict]):
        """Create queries and relevance judgments from Q&A pairs."""
        queries_file = self.output_path / "queries.jsonl"
        qrels_file = self.output_path / "qrels.jsonl"
        
        queries = []
        qrels = []
        query_id = 1
        
        # Group questions by document
        doc_questions = {}
        for item in docvqa_data:
            doc_id = None
            for key in ["docId", "doc_id", "document_id", "ucsf_document_id"]:
                if key in item:
                    doc_id = str(item[key])
                    break
            
            if doc_id and doc_id in documents:
                if doc_id not in doc_questions:
                    doc_questions[doc_id] = []
                doc_questions[doc_id].append(item)
        
        # Create queries from questions
        for doc_id, questions in doc_questions.items():
            for question_item in questions:
                # Extract question text
                question = ""
                for key in ["question", "text", "query"]:
                    if key in question_item:
                        question = question_item[key]
                        break
                
                if not question:
                    continue
                
                # Create query
                query_entry = {
                    "query_id": f"q{query_id:04d}",
                    "text": question
                }
                queries.append(query_entry)
                
                # Create relevance judgment
                # For DocVQA, if there's a question about a document, 
                # that document is highly relevant (relevance = 2)
                qrel_entry = {
                    "query_id": f"q{query_id:04d}",
                    "doc_id": doc_id,
                    "relevance": 2
                }
                qrels.append(qrel_entry)
                
                # Add answer information to metadata if available
                if "answers" in question_item or "answer" in question_item:
                    answers = question_item.get("answers", [question_item.get("answer", "")])
                    query_entry["metadata"] = {
                        "source_doc_id": doc_id,
                        "answers": answers if isinstance(answers, list) else [answers]
                    }
                
                query_id += 1
        
        # Write queries file
        with open(queries_file, 'w', encoding='utf-8') as f:
            for query in queries:
                f.write(json.dumps(query, ensure_ascii=False) + '\n')
        
        # Write qrels file
        with open(qrels_file, 'w', encoding='utf-8') as f:
            for qrel in qrels:
                f.write(json.dumps(qrel, ensure_ascii=False) + '\n')
        
        self.stats["queries_generated"] = len(queries)
        self.stats["qrels_generated"] = len(qrels)
        
        self.logger.info(f"Created {len(queries)} queries and {len(qrels)} relevance judgments")
    
    def _create_metadata_file(self):
        """Create dataset metadata file."""
        metadata = {
            "dataset_name": "DocVQA",
            "description": "Document Visual Question Answering dataset prepared for NewAIBench",
            "source": "DocVQA Dataset",
            "language": "english",
            "created_date": "2024-01-01",  # Update based on actual creation
            "version": "1.0.0",
            "ocr_engine": self.ocr_engine,
            "statistics": self.stats,
            "format": {
                "corpus": "jsonl",
                "queries": "jsonl",
                "qrels": "jsonl",
                "images": "png/jpg"
            },
            "fields": {
                "corpus": ["_id", "title", "text", "image_path", "metadata"],
                "queries": ["query_id", "text", "metadata"],
                "qrels": ["query_id", "doc_id", "relevance"]
            }
        }
        
        metadata_file = self.output_path / "metadata.json"
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)
        
        self.logger.info("Created metadata file")
    
    def _print_statistics(self):
        """Print preparation statistics."""
        self.logger.info("\n" + "="*50)
        self.logger.info("DOCVQA PREPARATION STATISTICS")
        self.logger.info("="*50)
        
        for key, value in self.stats.items():
            self.logger.info(f"{key.replace('_', ' ').title()}: {value}")
        
        # Calculate success rates
        if self.stats["processed_documents"] > 0:
            ocr_success_rate = (self.stats["ocr_successful"] / self.stats["processed_documents"]) * 100
            self.logger.info(f"OCR Success Rate: {ocr_success_rate:.1f}%")
        
        self.logger.info("="*50)


def main():
    """CLI entry point for DocVQA preparation."""
    parser = argparse.ArgumentParser(
        description="Prepare DocVQA dataset for NewAIBench",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic preparation
  python prepare_docvqa.py /path/to/docvqa /path/to/output

  # With specific OCR engine
  python prepare_docvqa.py /path/to/docvqa /path/to/output --ocr-engine paddleocr

  # Limit documents for testing
  python prepare_docvqa.py /path/to/docvqa /path/to/output --max-documents 100

  # Skip OCR extraction
  python prepare_docvqa.py /path/to/docvqa /path/to/output --no-ocr
        """
    )
    
    parser.add_argument("input_path", help="Path to raw DocVQA dataset")
    parser.add_argument("output_path", help="Path for processed output")
    
    parser.add_argument(
        "--ocr-engine",
        choices=["auto", "tesseract", "easyocr", "paddleocr"],
        default="auto",
        help="OCR engine to use (default: auto)"
    )
    
    parser.add_argument(
        "--max-documents",
        type=int,
        help="Maximum number of documents to process (for testing)"
    )
    
    parser.add_argument(
        "--no-ocr",
        action="store_true",
        help="Skip OCR extraction"
    )
    
    parser.add_argument(
        "--no-queries",
        action="store_true",
        help="Skip query generation"
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
    
    # Run preparation
    preparator = DocVQAPreparator(args.input_path, args.output_path, args.ocr_engine)
    
    success = preparator.prepare_dataset(
        max_documents=args.max_documents,
        extract_ocr=not args.no_ocr,
        create_queries=not args.no_queries
    )
    
    if success:
        print(f"\n✓ DocVQA dataset preparation completed!")
        print(f"Output: {args.output_path}")
    else:
        print("✗ DocVQA dataset preparation failed")
        sys.exit(1)


if __name__ == "__main__":
    main()
