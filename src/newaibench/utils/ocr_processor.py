"""
OCR Utility Module for NewAIBench Document Image Processing.

This module provides OCR capabilities for extracting text from document images,
supporting multiple OCR engines and languages with batch processing capabilities.
"""

import logging
import time
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass
import json

# OCR Engine Imports
try:
    import pytesseract
    from PIL import Image
    TESSERACT_AVAILABLE = True
except ImportError:
    TESSERACT_AVAILABLE = False
    warnings.warn("Tesseract (pytesseract + PIL) not available")

try:
    import easyocr
    EASYOCR_AVAILABLE = True
except ImportError:
    EASYOCR_AVAILABLE = False
    warnings.warn("EasyOCR not available")

try:
    import paddle
    from paddleocr import PaddleOCR
    PADDLEOCR_AVAILABLE = True
except ImportError:
    PADDLEOCR_AVAILABLE = False
    warnings.warn("PaddleOCR not available")

logger = logging.getLogger(__name__)


@dataclass
class OCRConfig:
    """Configuration for OCR processing."""
    
    # Engine selection
    engine: str = "auto"  # "tesseract", "easyocr", "paddleocr", "auto"
    
    # Language settings
    languages: List[str] = None  # ["en", "vi"] etc.
    
    # Processing options
    confidence_threshold: float = 0.5
    preprocessing: bool = True
    batch_size: int = 1
    max_workers: int = 4
    
    # Output options
    output_format: str = "text"  # "text", "detailed", "boxes"
    save_intermediate: bool = False
    
    # Engine-specific settings
    tesseract_config: Dict[str, Any] = None
    easyocr_config: Dict[str, Any] = None
    paddleocr_config: Dict[str, Any] = None
    
    def __post_init__(self):
        """Set default values and validate configuration."""
        if self.languages is None:
            self.languages = ["en"]
        
        if self.tesseract_config is None:
            self.tesseract_config = {
                "config": "--psm 3",  # Automatic page segmentation
                "timeout": 30
            }
        
        if self.easyocr_config is None:
            self.easyocr_config = {
                "gpu": False,
                "detail": 1
            }
        
        if self.paddleocr_config is None:
            self.paddleocr_config = {
                "use_angle_cls": True,
                "lang": "en" if "en" in self.languages else self.languages[0],
                "gpu": False,
                "show_log": False
            }


class OCREngine:
    """Base class for OCR engines."""
    
    def __init__(self, config: OCRConfig):
        self.config = config
        self.is_initialized = False
    
    def initialize(self):
        """Initialize the OCR engine."""
        pass
    
    def extract_text(self, image_path: Union[str, Path]) -> Dict[str, Any]:
        """Extract text from image."""
        raise NotImplementedError
    
    def cleanup(self):
        """Cleanup resources."""
        pass


class TesseractEngine(OCREngine):
    """Tesseract OCR engine implementation."""
    
    def initialize(self):
        """Initialize Tesseract OCR."""
        if not TESSERACT_AVAILABLE:
            raise RuntimeError("Tesseract not available. Install with: pip install pytesseract pillow")
        
        try:
            # Test if tesseract is properly installed
            pytesseract.get_tesseract_version()
            self.is_initialized = True
            logger.info("Tesseract OCR initialized successfully")
        except Exception as e:
            raise RuntimeError(f"Tesseract initialization failed: {e}")
    
    def extract_text(self, image_path: Union[str, Path]) -> Dict[str, Any]:
        """Extract text using Tesseract."""
        if not self.is_initialized:
            self.initialize()
        
        try:
            image = Image.open(image_path)
            
            # Prepare language string
            lang = "+".join(self.config.languages)
            
            # Extract text
            text = pytesseract.image_to_string(
                image,
                lang=lang,
                config=self.config.tesseract_config.get("config", "--psm 3"),
                timeout=self.config.tesseract_config.get("timeout", 30)
            )
            
            # Get confidence if detailed output requested
            confidence = None
            if self.config.output_format == "detailed":
                try:
                    data = pytesseract.image_to_data(image, lang=lang, output_type=pytesseract.Output.DICT)
                    confidences = [int(conf) for conf in data['conf'] if int(conf) > 0]
                    confidence = sum(confidences) / len(confidences) if confidences else 0
                except Exception:
                    confidence = 0
            
            return {
                "text": text.strip(),
                "confidence": confidence,
                "engine": "tesseract",
                "language": lang
            }
            
        except Exception as e:
            logger.error(f"Tesseract OCR failed for {image_path}: {e}")
            return {
                "text": "",
                "confidence": 0,
                "engine": "tesseract",
                "error": str(e)
            }


class EasyOCREngine(OCREngine):
    """EasyOCR engine implementation."""
    
    def initialize(self):
        """Initialize EasyOCR."""
        if not EASYOCR_AVAILABLE:
            raise RuntimeError("EasyOCR not available. Install with: pip install easyocr")
        
        try:
            self.reader = easyocr.Reader(
                self.config.languages,
                gpu=self.config.easyocr_config.get("gpu", False)
            )
            self.is_initialized = True
            logger.info(f"EasyOCR initialized for languages: {self.config.languages}")
        except Exception as e:
            raise RuntimeError(f"EasyOCR initialization failed: {e}")
    
    def extract_text(self, image_path: Union[str, Path]) -> Dict[str, Any]:
        """Extract text using EasyOCR."""
        if not self.is_initialized:
            self.initialize()
        
        try:
            results = self.reader.readtext(
                str(image_path),
                detail=self.config.easyocr_config.get("detail", 1)
            )
            
            # Process results
            text_parts = []
            confidences = []
            
            for result in results:
                if len(result) >= 2:
                    text_part = result[1]
                    confidence = result[2] if len(result) > 2 else 1.0
                    
                    if confidence >= self.config.confidence_threshold:
                        text_parts.append(text_part)
                        confidences.append(confidence)
            
            full_text = " ".join(text_parts)
            avg_confidence = sum(confidences) / len(confidences) if confidences else 0
            
            return {
                "text": full_text.strip(),
                "confidence": avg_confidence,
                "engine": "easyocr",
                "language": "+".join(self.config.languages),
                "details": results if self.config.output_format == "detailed" else None
            }
            
        except Exception as e:
            logger.error(f"EasyOCR failed for {image_path}: {e}")
            return {
                "text": "",
                "confidence": 0,
                "engine": "easyocr",
                "error": str(e)
            }


class PaddleOCREngine(OCREngine):
    """PaddleOCR engine implementation."""
    
    def initialize(self):
        """Initialize PaddleOCR."""
        if not PADDLEOCR_AVAILABLE:
            raise RuntimeError("PaddleOCR not available. Install with: pip install paddlepaddle paddleocr")
        
        try:
            # Set language for PaddleOCR
            lang = self.config.paddleocr_config.get("lang", "en")
            if "vi" in self.config.languages:
                lang = "vi"
            elif "zh" in self.config.languages:
                lang = "ch"
            
            self.ocr = PaddleOCR(
                use_angle_cls=self.config.paddleocr_config.get("use_angle_cls", True),
                lang=lang,
                use_gpu=self.config.paddleocr_config.get("gpu", False),
                show_log=self.config.paddleocr_config.get("show_log", False)
            )
            self.is_initialized = True
            logger.info(f"PaddleOCR initialized for language: {lang}")
        except Exception as e:
            raise RuntimeError(f"PaddleOCR initialization failed: {e}")
    
    def extract_text(self, image_path: Union[str, Path]) -> Dict[str, Any]:
        """Extract text using PaddleOCR."""
        if not self.is_initialized:
            self.initialize()
        
        try:
            results = self.ocr.ocr(str(image_path), cls=True)
            
            # Process results
            text_parts = []
            confidences = []
            
            if results and results[0]:
                for line in results[0]:
                    if line and len(line) >= 2:
                        text_part = line[1][0]  # Text content
                        confidence = line[1][1]  # Confidence score
                        
                        if confidence >= self.config.confidence_threshold:
                            text_parts.append(text_part)
                            confidences.append(confidence)
            
            full_text = " ".join(text_parts)
            avg_confidence = sum(confidences) / len(confidences) if confidences else 0
            
            return {
                "text": full_text.strip(),
                "confidence": avg_confidence,
                "engine": "paddleocr",
                "language": self.config.paddleocr_config.get("lang", "en"),
                "details": results if self.config.output_format == "detailed" else None
            }
            
        except Exception as e:
            logger.error(f"PaddleOCR failed for {image_path}: {e}")
            return {
                "text": "",
                "confidence": 0,
                "engine": "paddleocr",
                "error": str(e)
            }


class OCRProcessor:
    """Main OCR processor that manages different engines and batch processing."""
    
    def __init__(self, config: OCRConfig):
        """Initialize OCR processor with configuration."""
        self.config = config
        self.engine = None
        self.stats = {
            "processed": 0,
            "successful": 0,
            "failed": 0,
            "total_time": 0.0,
            "avg_confidence": 0.0
        }
    
    def _select_engine(self) -> OCREngine:
        """Select and initialize the appropriate OCR engine."""
        if self.config.engine == "auto":
            # Auto-select based on availability and language
            if "vi" in self.config.languages and PADDLEOCR_AVAILABLE:
                return PaddleOCREngine(self.config)
            elif EASYOCR_AVAILABLE:
                return EasyOCREngine(self.config)
            elif TESSERACT_AVAILABLE:
                return TesseractEngine(self.config)
            else:
                raise RuntimeError("No OCR engines available")
        
        elif self.config.engine == "tesseract":
            return TesseractEngine(self.config)
        elif self.config.engine == "easyocr":
            return EasyOCREngine(self.config)
        elif self.config.engine == "paddleocr":
            return PaddleOCREngine(self.config)
        else:
            raise ValueError(f"Unknown OCR engine: {self.config.engine}")
    
    def initialize(self):
        """Initialize the OCR processor."""
        self.engine = self._select_engine()
        self.engine.initialize()
        logger.info(f"OCR processor initialized with {self.engine.__class__.__name__}")
    
    def process_image(self, image_path: Union[str, Path]) -> Dict[str, Any]:
        """Process a single image."""
        if not self.engine:
            self.initialize()
        
        start_time = time.time()
        result = self.engine.extract_text(image_path)
        processing_time = time.time() - start_time
        
        # Update statistics
        self.stats["processed"] += 1
        self.stats["total_time"] += processing_time
        
        if result["text"]:
            self.stats["successful"] += 1
            if "confidence" in result and result["confidence"]:
                self.stats["avg_confidence"] = (
                    (self.stats["avg_confidence"] * (self.stats["successful"] - 1) + 
                     result["confidence"]) / self.stats["successful"]
                )
        else:
            self.stats["failed"] += 1
        
        result["processing_time"] = processing_time
        return result
    
    def process_batch(self, image_paths: List[Union[str, Path]], 
                     output_path: Optional[Union[str, Path]] = None,
                     show_progress: bool = True) -> Dict[str, Dict[str, Any]]:
        """Process multiple images in batch."""
        if not self.engine:
            self.initialize()
        
        results = {}
        total_images = len(image_paths)
        
        if self.config.max_workers > 1:
            # Parallel processing
            with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
                # Submit all tasks
                future_to_path = {
                    executor.submit(self.process_image, path): path 
                    for path in image_paths
                }
                
                # Collect results
                for i, future in enumerate(as_completed(future_to_path)):
                    path = future_to_path[future]
                    try:
                        result = future.result()
                        results[str(path)] = result
                        
                        if show_progress and (i + 1) % 10 == 0:
                            logger.info(f"Processed {i + 1}/{total_images} images")
                    except Exception as e:
                        logger.error(f"Error processing {path}: {e}")
                        results[str(path)] = {
                            "text": "",
                            "confidence": 0,
                            "error": str(e)
                        }
        else:
            # Sequential processing
            for i, path in enumerate(image_paths):
                try:
                    result = self.process_image(path)
                    results[str(path)] = result
                    
                    if show_progress and (i + 1) % 10 == 0:
                        logger.info(f"Processed {i + 1}/{total_images} images")
                except Exception as e:
                    logger.error(f"Error processing {path}: {e}")
                    results[str(path)] = {
                        "text": "",
                        "confidence": 0,
                        "error": str(e)
                    }
        
        # Save results if output path specified
        if output_path:
            self.save_results(results, output_path)
        
        logger.info(f"Batch processing completed: {len(results)} images processed")
        return results
    
    def save_results(self, results: Dict[str, Dict[str, Any]], 
                    output_path: Union[str, Path]):
        """Save OCR results to file."""
        output_path = Path(output_path)
        
        if output_path.suffix == ".json":
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
        elif output_path.suffix == ".jsonl":
            with open(output_path, 'w', encoding='utf-8') as f:
                for path, result in results.items():
                    entry = {"image_path": path, **result}
                    f.write(json.dumps(entry, ensure_ascii=False) + '\n')
        elif output_path.suffix == ".txt":
            with open(output_path, 'w', encoding='utf-8') as f:
                for path, result in results.items():
                    f.write(f"=== {path} ===\n")
                    f.write(f"{result.get('text', '')}\n\n")
        else:
            raise ValueError(f"Unsupported output format: {output_path.suffix}")
        
        logger.info(f"Results saved to {output_path}")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get processing statistics."""
        stats = self.stats.copy()
        if stats["processed"] > 0:
            stats["success_rate"] = stats["successful"] / stats["processed"]
            stats["avg_processing_time"] = stats["total_time"] / stats["processed"]
        else:
            stats["success_rate"] = 0
            stats["avg_processing_time"] = 0
        
        return stats
    
    def cleanup(self):
        """Cleanup resources."""
        if self.engine:
            self.engine.cleanup()


def create_ocr_processor(engine: str = "auto",
                        languages: List[str] = None,
                        confidence_threshold: float = 0.5,
                        **kwargs) -> OCRProcessor:
    """Create OCR processor with simplified configuration."""
    if languages is None:
        languages = ["en"]
    
    config = OCRConfig(
        engine=engine,
        languages=languages,
        confidence_threshold=confidence_threshold,
        **kwargs
    )
    
    return OCRProcessor(config)


# CLI functionality
def main():
    """Command line interface for OCR processing."""
    import argparse
    
    parser = argparse.ArgumentParser(description="NewAIBench OCR Utility")
    parser.add_argument("input_path", help="Input image file or directory")
    parser.add_argument("-o", "--output", help="Output file path")
    parser.add_argument("-e", "--engine", choices=["auto", "tesseract", "easyocr", "paddleocr"], 
                       default="auto", help="OCR engine to use")
    parser.add_argument("-l", "--languages", nargs="+", default=["en"], 
                       help="Languages to use for OCR")
    parser.add_argument("-c", "--confidence", type=float, default=0.5,
                       help="Confidence threshold")
    parser.add_argument("-w", "--workers", type=int, default=4,
                       help="Number of worker threads")
    parser.add_argument("-v", "--verbose", action="store_true",
                       help="Verbose output")
    
    args = parser.parse_args()
    
    # Setup logging
    level = logging.INFO if args.verbose else logging.WARNING
    logging.basicConfig(level=level)
    
    # Create processor
    processor = create_ocr_processor(
        engine=args.engine,
        languages=args.languages,
        confidence_threshold=args.confidence,
        max_workers=args.workers
    )
    
    input_path = Path(args.input_path)
    
    if input_path.is_file():
        # Process single image
        result = processor.process_image(input_path)
        print(f"Text extracted: {result['text']}")
        if "confidence" in result:
            print(f"Confidence: {result['confidence']:.2f}")
        
        if args.output:
            processor.save_results({str(input_path): result}, args.output)
    
    elif input_path.is_dir():
        # Process directory
        image_extensions = {".jpg", ".jpeg", ".png", ".tiff", ".tif", ".bmp", ".pdf"}
        image_paths = [
            p for p in input_path.rglob("*") 
            if p.suffix.lower() in image_extensions
        ]
        
        print(f"Found {len(image_paths)} images to process")
        
        results = processor.process_batch(image_paths, args.output, show_progress=True)
        
        # Print statistics
        stats = processor.get_statistics()
        print(f"\nProcessing Statistics:")
        print(f"  Processed: {stats['processed']}")
        print(f"  Successful: {stats['successful']}")
        print(f"  Failed: {stats['failed']}")
        print(f"  Success Rate: {stats['success_rate']:.2%}")
        print(f"  Average Confidence: {stats['avg_confidence']:.2f}")
        print(f"  Average Processing Time: {stats['avg_processing_time']:.2f}s")
    
    else:
        print(f"Error: {input_path} is not a valid file or directory")


if __name__ == "__main__":
    main()
