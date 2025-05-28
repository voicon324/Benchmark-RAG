"""
Utility modules for NewAIBench.

This package provides utility functions and classes for various tasks
including OCR processing, data validation, and format conversion.
"""

from .ocr_processor import (
    OCRConfig,
    OCRProcessor,
    TesseractEngine,
    EasyOCREngine,
    PaddleOCREngine,
    create_ocr_processor
)

__all__ = [
    "OCRConfig",
    "OCRProcessor", 
    "TesseractEngine",
    "EasyOCREngine",
    "PaddleOCREngine",
    "create_ocr_processor"
]
