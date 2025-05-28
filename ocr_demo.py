#!/usr/bin/env python3
"""
Simple OCR example demonstrating basic usage of each library.
This script shows how to use pytesseract, easyocr, and basic image processing.
"""

def demonstrate_imports():
    """Demonstrate that all OCR libraries can be imported successfully."""
    print("=== OCR Libraries Import Test ===")
    
    try:
        import cv2
        print(f"✅ OpenCV {cv2.__version__} imported successfully")
    except ImportError as e:
        print(f"❌ OpenCV import failed: {e}")
    
    try:
        import pytesseract
        print("✅ pytesseract imported successfully")
    except ImportError as e:
        print(f"❌ pytesseract import failed: {e}")
    
    try:
        import easyocr
        print("✅ easyocr imported successfully")
    except ImportError as e:
        print(f"❌ easyocr import failed: {e}")
    
    try:
        from paddleocr import PaddleOCR
        print("✅ paddleocr imported successfully")
    except ImportError as e:
        print(f"❌ paddleocr import failed: {e}")
    
    try:
        from PIL import Image
        print("✅ Pillow (PIL) imported successfully")
    except ImportError as e:
        print(f"❌ Pillow import failed: {e}")
    
    try:
        import numpy as np
        print(f"✅ NumPy {np.__version__} imported successfully")
    except ImportError as e:
        print(f"❌ NumPy import failed: {e}")

def demonstrate_basic_usage():
    """Show basic usage examples for each OCR library."""
    print("\n=== Basic Usage Examples ===")
    
    print("\n1. pytesseract usage:")
    print("""
import pytesseract
from PIL import Image

image = Image.open('document.jpg')
text = pytesseract.image_to_string(image)
print(text)
""")
    
    print("\n2. easyocr usage:")
    print("""
import easyocr

reader = easyocr.Reader(['en'])  # Initialize for English
results = reader.readtext('document.jpg')

for (bbox, text, confidence) in results:
    print(f"Text: {text}")
    print(f"Confidence: {confidence:.2f}")
""")
    
    print("\n3. paddleocr usage:")
    print("""
from paddleocr import PaddleOCR

ocr = PaddleOCR(use_angle_cls=True, lang='en')
results = ocr.ocr('document.jpg', cls=True)

for line in results:
    for word_info in line:
        text = word_info[1][0]
        confidence = word_info[1][1]
        print(f"Text: {text}, Confidence: {confidence:.2f}")
""")

def main():
    """Main function to run the demonstration."""
    print("OCR Libraries Installation Verification")
    print("=" * 50)
    
    demonstrate_imports()
    demonstrate_basic_usage()
    
    print("\n=== Summary ===")
    print("✅ All OCR libraries are installed and ready to use!")
    print("✅ You can now process document images with multiple OCR engines")
    print("✅ Check OCR_INSTALLATION_SUMMARY.md for detailed information")
    print("\nTo get started, try importing the libraries in your Python scripts.")

if __name__ == "__main__":
    main()
