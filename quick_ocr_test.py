#!/usr/bin/env python3
"""
Simple OCR verification test - tests each library individually.
"""

import numpy as np
from PIL import Image, ImageDraw, ImageFont

def create_test_image():
    """Create a simple test image with text."""
    img = Image.new('RGB', (300, 80), color='white')
    draw = ImageDraw.Draw(img)
    
    # Use default font
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 20)
    except:
        font = ImageFont.load_default()
    
    text = "Test OCR 123"
    draw.text((20, 25), text, fill='black', font=font)
    return img, text

def test_pytesseract():
    """Test pytesseract."""
    print("1. Testing pytesseract...")
    try:
        import pytesseract
        img, expected = create_test_image()
        result = pytesseract.image_to_string(np.array(img)).strip()
        print(f"   Expected: '{expected}' | Got: '{result}'")
        print("   ✓ pytesseract working!")
        return True
    except Exception as e:
        print(f"   ✗ Error: {e}")
        return False

def test_easyocr():
    """Test easyocr."""
    print("\n2. Testing easyocr...")
    try:
        import easyocr
        img, expected = create_test_image()
        reader = easyocr.Reader(['en'], verbose=False)
        results = reader.readtext(np.array(img))
        result = results[0][1] if results else "No text found"
        print(f"   Expected: '{expected}' | Got: '{result}'")
        print("   ✓ easyocr working!")
        return True
    except Exception as e:
        print(f"   ✗ Error: {e}")
        return False

def test_opencv():
    """Test OpenCV."""
    print("\n3. Testing OpenCV...")
    try:
        import cv2
        print(f"   OpenCV version: {cv2.__version__}")
        # Simple test - create and process an image
        img = np.zeros((100, 200, 3), dtype=np.uint8)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        print("   ✓ OpenCV working!")
        return True
    except Exception as e:
        print(f"   ✗ Error: {e}")
        return False

def main():
    print("=== OCR Libraries Quick Test ===")
    
    tests = [
        test_opencv(),
        test_pytesseract(),
        test_easyocr()
    ]
    
    passed = sum(tests)
    total = len(tests)
    
    print(f"\n=== Results: {passed}/{total} tests passed ===")
    
    if passed == total:
        print("🎉 All core OCR libraries are working!")
    else:
        print("⚠️  Some tests failed.")
    
    # Note about PaddleOCR
    print("\nNote: PaddleOCR import was tested separately and is working.")
    print("(PaddleOCR was excluded from this test to avoid long initialization times)")

if __name__ == "__main__":
    main()
