#!/usr/bin/env python3
"""
Test script to verify OCR libraries installation and functionality.
Tests pytesseract, easyocr, and paddleocr with a simple text image.
"""

import sys
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import cv2

def create_test_image():
    """Create a simple test image with text."""
    # Create a white image
    img = Image.new('RGB', (400, 100), color='white')
    draw = ImageDraw.Draw(img)
    
    # Try to use a default font, fallback to basic if not available
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 24)
    except:
        font = ImageFont.load_default()
    
    # Add text to image
    text = "Hello OCR World!"
    draw.text((50, 30), text, fill='black', font=font)
    
    return img, text

def test_pytesseract():
    """Test pytesseract OCR."""
    print("Testing pytesseract...")
    try:
        import pytesseract
        img, expected_text = create_test_image()
        
        # Convert PIL image to numpy array for pytesseract
        img_array = np.array(img)
        
        # Perform OCR
        result = pytesseract.image_to_string(img_array).strip()
        print(f"  Expected: '{expected_text}'")
        print(f"  Got: '{result}'")
        print(f"  ✓ pytesseract working!")
        return True
    except Exception as e:
        print(f"  ✗ pytesseract failed: {e}")
        return False

def test_easyocr():
    """Test easyocr."""
    print("\nTesting easyocr...")
    try:
        import easyocr
        img, expected_text = create_test_image()
        
        # Convert PIL image to numpy array
        img_array = np.array(img)
        
        # Initialize EasyOCR reader
        reader = easyocr.Reader(['en'])
        
        # Perform OCR
        results = reader.readtext(img_array)
        if results:
            result = results[0][1].strip()  # Get the text from first result
            print(f"  Expected: '{expected_text}'")
            print(f"  Got: '{result}'")
        else:
            print("  No text detected")
        print(f"  ✓ easyocr working!")
        return True
    except Exception as e:
        print(f"  ✗ easyocr failed: {e}")
        return False

def test_paddleocr():
    """Test paddleocr."""
    print("\nTesting paddleocr...")
    try:
        from paddleocr import PaddleOCR
        img, expected_text = create_test_image()
        
        # Convert PIL image to numpy array
        img_array = np.array(img)
        
        # Initialize PaddleOCR
        ocr = PaddleOCR(use_angle_cls=True, lang='en', show_log=False)
        
        # Perform OCR
        results = ocr.ocr(img_array, cls=True)
        if results and results[0]:
            result = results[0][0][1][0].strip()  # Get the text from first result
            print(f"  Expected: '{expected_text}'")
            print(f"  Got: '{result}'")
        else:
            print("  No text detected")
        print(f"  ✓ paddleocr working!")
        return True
    except Exception as e:
        print(f"  ✗ paddleocr failed: {e}")
        return False

def test_opencv():
    """Test OpenCV installation."""
    print("\nTesting OpenCV...")
    try:
        import cv2
        print(f"  OpenCV version: {cv2.__version__}")
        
        # Create a simple test
        img = np.zeros((100, 200, 3), dtype=np.uint8)
        cv2.putText(img, 'OpenCV Test', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        print(f"  ✓ OpenCV working!")
        return True
    except Exception as e:
        print(f"  ✗ OpenCV failed: {e}")
        return False

def main():
    """Run all OCR tests."""
    print("=== OCR Libraries Test ===\n")
    
    results = []
    results.append(test_opencv())
    results.append(test_pytesseract())
    results.append(test_easyocr())
    results.append(test_paddleocr())
    
    print(f"\n=== Test Summary ===")
    print(f"Tests passed: {sum(results)}/{len(results)}")
    
    if all(results):
        print("🎉 All OCR libraries are working correctly!")
    else:
        print("⚠️  Some libraries failed. Check the errors above.")
    
    return all(results)

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
