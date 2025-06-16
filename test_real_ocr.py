#!/usr/bin/env python3
"""
Test only real OCR engines (no mock)
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path("src")))

def test_only_real_ocr(image_path: str):
    """Test with only real OCR engines."""
    
    print("🔍 Testing REAL OCR Only (No Mock)")
    print("=" * 40)
    
    try:
        from modules.ocr.tesseract_ocr import TesseractOCR
        from modules.image_processor import ImageProcessor
        import time
        
        # Process image
        processor = ImageProcessor()
        result = processor.process_image(image_path)
        
        if not result:
            print("❌ Image processing failed")
            return False
        
        print(f"✅ Image processed (quality: {result['quality_metrics']['overall_score']}/100)")
        
        # Test only Tesseract (real OCR)
        print("\n🔍 Testing Tesseract (Real OCR)...")
        
        tesseract = TesseractOCR()
        start_time = time.time()
        
        text, confidence = tesseract.extract_text(result['processed_image'])
        processing_time = time.time() - start_time
        
        print(f"📝 Tesseract Results:")
        print(f"   Text: '{text}'")
        print(f"   Confidence: {confidence:.2f}")
        print(f"   Time: {processing_time:.2f}s")
        print(f"   Characters: {len(text)}")
        
        if len(text) > 0:
            print("\n✅ REAL OCR is working!")
            print(f"📖 Extracted Text:\n{text}")
            return True
        else:
            print("\n❌ REAL OCR extracted no text")
            print("💡 This means:")
            print("   - Image quality may be too poor for OCR")
            print("   - Text may be too small/blurry")
            print("   - Tesseract configuration needs adjustment")
            return False
            
    except Exception as e:
        print(f"❌ Real OCR test failed: {e}")
        return False

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python test_real_ocr.py <image_path>")
        sys.exit(1)
    
    image_path = sys.argv[1]
    success = test_only_real_ocr(image_path)
    
    if success:
        print("\n🎉 Real OCR is working with your image!")
    else:
        print("\n⚠️  Real OCR needs improvement - see suggestions above")
