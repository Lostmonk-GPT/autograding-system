#!/usr/bin/env python3
"""
Test the OCR pipeline with a real paper assignment
"""

import sys
import time
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path("src")))

def test_real_assignment(image_path: str):
    """Test OCR pipeline with a real assignment image."""
    
    print("📄 Testing OCR Pipeline with Real Paper Assignment")
    print("=" * 55)
    
    # Check if image exists
    if not Path(image_path).exists():
        print(f"❌ Image not found: {image_path}")
        print("\n📋 To test with real paper:")
        print("1. Take a clear photo of a handwritten/printed math assignment")
        print("2. Save it as: data/input/my_assignment.jpg")
        print("3. Run: python test_real_paper.py data/input/my_assignment.jpg")
        return False
    
    try:
        # Import pipeline components
        from modules.image_processor import ImageProcessor
        from modules.qr_extractor import QRExtractor  
        from modules.ocr.ocr_manager import OCRManager
        
        print(f"📂 Input image: {image_path}")
        print(f"📏 File size: {Path(image_path).stat().st_size / 1024:.1f} KB")
        
        # Initialize components
        image_processor = ImageProcessor()
        qr_extractor = QRExtractor()
        ocr_manager = OCRManager()
        
        print(f"🔧 Pipeline initialized with {len(ocr_manager.engines)} OCR engines")
        
        # Step 1: Image Processing
        print("\n🖼️  Step 1: Processing your scanned image...")
        start_time = time.time()
        
        result = image_processor.process_image(image_path)
        processing_time = time.time() - start_time
        
        if result:
            quality = result['quality_metrics']
            print(f"✅ Image processed in {processing_time:.2f}s")
            print(f"   📊 Quality score: {quality['overall_score']}/100")
            print(f"   📈 Sharpness: {quality['sharpness']:.1f}")
            print(f"   📈 Contrast: {quality['contrast']:.1f}")
            print(f"   📈 Brightness: {quality['brightness']:.1f}")
            print(f"   🎯 OCR suitable: {'✅ Yes' if quality['is_suitable_for_ocr'] else '❌ No'}")
            
            if not quality['is_suitable_for_ocr']:
                print("⚠️  Image quality may affect OCR accuracy")
                print("   💡 Try: better lighting, sharper focus, higher contrast")
        else:
            print("❌ Image processing failed")
            return False
        
        # Step 2: QR Code Detection
        print("\n🔲 Step 2: Looking for QR codes...")
        qr_start = time.time()
        
        qr_codes = qr_extractor.extract_from_file(image_path)
        qr_time = time.time() - qr_start
        
        print(f"✅ QR scan completed in {qr_time:.2f}s")
        if qr_codes:
            print(f"🎯 Found {len(qr_codes)} QR code(s)")
            for i, qr in enumerate(qr_codes):
                assignment_name = qr.get('assignment_name', 'Unknown')
                print(f"   📋 QR {i+1}: {assignment_name}")
        else:
            print("📭 No QR codes detected (this is normal for most papers)")
        
        # Step 3: OCR Text Extraction
        print("\n🔍 Step 3: Extracting text with OCR...")
        ocr_start = time.time()
        
        ocr_result = ocr_manager.extract_text(result['processed_image'], image_path)
        ocr_time = time.time() - ocr_start
        
        if ocr_result['success']:
            print(f"✅ OCR completed in {ocr_time:.2f}s")
            print(f"   🎯 Final confidence: {ocr_result['confidence']:.2f}")
            print(f"   📏 Text length: {len(ocr_result['final_text'])} characters")
            print(f"   🏆 Quality: {ocr_result['quality_metrics']['overall_quality']}")
            print(f"   🔧 Engines used: {', '.join(ocr_result['engines_used'])}")
            
            # Show extracted text
            text = ocr_result['final_text']
            print(f"\n📖 Extracted Text:")
            print("=" * 40)
            if len(text) > 500:
                print(f"{text[:500]}...")
                print(f"[... {len(text)-500} more characters]")
            else:
                print(text if text else "[No text extracted]")
            print("=" * 40)
            
            # Show individual engine results
            print(f"\n🔧 Individual Engine Performance:")
            for engine_result in ocr_result['engine_results']:
                status = "✅" if engine_result['success'] else "❌"
                print(f"   {status} {engine_result['engine_name']}: "
                      f"{engine_result['confidence']:.2f} confidence, "
                      f"{engine_result['character_count']} chars, "
                      f"{engine_result['processing_time']:.2f}s")
            
            # Analysis and recommendations
            print(f"\n📊 Analysis:")
            
            # Confidence analysis
            if ocr_result['confidence'] >= 0.8:
                print("🎯 Excellent confidence - text extraction very reliable")
            elif ocr_result['confidence'] >= 0.6:
                print("👍 Good confidence - text extraction mostly reliable")
            elif ocr_result['confidence'] >= 0.4:
                print("⚠️  Moderate confidence - may need manual review")
            else:
                print("❌ Low confidence - recommend manual review")
            
            # Character count analysis
            if len(text) > 100:
                print("📝 Substantial text extracted - good OCR performance")
            elif len(text) > 20:
                print("📝 Some text extracted - partial success")
            elif len(text) > 0:
                print("📝 Minimal text extracted - image may need improvement")
            else:
                print("📝 No text extracted - check image quality and content")
            
            # Recommendations
            print(f"\n💡 Recommendations:")
            if quality['overall_score'] < 70:
                print("   📸 Improve image quality: better lighting, focus, contrast")
            if ocr_result['confidence'] < 0.6:
                print("   ✍️  For handwriting: ensure clear, large writing")
                print("   🖨️  For printed text: ensure high contrast and resolution")
            if len(text) < 50:
                print("   🔍 Check if assignment contains mainly images/diagrams")
            
            print(f"\n⏱️  Total processing time: {processing_time + qr_time + ocr_time:.2f}s")
            
            return True
            
        else:
            print(f"❌ OCR failed: {ocr_result.get('error_message', 'Unknown error')}")
            return False
    
    except Exception as e:
        print(f"❌ Pipeline test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main function."""
    if len(sys.argv) < 2:
        print("📄 Test OCR Pipeline with Real Paper")
        print("=" * 35)
        print("\nUsage:")
        print("  python test_real_paper.py <image_path>")
        print("\nExample:")
        print("  python test_real_paper.py data/input/my_assignment.jpg")
        print("\n📋 Quick setup:")
        print("1. Take a photo of your math assignment")
        print("2. Save it to data/input/my_assignment.jpg")
        print("3. Run the command above")
        
        # List existing images in data/input if any
        input_dir = Path("data/input")
        if input_dir.exists():
            images = list(input_dir.glob("*.jpg")) + list(input_dir.glob("*.png"))
            if images:
                print(f"\n📂 Found {len(images)} image(s) in data/input:")
                for img in images[:5]:  # Show first 5
                    print(f"   📄 {img.name}")
                if images:
                    print(f"\n💡 Try: python test_real_paper.py data/input/{images[0].name}")
        
        sys.exit(1)
    
    image_path = sys.argv[1]
    success = test_real_assignment(image_path)
    
    if success:
        print("\n🎉 Real paper test completed successfully!")
        print("\n🚀 Your OCR pipeline is working with real assignments!")
    else:
        print("\n⚠️  Test completed with issues - see analysis above")
    
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()