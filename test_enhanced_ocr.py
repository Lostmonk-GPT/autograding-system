#!/usr/bin/env python3
"""
AutoGrading System - Complete OCR Testing Script
Test all new OCR enhancements and compare performance.
"""

import sys
import time
import cv2
import numpy as np
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path("src")))


def test_real_pix2text():
    """Test real Pix2Text installation and functionality."""
    print("🧮 Testing Real Pix2Text Engine")
    print("=" * 40)

    try:
        # Try to install Pix2Text if not available
        try:
            import pix2text
            print("✅ Pix2Text already installed")
        except ImportError:
            print("⚠️  Pix2Text not installed. Install with:")
            print("   pip install pix2text")
            print("   This may take a few minutes for the first install...")
            return False

        # Test our Pix2Text engine
        from modules.ocr.pix2text_ocr import Pix2TextEngine

        engine = Pix2TextEngine()
        capabilities = engine.get_capabilities()

        print(f"✅ Pix2Text engine initialized")
        print(
            f"   Real model: {'✅' if capabilities['real_model_available'] else '❌'}")
        print(f"   Device: {capabilities['device']}")
        print(f"   Model: {capabilities['model_type']}")

        # Test with mathematical content
        test_image = np.ones((150, 400), dtype=np.uint8) * 255
        cv2.putText(test_image, "x^2 + 5x - 6 = 0", (20, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, 0, 2)
        cv2.putText(test_image, "x = (-5 ± √49)/2", (20, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, 0, 2)

        start_time = time.time()
        text, confidence = engine.extract_text(test_image)
        processing_time = time.time() - start_time

        print(f"\n📝 Test Results:")
        print(f"   Text: '{text}'")
        print(f"   Confidence: {confidence:.2f}")
        print(f"   Time: {processing_time:.2f}s")
        print(f"   Success: {'✅' if len(text) > 0 else '❌'}")

        return len(text) > 0

    except Exception as e:
        print(f"❌ Pix2Text test failed: {e}")
        return False


def test_fixed_easyocr():
    """Test fixed EasyOCR implementation."""
    print("\n✍️  Testing Fixed EasyOCR Engine")
    print("=" * 40)

    try:
        # Check if EasyOCR is installed
        try:
            import easyocr
            print("✅ EasyOCR library available")
        except ImportError:
            print("⚠️  EasyOCR not installed. Install with:")
            print("   pip install easyocr")
            return False

        # Test our fixed EasyOCR engine
        from modules.ocr.easy_ocr import EasyOCREngine

        engine = EasyOCREngine()
        capabilities = engine.get_capabilities()

        print(f"✅ EasyOCR engine initialized")
        print(
            f"   Real model: {'✅' if capabilities['real_model_available'] else '❌'}")
        print(f"   GPU enabled: {'✅' if capabilities['gpu_enabled'] else '❌'}")
        print(f"   Languages: {', '.join(capabilities['languages'])}")

        # Test with handwriting-style content
        test_image = np.ones((120, 300), dtype=np.uint8) * 255

        # Simulate handwritten style
        font = cv2.FONT_HERSHEY_SCRIPT_SIMPLEX
        cv2.putText(test_image, "x = 4", (20, 40), font, 1.0, 0, 2)
        cv2.putText(test_image, "y = 2x + 3", (20, 80), font, 0.8, 0, 2)

        # Add slight noise to simulate handwriting
        noise = np.random.randint(-10, 10, test_image.shape, dtype=np.int16)
        test_image = np.clip(test_image.astype(
            np.int16) + noise, 0, 255).astype(np.uint8)

        start_time = time.time()
        text, confidence = engine.extract_text(test_image)
        processing_time = time.time() - start_time

        print(f"\n📝 Test Results:")
        print(f"   Text: '{text}'")
        print(f"   Confidence: {confidence:.2f}")
        print(f"   Time: {processing_time:.2f}s")
        print(f"   Success: {'✅' if len(text) > 0 else '❌'}")

        return len(text) > 0

    except Exception as e:
        print(f"❌ EasyOCR test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_advanced_preprocessing():
    """Test advanced image preprocessing."""
    print("\n🖼️  Testing Advanced Image Preprocessing")
    print("=" * 45)

    try:
        from modules.advanced_image_processor import AdvancedImageProcessor

        processor = AdvancedImageProcessor()
        print("✅ Advanced image processor initialized")

        # Create problematic image (simulating your real test case)
        print("\n🧪 Creating poor quality test image...")
        poor_image = np.ones((400, 600), dtype=np.uint8) * \
            250  # Too bright (like your 253.8)

        # Add low contrast mathematical content
        cv2.putText(poor_image, "2x + 3 = 7", (50, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, 30, 2)  # Low contrast
        cv2.putText(poor_image, "x = (7-3)/2", (50, 150),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, 30, 2)
        cv2.putText(poor_image, "x = 2", (50, 200),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, 30, 2)

        # Add noise
        noise = np.random.randint(0, 20, poor_image.shape, dtype=np.uint8)
        poor_image = cv2.subtract(poor_image, noise)

        print("✅ Created test image with quality issues:")
        print(f"   Brightness: {np.mean(poor_image):.1f} (too high)")
        print(f"   Contrast: {np.std(poor_image):.1f} (too low)")

        # Process with advanced pipeline
        print("\n🔄 Processing with advanced pipeline...")
        start_time = time.time()
        result = processor.process_for_ocr(poor_image)
        processing_time = time.time() - start_time

        if result['success']:
            original_quality = result['original_quality']['overall_score']
            final_quality = result['final_quality']['overall_score']
            improvement = result['improvement_score']

            print(
                f"✅ Advanced preprocessing completed in {processing_time:.2f}s")
            print(f"   Original quality: {original_quality:.1f}/100")
            print(f"   Final quality: {final_quality:.1f}/100")
            print(f"   Improvement: {improvement:+.1f} points")
            print(f"   Steps applied: {', '.join(result['processing_steps'])}")
            print(
                f"   OCR ready: {'✅' if result['recommended_for_ocr'] else '❌'}")

            return result['recommended_for_ocr']
        else:
            print(f"❌ Preprocessing failed: {result.get('error_message')}")
            return False

    except Exception as e:
        print(f"❌ Advanced preprocessing test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_complete_pipeline_with_enhancements():
    """Test complete OCR pipeline with all enhancements."""
    print("\n🎯 Testing Complete Enhanced OCR Pipeline")
    print("=" * 50)

    try:
        # Import all components
        from modules.advanced_image_processor import AdvancedImageProcessor
        from modules.ocr.ocr_manager import OCRManager
        from modules.qr_extractor import QRExtractor

        # Initialize components
        preprocessor = AdvancedImageProcessor()
        ocr_manager = OCRManager()
        qr_extractor = QRExtractor()

        print("✅ All pipeline components initialized")
        print(f"   Available OCR engines: {list(ocr_manager.engines.keys())}")

        # Create comprehensive test assignment
        print("\n📝 Creating comprehensive test assignment...")

        # Create assignment with QR code
        assignment_image = create_test_assignment_with_qr()
        if assignment_image is None:
            print("⚠️  Could not create QR assignment, using simple test")
            assignment_image = create_simple_test_assignment()

        print("✅ Test assignment created")

        # Step 1: Advanced Preprocessing
        print("\n🖼️  Step 1: Advanced Image Preprocessing...")
        preprocess_start = time.time()

        preprocess_result = preprocessor.process_for_ocr(assignment_image)
        preprocess_time = time.time() - preprocess_start

        if preprocess_result['success']:
            processed_image = preprocess_result['processed_image']
            quality_improvement = preprocess_result['improvement_score']

            print(f"✅ Preprocessing completed in {preprocess_time:.2f}s")
            print(f"   Quality improvement: {quality_improvement:+.1f} points")
            print(
                f"   Final quality: {preprocess_result['final_quality']['overall_score']:.1f}/100")
        else:
            print("⚠️  Preprocessing failed, using original image")
            processed_image = assignment_image

        # Step 2: QR Code Extraction
        print("\n🔲 Step 2: QR Code Extraction...")
        qr_start = time.time()

        qr_codes = qr_extractor.extract_qr_codes(processed_image)
        qr_time = time.time() - qr_start

        print(f"✅ QR extraction completed in {qr_time:.2f}s")
        print(f"   QR codes found: {len(qr_codes)}")

        # Step 3: Enhanced OCR Processing
        print("\n🔍 Step 3: Enhanced OCR Processing...")
        ocr_start = time.time()

        ocr_result = ocr_manager.extract_text(
            processed_image, "enhanced_test.jpg")
        ocr_time = time.time() - ocr_start

        if ocr_result['success']:
            print(f"✅ OCR completed in {ocr_time:.2f}s")
            print(f"   Engines used: {', '.join(ocr_result['engines_used'])}")
            print(f"   Final confidence: {ocr_result['confidence']:.2f}")
            print(
                f"   Text length: {len(ocr_result['final_text'])} characters")
            print(
                f"   Quality: {ocr_result['quality_metrics']['overall_quality']}")

            # Show extracted text preview
            text_preview = ocr_result['final_text'][:150]
            if len(ocr_result['final_text']) > 150:
                text_preview += "..."
            print(f"\n📄 Extracted Text Preview:")
            print(f"   \"{text_preview}\"")

            # Show individual engine performance
            print(f"\n🔧 Engine Performance:")
            for engine_result in ocr_result['engine_results']:
                status = "✅" if engine_result['success'] else "❌"
                print(f"   {status} {engine_result['engine_name']}: "
                      f"{engine_result['confidence']:.2f} confidence, "
                      f"{engine_result['character_count']} chars, "
                      f"{engine_result['processing_time']:.2f}s")

        else:
            print(f"❌ OCR failed: {ocr_result.get('error_message')}")
            ocr_time = 0

        # Calculate total performance
        total_time = preprocess_time + qr_time + ocr_time

        # Performance Analysis
        print(f"\n📊 Enhanced Pipeline Performance:")
        print("=" * 40)
        print(f"📏 Timing Breakdown:")
        print(f"   Preprocessing: {preprocess_time:.2f}s")
        print(f"   QR Extraction: {qr_time:.2f}s")
        print(f"   OCR Processing: {ocr_time:.2f}s")
        print(f"   Total Pipeline: {total_time:.2f}s")

        print(f"\n🎯 Results vs Targets:")
        if total_time <= 5.0:
            print(f"   Speed: ✅ Excellent ({total_time:.1f}s ≤ 5s target)")
        elif total_time <= 10.0:
            print(f"   Speed: ⚠️  Good ({total_time:.1f}s)")
        else:
            print(f"   Speed: ❌ Needs improvement ({total_time:.1f}s > 10s)")

        if ocr_result['success']:
            if ocr_result['confidence'] >= 0.8:
                print(
                    f"   Confidence: ✅ Excellent ({ocr_result['confidence']:.2f} ≥ 0.8)")
            elif ocr_result['confidence'] >= 0.6:
                print(
                    f"   Confidence: ⚠️  Good ({ocr_result['confidence']:.2f})")
            else:
                print(
                    f"   Confidence: ❌ Needs improvement ({ocr_result['confidence']:.2f})")

        # Success criteria
        pipeline_success = (
            ocr_result['success'] and
            ocr_result['confidence'] >= 0.6 and
            total_time <= 15.0 and
            len(ocr_result['final_text']) >= 10
        )

        if pipeline_success:
            print(f"\n🎉 Enhanced OCR Pipeline: ✅ SUCCESS!")
            print("   Ready for production use!")
        else:
            print(f"\n⚠️  Enhanced OCR Pipeline: ❌ NEEDS MORE WORK")
            print("   Check individual component results above.")

        return pipeline_success

    except Exception as e:
        print(f"❌ Complete pipeline test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def create_test_assignment_with_qr():
    """Create test assignment with QR code."""
    try:
        from modules.qr_generator import create_sample_assignment, QRCodeGenerator

        # Create assignment metadata
        metadata = create_sample_assignment()
        metadata.assignment_name = "Enhanced OCR Test Assignment"

        # Generate QR code
        generator = QRCodeGenerator()
        qr_image = generator.generate_qr_image(metadata)

        if qr_image is None:
            return None

        # Create assignment background
        assignment_height, assignment_width = 600, 800
        assignment_image = np.ones(
            (assignment_height, assignment_width, 3), dtype=np.uint8) * 255

        # Add header
        cv2.putText(assignment_image, "Algebra 1 - Enhanced OCR Test",
                    (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, 0, 2)

        # Add mathematical problems with varying difficulty
        problems = [
            "1. Solve: 2x + 5 = 13",
            "2. Find x: 3x - 7 = 14",
            "3. Equation: x/4 + 3 = 7",
            "4. System: x + y = 10",
            "5. Quadratic: x^2 - 5x + 6 = 0"
        ]

        y_pos = 100
        for problem in problems:
            cv2.putText(assignment_image, problem,
                        (70, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, 0, 2)
            y_pos += 80

        # Add QR code
        qr_array = np.array(qr_image.convert('RGB'))
        qr_h, qr_w = qr_array.shape[:2]

        margin = 20
        x_pos = assignment_width - qr_w - margin
        y_pos = margin

        if x_pos > 0 and y_pos + qr_h < assignment_height:
            assignment_image[y_pos:y_pos+qr_h, x_pos:x_pos+qr_w] = qr_array

        return assignment_image

    except Exception as e:
        print(f"⚠️  QR assignment creation failed: {e}")
        return None


def create_simple_test_assignment():
    """Create simple test assignment without QR code."""
    assignment_image = np.ones((400, 600, 3), dtype=np.uint8) * 255

    cv2.putText(assignment_image, "Math Test - Enhanced OCR",
                (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, 0, 2)
    cv2.putText(assignment_image, "1. Solve: 2x + 3 = 7",
                (70, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, 0, 2)
    cv2.putText(assignment_image, "2. Find: x = (7-3)/2",
                (70, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.7, 0, 2)
    cv2.putText(assignment_image, "3. Answer: x = 2",
                (70, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.7, 0, 2)

    return assignment_image


def test_your_real_image(image_path: str):
    """Test enhancements with your actual problematic image."""
    print(f"\n📷 Testing Your Real Image: {image_path}")
    print("=" * 50)

    try:
        if not Path(image_path).exists():
            print(f"❌ Image not found: {image_path}")
            return False

        # Load your real image
        real_image = cv2.imread(image_path)
        if real_image is None:
            print(f"❌ Could not load image: {image_path}")
            return False

        # Convert BGR to RGB
        real_image = cv2.cvtColor(real_image, cv2.COLOR_BGR2RGB)

        print(f"✅ Loaded real image: {real_image.shape}")

        # Test with original pipeline (your current results)
        print(f"\n📊 Original Pipeline Results (for comparison):")
        from modules.ocr.tesseract_ocr import TesseractOCR

        tesseract = TesseractOCR()
        original_start = time.time()
        original_text, original_conf = tesseract.extract_text(real_image)
        original_time = time.time() - original_start

        print(f"   Original Tesseract: '{original_text[:50]}...'")
        print(f"   Original confidence: {original_conf:.2f}")
        print(f"   Original time: {original_time:.2f}s")
        print(f"   Original chars: {len(original_text)}")

        # Test with enhanced pipeline
        print(f"\n🚀 Enhanced Pipeline Results:")

        from modules.advanced_image_processor import AdvancedImageProcessor
        from modules.ocr.ocr_manager import OCRManager

        # Step 1: Advanced preprocessing
        processor = AdvancedImageProcessor()
        preprocess_result = processor.process_for_ocr(real_image, image_path)

        if preprocess_result['success']:
            enhanced_image = preprocess_result['processed_image']
            quality_improvement = preprocess_result['improvement_score']

            print(
                f"   Preprocessing improvement: {quality_improvement:+.1f} points")
            print(
                f"   Enhanced quality: {preprocess_result['final_quality']['overall_score']:.1f}/100")
        else:
            enhanced_image = real_image
            print(f"   Preprocessing failed, using original")

        # Step 2: Enhanced OCR
        ocr_manager = OCRManager()
        enhanced_start = time.time()
        enhanced_result = ocr_manager.extract_text(enhanced_image, image_path)
        enhanced_time = time.time() - enhanced_start

        if enhanced_result['success']:
            print(
                f"   Enhanced OCR: '{enhanced_result['final_text'][:50]}...'")
            print(
                f"   Enhanced confidence: {enhanced_result['confidence']:.2f}")
            print(f"   Enhanced time: {enhanced_time:.2f}s")
            print(f"   Enhanced chars: {len(enhanced_result['final_text'])}")
            print(
                f"   Engines used: {', '.join(enhanced_result['engines_used'])}")

        # Compare results
        print(f"\n🏆 Enhancement Comparison:")
        print("=" * 30)

        if enhanced_result['success']:
            conf_improvement = enhanced_result['confidence'] - original_conf
            char_improvement = len(
                enhanced_result['final_text']) - len(original_text)
            time_change = enhanced_time - original_time

            print(
                f"   Confidence: {original_conf:.2f} → {enhanced_result['confidence']:.2f} ({conf_improvement:+.2f})")
            print(
                f"   Characters: {len(original_text)} → {len(enhanced_result['final_text'])} ({char_improvement:+d})")
            print(
                f"   Time: {original_time:.2f}s → {enhanced_time:.2f}s ({time_change:+.2f}s)")

            if conf_improvement > 0.1:
                print(f"   🎉 Significant confidence improvement!")
            if char_improvement > 10:
                print(f"   🎉 Significant text extraction improvement!")
            if enhanced_time < original_time:
                print(f"   🎉 Faster processing!")

            return True
        else:
            print(f"   ❌ Enhanced pipeline failed")
            return False

    except Exception as e:
        print(f"❌ Real image test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Main testing function."""
    print("🔍 AutoGrading System - Enhanced OCR Testing")
    print("=" * 55)
    print("Testing all new OCR enhancements and improvements")
    print()

    # Track test results
    test_results = {}

    # Test 1: Real Pix2Text
    test_results['pix2text'] = test_real_pix2text()

    # Test 2: Fixed EasyOCR
    test_results['easyocr'] = test_fixed_easyocr()

    # Test 3: Advanced Preprocessing
    test_results['preprocessing'] = test_advanced_preprocessing()

    # Test 4: Complete Enhanced Pipeline
    test_results['complete_pipeline'] = test_complete_pipeline_with_enhancements()

    # Test 5: Your Real Image (if available)
    real_image_path = "data/input/TestScan.jpg"  # Update this path
    if Path(real_image_path).exists():
        test_results['real_image'] = test_your_real_image(real_image_path)
    else:
        print(f"\n⚠️  Real image test skipped - {real_image_path} not found")
        print("   To test with your real image:")
        print(f"   python test_enhanced_ocr.py {real_image_path}")
        test_results['real_image'] = None

    # Final Summary
    print(f"\n🎯 Final Test Summary")
    print("=" * 25)

    passed_tests = sum(1 for result in test_results.values() if result is True)
    total_tests = sum(1 for result in test_results.values()
                      if result is not None)

    for test_name, result in test_results.items():
        if result is True:
            status = "✅ PASS"
        elif result is False:
            status = "❌ FAIL"
        else:
            status = "⚠️  SKIP"

        print(f"   {test_name}: {status}")

    success_rate = (passed_tests / total_tests) if total_tests > 0 else 0

    print(f"\n📊 Overall Results:")
    print(f"   Tests passed: {passed_tests}/{total_tests}")
    print(f"   Success rate: {success_rate:.1%}")

    if success_rate >= 0.8:
        print(f"\n🎉 Enhanced OCR system is ready for production!")
    elif success_rate >= 0.6:
        print(f"\n⚠️  Enhanced OCR system needs minor fixes")
    else:
        print(f"\n❌ Enhanced OCR system needs significant work")

    print(f"\n🚀 Next Steps:")
    if not test_results.get('pix2text', False):
        print("   • Install Pix2Text: pip install pix2text")
    if not test_results.get('easyocr', False):
        print("   • Fix EasyOCR configuration issues")
    if not test_results.get('preprocessing', False):
        print("   • Debug advanced preprocessing pipeline")
    if test_results.get('real_image', None) is None:
        print("   • Test with your real problematic image")

    return success_rate >= 0.6


if __name__ == "__main__":
    # Allow testing specific image
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
        test_your_real_image(image_path)
    else:
        success = main()
        sys.exit(0 if success else 1)
