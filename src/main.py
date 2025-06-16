#!/usr/bin/env python3
"""
AutoGrading System - Main Entry Point (Updated for Week 4-5 OCR Pipeline)
AI-powered autograding system for high school mathematics assignments.
"""

from config.config import config
import sys
import click
from pathlib import Path
import time

# Add src directory to Python path
src_path = Path(__file__).parent
sys.path.insert(0, str(src_path))


@click.group()
@click.version_option(version="1.0.0")
@click.option('--debug/--no-debug', default=None, help='Enable debug mode')
@click.option('--config-file', type=click.Path(exists=True), help='Path to configuration file')
def cli(debug, config_file):
    """AutoGrading System - AI-powered mathematics assignment grading."""
    if debug is not None:
        config.set('application.debug', debug)

    if config_file:
        from config.config import reload_config
        reload_config(config_file)


@cli.command()
def test():
    """Test the installation and configuration."""
    click.echo("🎯 AutoGrading System Installation Test")
    click.echo("=" * 50)

    # Test configuration
    click.echo(f"✅ Application: {config.get('application.name', 'Unknown')}")
    click.echo(f"✅ Version: {config.get('application.version', '1.0.0')}")
    click.echo(f"✅ Debug mode: {config.debug}")
    click.echo(f"✅ Offline mode: {config.offline_mode}")

    # Test Python environment
    click.echo(f"✅ Python version: {sys.version.split()[0]}")

    # Test core dependencies
    dependencies = [
        ('OpenCV', 'cv2'),
        ('PIL/Pillow', 'PIL'),
        ('NumPy', 'numpy'),
        ('YAML', 'yaml'),
        ('Python-dotenv', 'dotenv'),
        ('SymPy', 'sympy')
    ]

    click.echo("\n📦 Testing Core Dependencies:")
    for name, module in dependencies:
        try:
            __import__(module)
            click.echo(f"✅ {name}")
        except ImportError as e:
            click.echo(f"❌ {name}: {e}")

    # Test OCR dependencies
    click.echo("\n🔍 Testing OCR Dependencies:")
    ocr_dependencies = [
        ('Tesseract', 'pytesseract'),
        ('EasyOCR', 'easyocr'),
        ('PyZbar (QR codes)', 'pyzbar'),
        ('QRCode', 'qrcode'),
        ('Transformers', 'transformers')
    ]

    for name, module in ocr_dependencies:
        try:
            __import__(module)
            click.echo(f"✅ {name}")
        except ImportError as e:
            click.echo(f"⚠️  {name}: {e} (install if needed)")

    click.echo("\n🎉 Installation test complete!")
    click.echo("\n📝 Available OCR Commands:")
    click.echo("   • python src/main.py ocr-test        # Test OCR manager")
    click.echo("   • python src/main.py tesseract-test  # Test Tesseract")
    click.echo("   • python src/main.py easyocr-test    # Test EasyOCR")
    click.echo("   • python src/main.py math-test       # Test math corrector")
    click.echo(
        "   • python src/main.py ocr-full-test   # Complete OCR pipeline test")


@cli.command('ocr-test')
def ocr_test():
    """Test the complete OCR manager and pipeline."""
    click.echo("🔍 OCR Manager Test")
    click.echo("=" * 40)

    try:
        from modules.ocr.ocr_manager import OCRManager, demo_ocr_manager

        click.echo("🧪 Running OCR manager demo...")

        start_time = time.time()
        success = demo_ocr_manager()
        end_time = time.time()

        if success:
            click.echo(f"✅ OCR manager test completed successfully!")
            click.echo(
                f"   Processing time: {end_time - start_time:.2f} seconds")
        else:
            click.echo("❌ OCR manager test failed")

        # Test individual engine availability
        click.echo("\n🔧 Testing individual engines:")

        manager = OCRManager()
        available_engines = list(manager.engines.keys())

        for engine_name in available_engines:
            try:
                # Create a simple test image
                import numpy as np
                import cv2
                test_image = np.ones((100, 200), dtype=np.uint8) * 255
                cv2.putText(test_image, "Test 123", (20, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, 0, 2)

                result = manager._run_single_engine(test_image, engine_name)

                if result.success:
                    click.echo(
                        f"✅ {engine_name}: '{result.text[:30]}...' ({result.confidence:.2f})")
                else:
                    click.echo(f"❌ {engine_name}: {result.error_message}")

            except Exception as e:
                click.echo(f"❌ {engine_name}: {e}")

        # Show engine statistics
        stats = manager.get_engine_stats()
        click.echo(f"\n📊 Engine Statistics:")
        click.echo(f"   Available engines: {stats['total_engines']}")
        click.echo(f"   Engine list: {', '.join(stats['available_engines'])}")

    except ImportError as e:
        click.echo(f"❌ OCR manager import failed: {e}")
        click.echo("Make sure you've created the OCR modules in src/modules/ocr/")
    except Exception as e:
        click.echo(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()


@cli.command('tesseract-test')
def tesseract_test():
    """Test Tesseract OCR engine specifically."""
    click.echo("📝 Tesseract OCR Test")
    click.echo("=" * 30)

    try:
        from modules.ocr.tesseract_ocr import TesseractOCR, demo_tesseract_ocr

        click.echo("🧪 Running Tesseract demo...")

        success = demo_tesseract_ocr()

        if success:
            click.echo("✅ Tesseract test completed!")

            # Additional specific tests
            click.echo("\n🔬 Running specific mathematical tests...")

            tesseract = TesseractOCR()

            # Test with different mathematical content
            math_tests = [
                "Simple: 2 + 3 = 5",
                "Variables: 2x + 5 = 11",
                "Fractions: 3/4 + 1/2",
                "Exponents: x^2 + 3x - 4"
            ]

            for test_description in math_tests:
                # Create test image
                import numpy as np
                import cv2
                test_image = np.ones((80, 300), dtype=np.uint8) * 255

                # Extract test text from description
                test_text = test_description.split(": ")[1]
                cv2.putText(test_image, test_text, (10, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, 0, 2)

                # Extract with Tesseract
                extracted_text, confidence = tesseract.extract_text(test_image)

                click.echo(f"   {test_description}")
                click.echo(
                    f"     Extracted: '{extracted_text}' (confidence: {confidence:.2f})")

        else:
            click.echo("❌ Tesseract test failed")

    except ImportError as e:
        click.echo(f"❌ Tesseract import failed: {e}")
        click.echo("Install with: pip install pytesseract")
        click.echo("System requirement: Tesseract binary must be installed")
    except Exception as e:
        click.echo(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()


@cli.command('easyocr-test')
def easyocr_test():
    """Test EasyOCR engine specifically."""
    click.echo("✍️  EasyOCR Test")
    click.echo("=" * 25)

    try:
        from modules.ocr.easy_ocr import EasyOCREngine, demo_easyocr_engine

        click.echo("🧪 Running EasyOCR demo...")

        success = demo_easyocr_engine()

        if success:
            click.echo("✅ EasyOCR test completed!")

            # Test with handwriting simulation
            click.echo("\n🖋️  Testing handwriting recognition...")

            easyocr_engine = EasyOCREngine()

            # Create handwriting-style test
            import numpy as np
            import cv2

            handwriting_image = np.ones((120, 250), dtype=np.uint8) * 255

            # Simulate handwritten style with irregular text
            font = cv2.FONT_HERSHEY_SCRIPT_SIMPLEX
            cv2.putText(handwriting_image, "x = 5", (20, 50), font, 1.0, 0, 3)
            cv2.putText(handwriting_image, "y = 2x + 1",
                        (20, 90), font, 0.8, 0, 3)

            # Add slight noise to simulate handwriting
            noise = np.random.randint(-15, 15,
                                      handwriting_image.shape, dtype=np.int16)
            handwriting_image = np.clip(handwriting_image.astype(
                np.int16) + noise, 0, 255).astype(np.uint8)

            text, confidence = easyocr_engine.extract_text(handwriting_image)

            click.echo(f"   Handwriting extraction: '{text}'")
            click.echo(f"   Confidence: {confidence:.2f}")

        else:
            click.echo("❌ EasyOCR test failed")

    except ImportError as e:
        click.echo(f"❌ EasyOCR import failed: {e}")
        click.echo("Install with: pip install easyocr")
    except Exception as e:
        click.echo(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()


@cli.command('math-test')
def math_test():
    """Test mathematical symbol corrector."""
    click.echo("🔧 Mathematical Symbol Corrector Test")
    click.echo("=" * 40)

    try:
        from modules.ocr.math_corrector import MathCorrector, demo_math_corrector

        click.echo("🧪 Running math corrector demo...")

        success = demo_math_corrector()

        if success:
            click.echo("✅ Math corrector test completed!")

            # Interactive correction test
            click.echo("\n🔬 Testing specific corrections:")

            corrector = MathCorrector()

            # Test cases that might come from OCR
            ocr_errors = [
                "2x + 3 = l",      # l instead of 1
                "O + 5 = 5",       # O instead of 0
                "x^Z - 4 = O",     # Z->2, O->0
                "3×4÷2",           # Unicode operators
                "(2+3×4",          # Missing parenthesis
                "2x+3=7",          # No spacing
                "x/Z+l/3",         # Multiple errors
            ]

            for original in ocr_errors:
                corrected = corrector.correct_mathematical_text(original)
                analysis = corrector.analyze_corrections(original)

                click.echo(f"   '{original}' → '{corrected}'")
                click.echo(
                    f"     Improvement: {analysis['estimated_improvement']:.1%}")
                click.echo(
                    f"     SymPy valid: {'✅' if analysis.get('corrected_parseable', False) else '❌'}")

            # Show correction statistics
            stats = corrector.get_correction_stats()
            click.echo(f"\n📊 Correction Stats:")
            click.echo(f"   Total corrections: {stats['total_corrections']}")
            click.echo(
                f"   Validation rate: {stats['validation_success_rate']:.1%}")

        else:
            click.echo("❌ Math corrector test failed")

    except ImportError as e:
        click.echo(f"❌ Math corrector import failed: {e}")
        click.echo("Install with: pip install sympy")
    except Exception as e:
        click.echo(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()


@cli.command('pix2text-test')
def pix2text_test():
    """Test Pix2Text mathematical formula OCR (mock version)."""
    click.echo("🧮 Pix2Text Engine Test")
    click.echo("=" * 30)

    try:
        from modules.ocr.mock_pix2text import MockPix2TextEngine, demo_mock_pix2text

        click.echo("🧪 Running Pix2Text demo (mock version)...")

        success = demo_mock_pix2text()

        if success:
            click.echo("✅ Pix2Text test completed!")

            # Show capabilities
            mock_engine = MockPix2TextEngine()
            capabilities = mock_engine.get_capabilities()

            click.echo(f"\n⚙️  Engine Capabilities:")
            click.echo(f"   Engine: {capabilities['engine_name']}")
            click.echo(
                f"   Supported formats: {', '.join(capabilities['supported_formats'])}")
            click.echo(
                f"   Formula types: {len(capabilities['formula_types'])}")
            click.echo(f"   Mock mode: {capabilities['mock_mode']}")

        else:
            click.echo("❌ Pix2Text test failed")

    except ImportError as e:
        click.echo(f"❌ Pix2Text import failed: {e}")
    except Exception as e:
        click.echo(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()


@cli.command('ocr-full-test')
def ocr_full_test():
    """Run comprehensive OCR pipeline test with sample assignment."""
    click.echo("🎯 Complete OCR Pipeline Test")
    click.echo("=" * 45)

    try:
        # Import required modules
        from modules.image_processor import ImageProcessor
        from modules.qr_extractor import QRExtractor
        from modules.ocr.ocr_manager import OCRManager

        click.echo("🏗️  Setting up complete pipeline...")

        # Initialize components
        image_processor = ImageProcessor()
        qr_extractor = QRExtractor()
        ocr_manager = OCRManager()

        click.echo(f"✅ Components initialized:")
        click.echo(f"   Image Processor: Ready")
        click.echo(f"   QR Extractor: Ready")
        click.echo(
            f"   OCR Manager: {len(ocr_manager.engines)} engines available")

        # Create comprehensive test assignment
        click.echo("\n📝 Creating test assignment with QR code...")

        # Import numpy and cv2 here
        import numpy as np
        import cv2

        try:
            from modules.qr_generator import create_sample_assignment, QRCodeGenerator

            # Create assignment metadata
            metadata = create_sample_assignment()
            metadata.assignment_name = "OCR Pipeline Test Assignment"
            metadata.subject = "algebra1"

            # Generate QR code
            qr_generator = QRCodeGenerator()

            # Try different QR generation methods
            qr_image = None
            try:
                # Method 1: Try generate_qr_image
                qr_image = qr_generator.generate_qr_image(metadata)
            except AttributeError:
                try:
                    # Method 2: Try generate_qr_code
                    qr_image = qr_generator.generate_qr_code(
                        metadata.to_json())
                except AttributeError:
                    # Method 3: Try basic generation
                    import qrcode
                    qr = qrcode.QRCode(version=1, box_size=10, border=5)
                    qr.add_data(metadata.to_json())
                    qr.make(fit=True)
                    qr_image = qr.make_image(
                        fill_color="black", back_color="white")

            # Create assignment with mathematical content
            import numpy as np
            import cv2

            # Create assignment background
            assignment_height, assignment_width = 600, 800
            assignment_image = np.ones(
                (assignment_height, assignment_width, 3), dtype=np.uint8) * 255

            # Add header
            cv2.putText(assignment_image, "Algebra 1 - Linear Equations",
                        (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, 0, 2)

            # Add mathematical problems
            problems = [
                "1. Solve for x: 2x + 5 = 13",
                "2. Find x: 3x - 7 = 14",
                "3. Equation: x/4 + 3 = 7",
                "4. Linear: y = 2x + 1",
                "5. System: x + y = 10, x - y = 2"
            ]

            y_pos = 100
            for problem in problems:
                cv2.putText(assignment_image, problem,
                            (70, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, 0, 2)
                y_pos += 80

            # Add QR code to top-right corner
            if qr_image is not None:
                qr_array = np.array(qr_image.convert('RGB'))
                qr_h, qr_w = qr_array.shape[:2]

                # Place QR code
                margin = 20
                x_pos = assignment_width - qr_w - margin
                y_pos = margin

                if x_pos > 0 and y_pos + qr_h < assignment_height:
                    assignment_image[y_pos:y_pos+qr_h,
                                     x_pos:x_pos+qr_w] = qr_array
                    click.echo("✅ QR code embedded in assignment")

            click.echo("✅ Test assignment created")

        except Exception as e:
            click.echo(f"⚠️  QR generation failed: {e}")
            click.echo("   Creating assignment without QR code...")

            # Create simple assignment without QR
            assignment_image = np.ones((400, 600, 3), dtype=np.uint8) * 255
            cv2.putText(assignment_image, "Math Test: 2x + 3 = 7",
                        (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, 0, 3)

        # Step 1: Image Processing
        click.echo("\n🖼️  Step 1: Image Processing...")

        start_time = time.time()
        processed_result = image_processor.process_image_array(
            assignment_image)
        processing_time = time.time() - start_time

        if processed_result and processed_result.get('success', False):
            click.echo(f"✅ Image processed in {processing_time:.2f}s")
            quality = processed_result['quality_metrics']
            click.echo(f"   Quality score: {quality['overall_score']}/100")
            click.echo(
                f"   OCR suitable: {'✅' if quality['is_suitable_for_ocr'] else '❌'}")

            processed_image = processed_result['processed_image']
        else:
            click.echo("⚠️  Using original image")
            processed_image = assignment_image

        # Step 2: QR Code Extraction
        click.echo("\n🔲 Step 2: QR Code Extraction...")

        start_time = time.time()
        qr_codes = qr_extractor.extract_qr_codes(processed_image)
        qr_time = time.time() - start_time

        click.echo(f"✅ QR extraction completed in {qr_time:.2f}s")
        click.echo(f"   QR codes found: {len(qr_codes)}")

        if qr_codes:
            for i, qr_code in enumerate(qr_codes):
                click.echo(
                    f"   QR {i+1}: {qr_code.get('assignment_name', 'Unknown')}")

        # Step 3: OCR Text Extraction
        click.echo("\n🔍 Step 3: OCR Text Extraction...")

        start_time = time.time()
        ocr_result = ocr_manager.extract_text(
            processed_image, "test_assignment.jpg")
        ocr_time = time.time() - start_time

        if ocr_result['success']:
            click.echo(f"✅ OCR completed in {ocr_time:.2f}s")
            click.echo(
                f"   Engines used: {', '.join(ocr_result['engines_used'])}")
            click.echo(f"   Final confidence: {ocr_result['confidence']:.2f}")
            click.echo(
                f"   Text length: {len(ocr_result['final_text'])} characters")
            click.echo(
                f"   Quality: {ocr_result['quality_metrics']['overall_quality']}")

            # Show extracted text (truncated)
            text_preview = ocr_result['final_text'][:200]
            if len(ocr_result['final_text']) > 200:
                text_preview += "..."

            click.echo(f"\n📄 Extracted Text Preview:")
            click.echo(f"   \"{text_preview}\"")

            # Show individual engine results
            click.echo(f"\n🔧 Individual Engine Results:")
            for engine_result in ocr_result['engine_results']:
                status = "✅" if engine_result['success'] else "❌"
                click.echo(f"   {status} {engine_result['engine_name']}: "
                           f"{engine_result['confidence']:.2f} confidence, "
                           f"{engine_result['character_count']} chars")
        else:
            click.echo(
                f"❌ OCR failed: {ocr_result.get('error_message', 'Unknown error')}")

        # Calculate total pipeline time
        total_time = processing_time + qr_time + ocr_time

        # Final Results Summary
        click.echo("\n📊 Pipeline Performance Summary:")
        click.echo("=" * 40)
        click.echo(f"📏 Timing Breakdown:")
        click.echo(f"   Image Processing: {processing_time:.2f}s")
        click.echo(f"   QR Extraction: {qr_time:.2f}s")
        click.echo(f"   OCR Processing: {ocr_time:.2f}s")
        click.echo(f"   Total Pipeline: {total_time:.2f}s")

        click.echo(f"\n🎯 Results:")
        click.echo(f"   QR Codes Found: {len(qr_codes)}")
        click.echo(f"   OCR Success: {'✅' if ocr_result['success'] else '❌'}")

        if ocr_result['success']:
            click.echo(
                f"   Text Extracted: {len(ocr_result['final_text'])} characters")
            click.echo(f"   Final Confidence: {ocr_result['confidence']:.2f}")
            click.echo(
                f"   Processing Quality: {ocr_result['quality_metrics']['overall_quality']}")

        # Performance benchmarks
        click.echo(f"\n⚡ Performance Analysis:")
        if total_time <= 5.0:
            click.echo(
                f"   Speed: ✅ Excellent ({total_time:.1f}s ≤ 5s target)")
        elif total_time <= 10.0:
            click.echo(f"   Speed: ⚠️  Good ({total_time:.1f}s)")
        else:
            click.echo(
                f"   Speed: ❌ Needs improvement ({total_time:.1f}s > 10s)")

        if ocr_result['success'] and ocr_result['confidence'] >= 0.7:
            click.echo(
                f"   Confidence: ✅ High ({ocr_result['confidence']:.2f} ≥ 0.7)")
        elif ocr_result['success'] and ocr_result['confidence'] >= 0.5:
            click.echo(
                f"   Confidence: ⚠️  Moderate ({ocr_result['confidence']:.2f})")
        else:
            click.echo(
                f"   Confidence: ❌ Low ({ocr_result.get('confidence', 0):.2f})")

        # Success determination
        pipeline_success = (
            ocr_result['success'] and
            ocr_result['confidence'] >= 0.5 and
            total_time <= 15.0 and
            len(ocr_result['final_text']) >= 10
        )

        if pipeline_success:
            click.echo(f"\n🎉 Complete OCR Pipeline Test: ✅ PASSED")
            click.echo("   Ready for Week 6 AI Grading integration!")
        else:
            click.echo(f"\n⚠️  Complete OCR Pipeline Test: ❌ NEEDS IMPROVEMENT")
            click.echo("   Review individual component results above.")

        return pipeline_success

    except ImportError as e:
        click.echo(f"❌ Pipeline component import failed: {e}")
        click.echo("Make sure all OCR modules are properly created")
        return False
    except Exception as e:
        click.echo(f"❌ Full pipeline test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


# Helper function for image processing from array
def add_image_processor_array_method():
    """Add method to process image from numpy array."""
    try:
        from modules.image_processor import ImageProcessor
        import tempfile
        import cv2

        def process_image_array(self, image_array):
            """Process image from numpy array."""
            try:
                # Save array to temporary file
                with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
                    # Convert RGB to BGR for OpenCV
                    if len(image_array.shape) == 3:
                        bgr_image = cv2.cvtColor(
                            image_array, cv2.COLOR_RGB2BGR)
                    else:
                        bgr_image = image_array

                    cv2.imwrite(tmp.name, bgr_image)

                    # Process using existing method
                    result = self.process_image(tmp.name)
                    result['success'] = True

                    # Cleanup
                    Path(tmp.name).unlink(missing_ok=True)

                    return result

            except Exception as e:
                return {'success': False, 'error': str(e)}

        # Add method to ImageProcessor class
        ImageProcessor.process_image_array = process_image_array

    except ImportError:
        pass  # Module not available yet


# Add the helper method
add_image_processor_array_method()


@cli.command()
def status():
    """Show current development status."""
    click.echo("🚀 AutoGrading System Development Status")
    click.echo("=" * 45)

    phases = [
        ("Week 1", "Environment Setup", "✅ Complete"),
        ("Week 2", "QR Code System", "✅ Complete"),
        ("Week 3", "Image Processing", "✅ Complete"),
        ("Week 4-5", "OCR Pipeline", "🔄 In Progress"),
        ("Week 6", "AI Grading", "⏳ Pending"),
        ("Week 7", "PDF Generation", "⏳ Pending"),
        ("Week 8", "External Integrations", "⏳ Pending"),
        ("Week 9", "Main Application", "⏳ Pending"),
        ("Week 10", "Testing & QA", "⏳ Pending"),
        ("Week 11", "Documentation", "⏳ Pending")
    ]

    for week, phase, status in phases:
        click.echo(f"{week:8} | {phase:20} | {status}")

    click.echo("\n📋 Week 4-5 Current Tasks:")
    click.echo("✅ OCR engine manager with multi-engine support")
    click.echo("✅ Tesseract OCR with mathematical optimization")
    click.echo("✅ EasyOCR for handwritten content recognition")
    click.echo("✅ Mathematical symbol corrector with SymPy validation")
    click.echo("✅ Mock Pix2Text for mathematical formula OCR")
    click.echo("✅ Complete OCR pipeline integration")

    click.echo("\n🧪 Available OCR Tests:")
    click.echo("• python src/main.py ocr-test         # Test OCR manager")
    click.echo("• python src/main.py tesseract-test   # Test Tesseract engine")
    click.echo("• python src/main.py easyocr-test     # Test EasyOCR engine")
    click.echo("• python src/main.py math-test        # Test math corrector")
    click.echo("• python src/main.py pix2text-test    # Test Pix2Text (mock)")
    click.echo("• python src/main.py ocr-full-test    # Complete pipeline test")

    click.echo("\n📁 Week 6 Preview:")
    click.echo("• Mock AI grading system with GPT simulation")
    click.echo("• Subject-specific rubrics (Algebra 1, AP Calculus)")
    click.echo("• Manual review queue for low-confidence results")
    click.echo("• Integration with OCR pipeline output")


# Keep existing commands for backward compatibility
@cli.command('qr-test')
def qr_test():
    """Test QR code generation functionality."""
    click.echo("🔲 QR Code Generation Test")
    click.echo("=" * 40)

    try:
        from modules.qr_generator import demo_qr_generation, create_sample_assignment, QRCodeGenerator

        click.echo("🧪 Testing QR code generation...")

        metadata = create_sample_assignment()
        click.echo(f"✅ Created sample assignment: {metadata.assignment_name}")

        json_data = metadata.to_json()
        click.echo(f"✅ Metadata JSON size: {len(json_data)} characters")

        generator = QRCodeGenerator()
        test_id = generator.generate_assignment_id("TEST")
        click.echo(f"✅ Generated test ID: {test_id}")

        try:
            result = demo_qr_generation()
            if result:
                click.echo("✅ Full QR generation demo completed successfully")
            else:
                click.echo("⚠️  QR generation demo had issues")
        except ImportError as e:
            click.echo(f"⚠️  QR image generation not available: {e}")

        click.echo("\n🎉 QR code system test completed!")

    except ImportError as e:
        click.echo(f"❌ QR module import failed: {e}")
    except Exception as e:
        click.echo(f"❌ Test failed: {e}")


@cli.command()
@click.argument('image_path', type=click.Path(exists=True))
def process(image_path):
    """Process a single assignment image through the OCR pipeline."""
    click.echo(f"📄 Processing assignment: {image_path}")

    try:
        from modules.image_processor import ImageProcessor
        from modules.qr_extractor import QRExtractor
        from modules.ocr.ocr_manager import OCRManager

        # Initialize pipeline components
        image_processor = ImageProcessor()
        qr_extractor = QRExtractor()
        ocr_manager = OCRManager()

        click.echo("🏗️  Processing pipeline initialized")

        # Process image
        click.echo("🖼️  Processing image...")
        start_time = time.time()

        processed_result = image_processor.process_image(image_path)
        processing_time = time.time() - start_time

        if processed_result:
            click.echo(f"✅ Image processed in {processing_time:.2f}s")

            # Extract QR codes
            click.echo("🔲 Extracting QR codes...")
            qr_codes = qr_extractor.extract_from_file(image_path)
            click.echo(f"✅ Found {len(qr_codes)} QR codes")

            # Extract text with OCR
            click.echo("🔍 Extracting text with OCR...")
            ocr_start = time.time()
            ocr_result = ocr_manager.extract_text(
                processed_result['processed_image'], image_path)
            ocr_time = time.time() - ocr_start

            if ocr_result['success']:
                click.echo(f"✅ OCR completed in {ocr_time:.2f}s")
                click.echo(f"   Confidence: {ocr_result['confidence']:.2f}")
                click.echo(
                    f"   Text length: {len(ocr_result['final_text'])} characters")

                # Show preview of extracted text
                preview = ocr_result['final_text'][:100]
                if len(ocr_result['final_text']) > 100:
                    preview += "..."
                click.echo(f"   Preview: \"{preview}\"")
            else:
                click.echo(f"❌ OCR failed: {ocr_result.get('error_message')}")

        else:
            click.echo("❌ Image processing failed")

    except ImportError as e:
        click.echo(f"❌ Import failed: {e}")
        click.echo("Run 'python src/main.py ocr-test' to check OCR setup")
    except Exception as e:
        click.echo(f"❌ Processing failed: {e}")


@cli.command()
@click.argument('input_dir', type=click.Path(exists=True))
def batch(input_dir):
    """Process multiple assignments from a directory (placeholder for Week 9)."""
    click.echo(f"📁 Batch processing from: {input_dir}")
    click.echo("Advanced batch processing will be available in Week 9")
    click.echo("Current status: OCR pipeline components ready")


if __name__ == "__main__":
    cli()
