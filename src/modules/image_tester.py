#!/usr/bin/env python3
"""
AutoGrading System - Image Processing Testing Module
Comprehensive testing for image processing and QR extraction capabilities.
"""

import cv2
import numpy as np
from typing import Dict, List, Any, Tuple
import tempfile
import time
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class ImageProcessingTester:
    """Comprehensive testing suite for image processing capabilities."""

    def __init__(self):
        """Initialize the testing suite."""
        self.test_results = []
        self.temp_files = []

    def run_all_tests(self) -> Dict[str, Any]:
        """Run all image processing tests.

        Returns:
            Dictionary containing test results and summary
        """
        print("🧪 Running Image Processing Test Suite")
        print("=" * 50)

        start_time = time.time()

        # Run individual test categories
        tests = [
            ("Basic Image Processing", self._test_basic_processing),
            ("QR Code Detection", self._test_qr_detection),
            ("Quality Assessment", self._test_quality_assessment),
            ("File Format Support", self._test_file_formats),
            ("Robustness Testing", self._test_robustness),
            ("Performance Testing", self._test_performance)
        ]

        overall_results = {
            'tests_run': 0,
            'tests_passed': 0,
            'test_details': [],
            'total_time': 0.0,
            'success_rate': 0.0
        }

        for test_name, test_func in tests:
            print(f"\n📋 {test_name}")
            print("-" * 30)

            try:
                result = test_func()
                overall_results['test_details'].append({
                    'name': test_name,
                    'result': result,
                    'passed': result.get('passed', False)
                })

                overall_results['tests_run'] += 1
                if result.get('passed', False):
                    overall_results['tests_passed'] += 1

            except Exception as e:
                print(f"❌ Test category failed: {e}")
                overall_results['test_details'].append({
                    'name': test_name,
                    'result': {'passed': False, 'error': str(e)},
                    'passed': False
                })
                overall_results['tests_run'] += 1

        # Calculate summary statistics
        overall_results['total_time'] = time.time() - start_time
        if overall_results['tests_run'] > 0:
            overall_results['success_rate'] = (
                overall_results['tests_passed'] /
                overall_results['tests_run'] * 100
            )

        self._cleanup_temp_files()
        self._print_summary(overall_results)

        return overall_results

    def _test_basic_processing(self) -> Dict[str, Any]:
        """Test basic image processing pipeline."""
        try:
            from modules.image_processor import ImageProcessor, create_sample_test_image

            processor = ImageProcessor()
            test_image = create_sample_test_image()

            results = {
                'passed': True,
                'subtests': [],
                'errors': []
            }

            # Test individual processing steps
            subtests = [
                ("Denoising", lambda: processor._denoise_image(test_image)),
                ("Deskewing", lambda: processor._deskew_image(test_image)[0]),
                ("Contrast Enhancement",
                 lambda: processor._enhance_contrast(test_image)),
                ("Resizing", lambda: processor._resize_image(test_image, 1000)),
                ("Quality Assessment",
                 lambda: processor._assess_image_quality(test_image))
            ]

            for test_name, test_func in subtests:
                try:
                    result = test_func()
                    if result is not None:
                        print(f"✅ {test_name}")
                        results['subtests'].append(
                            {'name': test_name, 'passed': True})
                    else:
                        print(f"❌ {test_name}: Returned None")
                        results['subtests'].append(
                            {'name': test_name, 'passed': False})
                        results['passed'] = False
                except Exception as e:
                    print(f"❌ {test_name}: {e}")
                    results['subtests'].append(
                        {'name': test_name, 'passed': False, 'error': str(e)})
                    results['errors'].append(f"{test_name}: {e}")
                    results['passed'] = False

            # Test full pipeline
            try:
                temp_file = self._create_temp_image(test_image)
                pipeline_result = processor.process_image(temp_file)

                if pipeline_result and 'processed_image' in pipeline_result:
                    print("✅ Full processing pipeline")
                    results['subtests'].append(
                        {'name': 'Full Pipeline', 'passed': True})
                else:
                    print("❌ Full processing pipeline: Invalid result")
                    results['passed'] = False

            except Exception as e:
                print(f"❌ Full processing pipeline: {e}")
                results['errors'].append(f"Full Pipeline: {e}")
                results['passed'] = False

            return results

        except ImportError as e:
            return {'passed': False, 'error': f"Import failed: {e}"}
        except Exception as e:
            return {'passed': False, 'error': str(e)}

    def _test_qr_detection(self) -> Dict[str, Any]:
        """Test QR code detection and extraction."""
        try:
            from modules.qr_extractor import QRExtractor, create_test_image_with_qr

            extractor = QRExtractor()

            results = {
                'passed': True,
                'subtests': [],
                'errors': [],
                'qr_codes_detected': 0
            }

            # Test QR detection on sample image
            test_image, original_metadata = create_test_image_with_qr()

            if test_image is not None:
                try:
                    qr_codes = extractor.extract_qr_codes(test_image)
                    results['qr_codes_detected'] = len(qr_codes)

                    if qr_codes:
                        print(f"✅ QR Detection: Found {len(qr_codes)} codes")
                        results['subtests'].append(
                            {'name': 'QR Detection', 'passed': True})

                        # Test metadata validation
                        valid_metadata = 0
                        for qr_code in qr_codes:
                            if qr_code.get('extraction_success', False):
                                valid_metadata += 1

                        if valid_metadata > 0:
                            print(
                                f"✅ Metadata Extraction: {valid_metadata} valid")
                            results['subtests'].append(
                                {'name': 'Metadata Extraction', 'passed': True})
                        else:
                            print("❌ Metadata Extraction: No valid metadata")
                            results['subtests'].append(
                                {'name': 'Metadata Extraction', 'passed': False})
                            results['passed'] = False
                    else:
                        print("❌ QR Detection: No codes found")
                        results['subtests'].append(
                            {'name': 'QR Detection', 'passed': False})
                        results['passed'] = False

                except Exception as e:
                    print(f"❌ QR Detection failed: {e}")
                    results['errors'].append(f"QR Detection: {e}")
                    results['passed'] = False
            else:
                print("⚠️  Skipping QR detection - no test image available")
                results['subtests'].append(
                    {'name': 'QR Detection', 'passed': False, 'skipped': True})

            # Test detection methods
            available_methods = list(extractor.detectors.keys())
            print(
                f"✅ Detection methods available: {', '.join(available_methods)}")
            results['detection_methods'] = available_methods

            return results

        except ImportError as e:
            return {'passed': False, 'error': f"Import failed: {e}"}
        except Exception as e:
            return {'passed': False, 'error': str(e)}

    def _test_quality_assessment(self) -> Dict[str, Any]:
        """Test image quality assessment capabilities."""
        try:
            from modules.image_processor import ImageProcessor

            processor = ImageProcessor()

            results = {
                'passed': True,
                'subtests': [],
                'quality_scores': []
            }

            # Create test images with different qualities
            test_cases = [
                ("High Quality", self._create_high_quality_image()),
                ("Low Quality", self._create_low_quality_image()),
                ("Blurry", self._create_blurry_image()),
                ("Low Contrast", self._create_low_contrast_image())
            ]

            for case_name, test_image in test_cases:
                try:
                    quality = processor._assess_image_quality(test_image)
                    score = quality.get('overall_score', 0)
                    results['quality_scores'].append(
                        {'case': case_name, 'score': score})

                    print(f"✅ {case_name}: Quality score {score}/100")
                    results['subtests'].append(
                        {'name': case_name, 'passed': True, 'score': score})

                except Exception as e:
                    print(f"❌ {case_name}: {e}")
                    results['subtests'].append(
                        {'name': case_name, 'passed': False, 'error': str(e)})
                    results['passed'] = False

            # Verify score differentiation
            if len(results['quality_scores']) >= 2:
                scores = [item['score'] for item in results['quality_scores']]
                if max(scores) - min(scores) > 20:  # Should have meaningful score differences
                    print("✅ Quality differentiation: Scores vary appropriately")
                    results['subtests'].append(
                        {'name': 'Score Differentiation', 'passed': True})
                else:
                    print("⚠️  Quality differentiation: Scores too similar")
                    results['subtests'].append(
                        {'name': 'Score Differentiation', 'passed': False})

            return results

        except ImportError as e:
            return {'passed': False, 'error': f"Import failed: {e}"}
        except Exception as e:
            return {'passed': False, 'error': str(e)}

    def _test_file_formats(self) -> Dict[str, Any]:
        """Test support for different file formats."""
        try:
            from modules.image_processor import ImageProcessor

            processor = ImageProcessor()
            test_image = self._create_high_quality_image()

            results = {
                'passed': True,
                'subtests': [],
                'supported_formats': []
            }

            # Test different image formats
            formats = ['.jpg', '.jpeg', '.png']

            for fmt in formats:
                try:
                    temp_file = self._create_temp_image(test_image, fmt)
                    result = processor.process_image(temp_file)

                    if result and 'processed_image' in result:
                        print(f"✅ {fmt.upper()} format")
                        results['subtests'].append(
                            {'name': f'{fmt.upper()} Support', 'passed': True})
                        results['supported_formats'].append(fmt)
                    else:
                        print(f"❌ {fmt.upper()} format: Processing failed")
                        results['subtests'].append(
                            {'name': f'{fmt.upper()} Support', 'passed': False})
                        results['passed'] = False

                except Exception as e:
                    print(f"❌ {fmt.upper()} format: {e}")
                    results['subtests'].append(
                        {'name': f'{fmt.upper()} Support', 'passed': False, 'error': str(e)})
                    results['passed'] = False

            return results

        except ImportError as e:
            return {'passed': False, 'error': f"Import failed: {e}"}
        except Exception as e:
            return {'passed': False, 'error': str(e)}

    def _test_robustness(self) -> Dict[str, Any]:
        """Test robustness under various conditions."""
        try:
            from modules.image_processor import ImageProcessor

            processor = ImageProcessor()

            results = {
                'passed': True,
                'subtests': [],
                'robustness_score': 0
            }

            # Test various challenging conditions
            test_conditions = [
                ("Rotated Image", self._create_rotated_image()),
                ("Noisy Image", self._create_noisy_image()),
                ("Very Small Image", self._create_small_image()),
                ("Very Large Image", self._create_large_image()),
                ("Extreme Contrast", self._create_extreme_contrast_image())
            ]

            passed_tests = 0

            for condition_name, test_image in test_conditions:
                try:
                    temp_file = self._create_temp_image(test_image)
                    result = processor.process_image(temp_file)

                    if result and 'processed_image' in result:
                        print(f"✅ {condition_name}")
                        results['subtests'].append(
                            {'name': condition_name, 'passed': True})
                        passed_tests += 1
                    else:
                        print(f"❌ {condition_name}: Processing failed")
                        results['subtests'].append(
                            {'name': condition_name, 'passed': False})

                except Exception as e:
                    print(f"❌ {condition_name}: {e}")
                    results['subtests'].append(
                        {'name': condition_name, 'passed': False, 'error': str(e)})

            # Calculate robustness score
            results['robustness_score'] = (
                passed_tests / len(test_conditions)) * 100
            # 60% pass rate required
            results['passed'] = results['robustness_score'] >= 60

            print(f"📊 Robustness Score: {results['robustness_score']:.1f}%")

            return results

        except ImportError as e:
            return {'passed': False, 'error': f"Import failed: {e}"}
        except Exception as e:
            return {'passed': False, 'error': str(e)}

    def _test_performance(self) -> Dict[str, Any]:
        """Test processing performance."""
        try:
            from modules.image_processor import ImageProcessor

            processor = ImageProcessor()

            results = {
                'passed': True,
                'subtests': [],
                'timing_results': []
            }

            # Test processing times for different image sizes
            test_sizes = [
                ("Small (400x300)", (400, 300)),
                ("Medium (800x600)", (800, 600)),
                ("Large (1600x1200)", (1600, 1200))
            ]

            for size_name, (width, height) in test_sizes:
                try:
                    # Create test image of specific size
                    test_image = np.ones(
                        (height, width, 3), dtype=np.uint8) * 255

                    # Add some content
                    cv2.rectangle(test_image, (50, 50),
                                  (width-50, height-50), (0, 0, 0), 2)

                    # Time the processing
                    temp_file = self._create_temp_image(test_image)

                    start_time = time.time()
                    result = processor.process_image(temp_file)
                    processing_time = time.time() - start_time

                    results['timing_results'].append({
                        'size': size_name,
                        'dimensions': (width, height),
                        'time': processing_time
                    })

                    # Performance requirements (should process within reasonable time)
                    time_limit = 10.0  # 10 seconds for any image

                    if processing_time <= time_limit:
                        print(f"✅ {size_name}: {processing_time:.2f}s")
                        results['subtests'].append(
                            {'name': size_name, 'passed': True, 'time': processing_time})
                    else:
                        print(
                            f"❌ {size_name}: {processing_time:.2f}s (too slow)")
                        results['subtests'].append(
                            {'name': size_name, 'passed': False, 'time': processing_time})
                        results['passed'] = False

                except Exception as e:
                    print(f"❌ {size_name}: {e}")
                    results['subtests'].append(
                        {'name': size_name, 'passed': False, 'error': str(e)})
                    results['passed'] = False

            return results

        except ImportError as e:
            return {'passed': False, 'error': f"Import failed: {e}"}
        except Exception as e:
            return {'passed': False, 'error': str(e)}

    # Helper methods for creating test images
    def _create_high_quality_image(self) -> np.ndarray:
        """Create a high-quality test image."""
        image = np.ones((600, 800, 3), dtype=np.uint8) * 255
        cv2.rectangle(image, (50, 50), (750, 100), (0, 0, 0), 2)
        cv2.rectangle(image, (50, 120), (700, 150), (0, 0, 0), 1)
        return image

    def _create_low_quality_image(self) -> np.ndarray:
        """Create a low-quality test image."""
        image = self._create_high_quality_image()
        # Add noise
        noise = np.random.randint(0, 50, image.shape, dtype=np.uint8)
        return cv2.add(image, noise)

    def _create_blurry_image(self) -> np.ndarray:
        """Create a blurry test image."""
        image = self._create_high_quality_image()
        return cv2.GaussianBlur(image, (15, 15), 0)

    def _create_low_contrast_image(self) -> np.ndarray:
        """Create a low-contrast test image."""
        image = self._create_high_quality_image()
        # Reduce contrast by scaling pixel values toward gray
        gray_value = 128
        return cv2.addWeighted(image, 0.3, np.full_like(image, gray_value), 0.7, 0)

    def _create_rotated_image(self) -> np.ndarray:
        """Create a rotated test image."""
        image = self._create_high_quality_image()
        h, w = image.shape[:2]
        center = (w // 2, h // 2)
        rotation_matrix = cv2.getRotationMatrix2D(
            center, 15, 1.0)  # 15 degree rotation
        return cv2.warpAffine(image, rotation_matrix, (w, h), borderValue=(255, 255, 255))

    def _create_noisy_image(self) -> np.ndarray:
        """Create a very noisy test image."""
        image = self._create_high_quality_image()
        noise = np.random.randint(0, 100, image.shape, dtype=np.uint8)
        return cv2.add(image, noise)

    def _create_small_image(self) -> np.ndarray:
        """Create a very small test image."""
        image = np.ones((150, 200, 3), dtype=np.uint8) * 255
        cv2.rectangle(image, (10, 10), (190, 30), (0, 0, 0), 1)
        cv2.rectangle(image, (10, 40), (180, 60), (0, 0, 0), 1)
        return image

    def _create_large_image(self) -> np.ndarray:
        """Create a very large test image."""
        image = np.ones((2000, 3000, 3), dtype=np.uint8) * 255
        cv2.rectangle(image, (100, 100), (2900, 200), (0, 0, 0), 4)
        cv2.rectangle(image, (100, 250), (2800, 350), (0, 0, 0), 2)
        return image

    def _create_extreme_contrast_image(self) -> np.ndarray:
        """Create an image with extreme contrast."""
        image = np.ones((600, 800, 3), dtype=np.uint8) * 255
        # Pure black rectangles on white background
        cv2.rectangle(image, (50, 50), (750, 100), (0, 0, 0), -1)
        cv2.rectangle(image, (50, 120), (700, 150), (0, 0, 0), -1)
        return image

    def _create_temp_image(self, image: np.ndarray, format: str = '.jpg') -> str:
        """Create a temporary image file."""
        temp_file = tempfile.NamedTemporaryFile(suffix=format, delete=False)
        temp_path = temp_file.name
        temp_file.close()

        # Convert RGB to BGR for OpenCV
        if len(image.shape) == 3:
            bgr_image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        else:
            bgr_image = image

        cv2.imwrite(temp_path, bgr_image)
        self.temp_files.append(temp_path)
        return temp_path

    def _cleanup_temp_files(self):
        """Clean up temporary files."""
        for temp_file in self.temp_files:
            try:
                Path(temp_file).unlink(missing_ok=True)
            except Exception as e:
                logger.warning(f"Failed to cleanup temp file {temp_file}: {e}")
        self.temp_files = []

    def _print_summary(self, results: Dict[str, Any]):
        """Print test summary."""
        print("\n" + "=" * 50)
        print("🎯 IMAGE PROCESSING TEST SUMMARY")
        print("=" * 50)

        print(f"📊 Overall Results:")
        print(f"   Tests Run: {results['tests_run']}")
        print(f"   Tests Passed: {results['tests_passed']}")
        print(f"   Success Rate: {results['success_rate']:.1f}%")
        print(f"   Total Time: {results['total_time']:.2f}s")

        if results['success_rate'] >= 90:
            print("✅ EXCELLENT: Image processing system is highly robust!")
        elif results['success_rate'] >= 75:
            print("✅ GOOD: Image processing system is working well")
        elif results['success_rate'] >= 50:
            print("⚠️  ACCEPTABLE: Image processing has some issues")
        else:
            print("❌ NEEDS WORK: Image processing system requires fixes")

        print("\n📋 Test Category Details:")
        for test_detail in results['test_details']:
            name = test_detail['name']
            passed = test_detail['passed']
            status = "✅ PASS" if passed else "❌ FAIL"
            print(f"   {name}: {status}")

            # Show sub-test details if available
            result = test_detail['result']
            if 'subtests' in result:
                for subtest in result['subtests'][:3]:  # Show first 3 subtests
                    sub_status = "✅" if subtest['passed'] else "❌"
                    print(f"     {sub_status} {subtest['name']}")

        print("\n🔧 Next Steps:")
        if results['success_rate'] < 100:
            print("   • Review failed tests and fix issues")
            print("   • Check dependencies and configurations")
            print("   • Run individual test categories for debugging")
        else:
            print("   • All tests passed! Ready to proceed to Week 4 (OCR Pipeline)")

        print("   • Integration tests with real scanned assignments")
        print("   • Performance optimization if needed")


def demo_comprehensive_testing():
    """Run comprehensive image processing testing demo."""
    print("🚀 AutoGrading Image Processing - Comprehensive Testing")
    print("=" * 60)
    print("This test suite validates all Week 3 image processing capabilities")
    print()

    try:
        tester = ImageProcessingTester()
        results = tester.run_all_tests()

        # Return results for CLI integration
        return results

    except Exception as e:
        print(f"❌ Test suite failed to run: {e}")
        import traceback
        traceback.print_exc()
        return {
            'tests_run': 0,
            'tests_passed': 0,
            'success_rate': 0.0,
            'error': str(e)
        }


class QuickImageTest:
    """Quick image processing test for basic validation."""

    @staticmethod
    def run_quick_test() -> bool:
        """Run a quick validation test."""
        print("⚡ Quick Image Processing Test")
        print("-" * 30)

        try:
            # Test basic imports with proper module path
            from modules.image_processor import ImageProcessor, create_sample_test_image
            from modules.qr_extractor import QRExtractor

            print("✅ Module imports successful")

            # Test basic functionality
            processor = ImageProcessor()
            test_image = create_sample_test_image()

            # Test a simple processing step
            result = processor._assess_image_quality(test_image)

            if result and 'overall_score' in result:
                score = result['overall_score']
                print(f"✅ Quality assessment: {score}/100")

                # Test QR extractor initialization
                extractor = QRExtractor()
                methods = list(extractor.detectors.keys())
                print(f"✅ QR detection methods: {', '.join(methods)}")

                print("🎉 Quick test PASSED - basic functionality working")
                return True
            else:
                print("❌ Quality assessment failed")
                return False

        except ImportError as e:
            print(f"❌ Import failed: {e}")
            print(
                "   Make sure modules.image_processor and modules.qr_extractor are available")
            return False
        except Exception as e:
            print(f"❌ Quick test failed: {e}")
            return False


if __name__ == "__main__":
    # Run quick test by default, comprehensive test with --full flag
    import sys

    if "--full" in sys.argv:
        demo_comprehensive_testing()
    else:
        QuickImageTest.run_quick_test()
