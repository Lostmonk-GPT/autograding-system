#!/usr/bin/env python3
"""
AutoGrading System - EasyOCR Engine
EasyOCR implementation optimized for handwritten mathematical content.
"""

import cv2
import numpy as np
from typing import Tuple, Dict, Any, Optional, List
import logging
import re

# Setup logging
logger = logging.getLogger(__name__)


class EasyOCREngine:
    """EasyOCR engine optimized for handwritten mathematical content."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize EasyOCR engine.

        Args:
            config: Configuration dictionary for EasyOCR settings
        """
        self.config = config or self._default_config()
        self.reader = None
        self._init_reader()

    def _default_config(self) -> Dict[str, Any]:
        """Get default EasyOCR configuration."""
        return {
            'languages': ['en'],
            'gpu_enabled': False,  # Set to True if CUDA available
            'confidence_threshold': 0.3,
            'text_threshold': 0.7,
            'link_threshold': 0.4,
            'canvas_size': 2560,
            'mag_ratio': 1.5,
            'preprocessing': {
                'enhance_contrast': True,
                'denoise': True,
                'resize_factor': 1.5
            },
            'handwriting_optimization': {
                'enable_handwriting_mode': True,
                'character_spacing_threshold': 30,
                'line_spacing_threshold': 50
            }
        }

    def _init_reader(self):
        """Initialize EasyOCR reader."""
        try:
            import easyocr

            # Initialize with configuration
            self.reader = easyocr.Reader(
                self.config['languages'],
                gpu=self.config['gpu_enabled']
            )

            logger.info(
                f"EasyOCR initialized with languages: {self.config['languages']}")

            # Test basic functionality
            test_image = np.ones((100, 200), dtype=np.uint8) * 255
            cv2.putText(test_image, "Test", (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, 0, 2)

            try:
                results = self.reader.readtext(test_image)
                logger.debug(
                    f"EasyOCR test successful: {len(results)} detections")
            except Exception as e:
                logger.warning(f"EasyOCR test failed: {e}")

        except ImportError:
            raise ImportError(
                "EasyOCR not installed. Install with: pip install easyocr")
        except Exception as e:
            raise RuntimeError(f"EasyOCR initialization failed: {e}")

    def extract_text(self, image: np.ndarray) -> Tuple[str, float]:
        """Extract text from image using EasyOCR.

        Args:
            image: Input image as numpy array

        Returns:
            Tuple of (extracted_text, confidence_score)
        """
        try:
            # Preprocess image for better handwriting recognition
            processed_image = self._preprocess_for_easyocr(image)

            # Extract text with EasyOCR
            results = self.reader.readtext(
                processed_image,
                text_threshold=self.config['text_threshold'],
                link_threshold=self.config['link_threshold'],
                canvas_size=self.config['canvas_size'],
                mag_ratio=self.config['mag_ratio']
            )

            # Process results
            text, confidence = self._process_easyocr_results(
                results, processed_image.shape)

            # Post-process for mathematical content
            text = self._post_process_mathematical_text(text)

            logger.debug(
                f"EasyOCR extracted {len(text)} characters with {confidence:.2f} confidence")

            return text, confidence

        except Exception as e:
            logger.error(f"EasyOCR extraction failed: {e}")
            return "", 0.0

    def _preprocess_for_easyocr(self, image: np.ndarray) -> np.ndarray:
        """Preprocess image specifically for EasyOCR handwriting recognition."""
        processed = image.copy()

        # Convert to grayscale if needed
        if len(processed.shape) == 3:
            processed = cv2.cvtColor(processed, cv2.COLOR_RGB2GRAY)

        # Resize for better recognition if needed
        if self.config['preprocessing']['resize_factor'] != 1.0:
            scale = self.config['preprocessing']['resize_factor']
            new_width = int(processed.shape[1] * scale)
            new_height = int(processed.shape[0] * scale)
            processed = cv2.resize(
                processed, (new_width, new_height), interpolation=cv2.INTER_CUBIC)

        # Enhance contrast for handwritten text
        if self.config['preprocessing']['enhance_contrast']:
            processed = self._enhance_contrast_for_handwriting(processed)

        # Denoise while preserving handwriting characteristics
        if self.config['preprocessing']['denoise']:
            processed = self._denoise_handwriting(processed)

        # Convert back to 3-channel for EasyOCR (it expects RGB)
        if len(processed.shape) == 2:
            processed = cv2.cvtColor(processed, cv2.COLOR_GRAY2RGB)

        return processed

    def _enhance_contrast_for_handwriting(self, image: np.ndarray) -> np.ndarray:
        """Enhance contrast specifically for handwritten content."""
        try:
            # Use adaptive histogram equalization
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
            enhanced = clahe.apply(image)

            # Apply gentle sharpening for handwriting
            kernel = np.array([[-0.5, -1, -0.5],
                               [-1, 7, -1],
                               [-0.5, -1, -0.5]])
            sharpened = cv2.filter2D(enhanced, -1, kernel)

            return sharpened

        except Exception as e:
            logger.warning(f"Contrast enhancement failed: {e}")
            return image

    def _denoise_handwriting(self, image: np.ndarray) -> np.ndarray:
        """Denoise while preserving handwriting characteristics."""
        try:
            # Use bilateral filter to preserve edges (handwriting strokes)
            denoised = cv2.bilateralFilter(image, 9, 75, 75)

            # Alternative: use Non-local Means denoising with lower strength
            # to preserve handwriting details
            # denoised = cv2.fastNlMeansDenoising(image, None, 10, 7, 21)

            return denoised

        except Exception as e:
            logger.warning(f"Handwriting denoising failed: {e}")
            return image

    def _process_easyocr_results(self, results: List, image_shape: Tuple[int, int]) -> Tuple[str, float]:
        """Process EasyOCR results into combined text and confidence."""
        if not results:
            return "", 0.0

        # Filter results by confidence threshold
        filtered_results = [
            result for result in results
            if len(result) >= 3 and result[2] >= self.config['confidence_threshold']
        ]

        if not filtered_results:
            return "", 0.0

        # Sort results by position for proper text order
        if self.config['handwriting_optimization']['enable_handwriting_mode']:
            combined_text = self._combine_handwriting_text(
                filtered_results, image_shape)
        else:
            combined_text = self._combine_text_simple(filtered_results)

        # Calculate average confidence
        confidences = [result[2] for result in filtered_results]
        avg_confidence = sum(confidences) / len(confidences)

        return combined_text, avg_confidence

    def _combine_handwriting_text(self, results: List, image_shape: Tuple[int, int]) -> str:
        """Combine text results with proper spacing for handwritten content."""
        try:
            # Extract text elements with their bounding boxes
            text_elements = []

            for result in results:
                bbox, text, confidence = result

                # Calculate center point of bounding box
                x_coords = [point[0] for point in bbox]
                y_coords = [point[1] for point in bbox]
                center_x = sum(x_coords) / len(x_coords)
                center_y = sum(y_coords) / len(y_coords)

                # Calculate bounding box dimensions
                width = max(x_coords) - min(x_coords)
                height = max(y_coords) - min(y_coords)

                text_elements.append({
                    'text': text.strip(),
                    'center_x': center_x,
                    'center_y': center_y,
                    'top': min(y_coords),
                    'left': min(x_coords),
                    'width': width,
                    'height': height,
                    'confidence': confidence
                })

            if not text_elements:
                return ""

            # Group elements into lines based on y-coordinates
            line_threshold = self.config['handwriting_optimization']['line_spacing_threshold']
            lines = self._group_into_lines(text_elements, line_threshold)

            # Combine lines with appropriate spacing
            result_lines = []
            for line in lines:
                # Sort elements in line by x-coordinate
                line.sort(key=lambda x: x['center_x'])

                # Combine elements with spacing
                line_text = self._combine_line_elements(line)
                if line_text.strip():
                    result_lines.append(line_text.strip())

            return '\n'.join(result_lines)

        except Exception as e:
            logger.warning(f"Handwriting text combination failed: {e}")
            return self._combine_text_simple(results)

    def _group_into_lines(self, elements: List[Dict], line_threshold: int) -> List[List[Dict]]:
        """Group text elements into lines based on y-coordinates."""
        if not elements:
            return []

        # Sort by y-coordinate
        sorted_elements = sorted(elements, key=lambda x: x['center_y'])

        lines = []
        current_line = [sorted_elements[0]]
        current_line_y = sorted_elements[0]['center_y']

        for element in sorted_elements[1:]:
            # Check if element belongs to current line
            y_distance = abs(element['center_y'] - current_line_y)

            if y_distance <= line_threshold:
                current_line.append(element)
                # Update line center (weighted average)
                total_width = sum(e['width'] for e in current_line)
                weighted_y = sum(e['center_y'] * e['width']
                                 for e in current_line) / total_width
                current_line_y = weighted_y
            else:
                # Start new line
                lines.append(current_line)
                current_line = [element]
                current_line_y = element['center_y']

        # Add the last line
        if current_line:
            lines.append(current_line)

        return lines

    def _combine_line_elements(self, line_elements: List[Dict]) -> str:
        """Combine elements within a line with appropriate spacing."""
        if not line_elements:
            return ""

        if len(line_elements) == 1:
            return line_elements[0]['text']

        result = ""
        char_threshold = self.config['handwriting_optimization']['character_spacing_threshold']

        for i, element in enumerate(line_elements):
            result += element['text']

            # Add space if there's a significant gap to next element
            if i < len(line_elements) - 1:
                next_element = line_elements[i + 1]
                gap = next_element['left'] - \
                    (element['left'] + element['width'])

                if gap > char_threshold:
                    result += " "

        return result

    def _combine_text_simple(self, results: List) -> str:
        """Simple text combination for non-handwriting mode."""
        texts = [result[1].strip() for result in results if len(result) >= 2]
        return ' '.join(texts)

    def _post_process_mathematical_text(self, text: str) -> str:
        """Post-process text for mathematical content."""
        if not text:
            return text

        # Clean up whitespace
        text = re.sub(r'\s+', ' ', text).strip()

        # Fix common mathematical notation issues
        text = self._fix_mathematical_symbols(text)

        return text

    def _fix_mathematical_symbols(self, text: str) -> str:
        """Fix common mathematical symbol recognition issues."""
        # Common EasyOCR corrections for mathematical content
        corrections = [
            # Common misrecognitions
            ('O', '0'),  # Letter O to zero (in context)
            ('l', '1'),  # Lowercase L to one (in context)
            ('I', '1'),  # Uppercase I to one (in context)
            ('S', '5'),  # S to 5 (in context)
            ('Z', '2'),  # Z to 2 (in context)

            # Mathematical operators
            ('×', '*'),  # Multiplication symbol
            ('÷', '/'),  # Division symbol
            ('−', '-'),  # Minus sign
            ('±', '±'),  # Plus-minus (keep as is)

            # Fraction notation
            (r'(\d+)\s*/\s*(\d+)', r'\1/\2'),

            # Equation spacing
            (r'\s*=\s*', ' = '),
            (r'\s*\+\s*', ' + '),
            (r'\s*-\s*', ' - '),
            (r'\s*\*\s*', ' * '),

            # Parentheses
            (r'\s*\(\s*', '('),
            (r'\s*\)\s*', ') '),
        ]

        # Apply context-aware corrections
        for old, new in corrections:
            if old in ['O', 'l', 'I', 'S', 'Z']:
                # Only replace if surrounded by digits or operators
                pattern = rf'(?<=[0-9+\-*/=\s]){old}(?=[0-9+\-*/=\s])'
                text = re.sub(pattern, new, text)
            else:
                text = re.sub(old, new, text)

        # Final cleanup
        text = re.sub(r'\s+', ' ', text).strip()

        return text

    def test_with_sample_handwriting(self) -> Dict[str, Any]:
        """Test EasyOCR with simulated handwritten mathematical content."""
        try:
            # Create test image simulating handwritten math
            test_image = np.ones((150, 300), dtype=np.uint8) * 255

            # Simulate handwritten style with thicker, slightly irregular text
            font = cv2.FONT_HERSHEY_SCRIPT_SIMPLEX

            cv2.putText(test_image, "2x + 5 = 11", (20, 50), font, 0.8, 0, 3)
            cv2.putText(test_image, "x = 3", (20, 100), font, 0.8, 0, 3)

            # Add slight noise to simulate handwriting imperfections
            noise = np.random.randint(-20, 20,
                                      test_image.shape, dtype=np.int16)
            test_image = np.clip(test_image.astype(
                np.int16) + noise, 0, 255).astype(np.uint8)

            # Extract text
            text, confidence = self.extract_text(test_image)

            # Check for expected mathematical elements
            expected_elements = ["2x", "5", "11", "x", "3", "+", "="]
            found_elements = sum(
                1 for element in expected_elements if element in text.replace(" ", ""))

            return {
                'success': True,
                'extracted_text': text,
                'confidence': confidence,
                'expected_elements': len(expected_elements),
                'found_elements': found_elements,
                'recognition_rate': found_elements / len(expected_elements),
                # 60% threshold for handwriting
                'test_passed': found_elements >= len(expected_elements) * 0.6
            }

        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'extracted_text': '',
                'confidence': 0.0,
                'test_passed': False
            }

    def get_supported_languages(self) -> List[str]:
        """Get list of supported languages."""
        return self.config['languages']


def demo_easyocr_engine():
    """Demo function to test EasyOCR engine capabilities."""
    print("✍️  EasyOCR Engine Demo")
    print("=" * 30)

    try:
        # Create EasyOCR engine
        easyocr_engine = EasyOCREngine()

        print("✅ EasyOCR engine initialized")
        print(
            f"   Languages: {', '.join(easyocr_engine.get_supported_languages())}")
        print(f"   GPU enabled: {easyocr_engine.config['gpu_enabled']}")

        # Test with sample handwritten content
        print("\n🧪 Testing with simulated handwritten mathematical content...")

        test_result = easyocr_engine.test_with_sample_handwriting()

        if test_result['success']:
            print(f"✅ Handwriting test completed")
            print(f"   Extracted: '{test_result['extracted_text']}'")
            print(f"   Confidence: {test_result['confidence']:.2f}")
            print(
                f"   Recognition rate: {test_result['recognition_rate']:.1%}")
            print(
                f"   Test passed: {'✅' if test_result['test_passed'] else '❌'}")
        else:
            print(f"❌ Handwriting test failed: {test_result['error']}")

        # Test with printed mathematical content
        print("\n📝 Testing with printed mathematical content...")

        # Create printed math test
        printed_image = np.ones((100, 250), dtype=np.uint8) * 255
        cv2.putText(printed_image, "y = mx + b", (20, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, 0, 2)

        text, confidence = easyocr_engine.extract_text(printed_image)
        print(f"✅ Printed text: '{text}' (confidence: {confidence:.2f})")

        # Test configuration options
        print(f"\n⚙️  Configuration:")
        print(
            f"   Confidence threshold: {easyocr_engine.config['confidence_threshold']}")
        print(f"   Text threshold: {easyocr_engine.config['text_threshold']}")
        print(
            f"   Handwriting mode: {easyocr_engine.config['handwriting_optimization']['enable_handwriting_mode']}")
        print(
            f"   Character spacing: {easyocr_engine.config['handwriting_optimization']['character_spacing_threshold']}px")

        print("\n🎉 EasyOCR engine demo completed!")
        return True

    except Exception as e:
        print(f"❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    demo_easyocr_engine()
