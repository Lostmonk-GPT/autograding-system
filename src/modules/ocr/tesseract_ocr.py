#!/usr/bin/env python3
"""
AutoGrading System - Tesseract OCR Engine
Specialized Tesseract implementation for mathematical content recognition.
"""

import cv2
import numpy as np
import pytesseract
from typing import Tuple, Dict, Any, Optional
import logging
import re

# Setup logging
logger = logging.getLogger(__name__)


class TesseractOCR:
    """Tesseract OCR engine optimized for mathematical content."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize Tesseract OCR engine.

        Args:
            config: Configuration dictionary for Tesseract settings
        """
        self.config = config or self._default_config()
        self._validate_tesseract_installation()

    def _default_config(self) -> Dict[str, Any]:
        """Get default Tesseract configuration."""
        return {
            'math_config': '--psm 6 -c tessedit_char_whitelist=0123456789+-*/=().,ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz',
            'general_config': '--psm 6',
            'confidence_threshold': 30,
            'preprocessing': {
                'apply_threshold': True,
                'scale_factor': 2.0,
                'denoise': True
            },
            'math_optimization': {
                'enable_equation_mode': True,
                'preserve_spacing': True,
                'enhance_symbols': True
            }
        }

    def _validate_tesseract_installation(self):
        """Validate that Tesseract is properly installed."""
        try:
            version = pytesseract.get_tesseract_version()
            logger.info(f"Tesseract version detected: {version}")

            # Test basic functionality
            test_image = np.ones((100, 100), dtype=np.uint8) * 255
            cv2.putText(test_image, "123", (10, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, 0, 2)

            test_result = pytesseract.image_to_string(test_image)
            if "123" not in test_result:
                logger.warning("Tesseract test failed, but continuing anyway")

        except Exception as e:
            logger.error(f"Tesseract validation failed: {e}")
            raise RuntimeError(
                f"Tesseract not properly installed or configured: {e}")

    def extract_text(self, image: np.ndarray) -> Tuple[str, float]:
        """Extract text from image using Tesseract.

        Args:
            image: Input image as numpy array

        Returns:
            Tuple of (extracted_text, confidence_score)
        """
        try:
            # Preprocess image for better OCR
            processed_image = self._preprocess_for_tesseract(image)

            # Try mathematical content extraction first
            text, confidence = self._extract_mathematical_content(
                processed_image)

            # If mathematical extraction fails or gives low confidence, try general OCR
            if confidence < self.config.get('confidence_threshold', 30):
                general_text, general_confidence = self._extract_general_content(
                    processed_image)

                # Use better result
                if general_confidence > confidence:
                    text, confidence = general_text, general_confidence

            # Post-process text
            text = self._post_process_text(text)

            # Normalize confidence to 0-1 range
            normalized_confidence = min(
                confidence / 100.0, 1.0) if confidence > 0 else 0.0

            logger.debug(
                f"Tesseract extracted {len(text)} characters with {normalized_confidence:.2f} confidence")

            return text, normalized_confidence

        except Exception as e:
            logger.error(f"Tesseract extraction failed: {e}")
            return "", 0.0

    def _preprocess_for_tesseract(self, image: np.ndarray) -> np.ndarray:
        """Preprocess image specifically for Tesseract OCR."""
        processed = image.copy()

        # Convert to grayscale if needed
        if len(processed.shape) == 3:
            processed = cv2.cvtColor(processed, cv2.COLOR_RGB2GRAY)

        # Scale up for better recognition
        scale_factor = self.config.get(
            'preprocessing', {}).get('scale_factor', 2.0)
        if scale_factor > 1.0:
            new_width = int(processed.shape[1] * scale_factor)
            new_height = int(processed.shape[0] * scale_factor)
            processed = cv2.resize(
                processed, (new_width, new_height), interpolation=cv2.INTER_CUBIC)

        # Denoise if enabled
        if self.config.get('preprocessing', {}).get('denoise', True):
            processed = cv2.fastNlMeansDenoising(processed)

        # Apply threshold for better text recognition
        if self.config.get('preprocessing', {}).get('apply_threshold', True):
            # Use Otsu's thresholding
            _, processed = cv2.threshold(
                processed, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Enhance mathematical symbols if enabled
        if self.config.get('math_optimization', {}).get('enhance_symbols', True):
            processed = self._enhance_mathematical_symbols(processed)

        return processed

    def _enhance_mathematical_symbols(self, image: np.ndarray) -> np.ndarray:
        """Enhance mathematical symbols for better recognition."""
        try:
            # Apply morphological operations to clean up symbols
            kernel = np.ones((2, 2), np.uint8)

            # Close small gaps in symbols
            enhanced = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)

            # Remove small noise
            enhanced = cv2.morphologyEx(enhanced, cv2.MORPH_OPEN, kernel)

            return enhanced

        except Exception as e:
            logger.warning(f"Symbol enhancement failed: {e}")
            return image

    def _extract_mathematical_content(self, image: np.ndarray) -> Tuple[str, float]:
        """Extract mathematical content using specialized configuration."""
        try:
            config = self.config.get('math_config', '--psm 6 -c tessedit_char_whitelist=0123456789+-*/=().,ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz')

            # Get text with confidence data
            data = pytesseract.image_to_data(
                image, config=config, output_type=pytesseract.Output.DICT)

            # Filter out low-confidence detections
            confidences = [int(conf) for conf in data['conf'] if int(conf) > 0]
            texts = [data['text'][i]
                     for i, conf in enumerate(data['conf']) if int(conf) > 0]

            if not texts:
                return "", 0.0

            # Combine text with proper spacing for mathematical expressions
            if self.config.get('math_optimization', {}).get('preserve_spacing', True):
                combined_text = self._combine_math_text_with_spacing(data)
            else:
                combined_text = ' '.join(texts).strip()

            # Calculate average confidence
            avg_confidence = sum(confidences) / \
                len(confidences) if confidences else 0.0

            return combined_text, avg_confidence

        except Exception as e:
            logger.warning(f"Mathematical content extraction failed: {e}")
            return "", 0.0

    def _combine_math_text_with_spacing(self, data: Dict) -> str:
        """Combine text elements with appropriate spacing for mathematical expressions."""
        try:
            # Group text by lines based on y-coordinates
            text_elements = []

            for i, (text, conf, left, top, width, height) in enumerate(zip(
                data['text'], data['conf'], data['left'], data['top'], data['width'], data['height']
            )):
                if int(conf) > 0 and text.strip():
                    text_elements.append({
                        'text': text.strip(),
                        'conf': int(conf),
                        'left': int(left),
                        'top': int(top),
                        'right': int(left) + int(width),
                        'bottom': int(top) + int(height)
                    })

            if not text_elements:
                return ""

            # Sort by top position (y-coordinate) first, then left position (x-coordinate)
            text_elements.sort(key=lambda x: (x['top'], x['left']))

            # Group into lines
            lines = []
            current_line = []
            current_line_top = text_elements[0]['top']
            line_height_threshold = 20  # pixels

            for element in text_elements:
                # Check if this element is on the same line
                if abs(element['top'] - current_line_top) <= line_height_threshold:
                    current_line.append(element)
                else:
                    # Start new line
                    if current_line:
                        lines.append(current_line)
                    current_line = [element]
                    current_line_top = element['top']

            # Add the last line
            if current_line:
                lines.append(current_line)

            # Combine lines with appropriate spacing
            result_lines = []
            for line in lines:
                # Sort elements in line by left position
                line.sort(key=lambda x: x['left'])

                # Combine elements in line with spacing
                line_text = ""
                prev_right = 0

                for element in line:
                    # Add space if there's a gap between elements
                    # 10px gap threshold
                    if prev_right > 0 and element['left'] - prev_right > 10:
                        line_text += " "

                    line_text += element['text']
                    prev_right = element['right']

                result_lines.append(line_text.strip())

            return '\n'.join(result_lines)

        except Exception as e:
            logger.warning(f"Math text spacing failed: {e}")
            # Fallback to simple combination
            return ' '.join([text for text in data['text'] if text.strip()])

    def _extract_general_content(self, image: np.ndarray) -> Tuple[str, float]:
        """Extract general text content using standard configuration."""
        try:
            config = self.config.get('general_config', '--psm 6')

            # Get text with confidence
            data = pytesseract.image_to_data(
                image, config=config, output_type=pytesseract.Output.DICT)

            # Filter and combine text
            confidences = [int(conf) for conf in data['conf'] if int(conf) > 0]
            texts = [data['text'][i]
                     for i, conf in enumerate(data['conf']) if int(conf) > 0]

            if not texts:
                return "", 0.0

            combined_text = ' '.join(texts).strip()
            avg_confidence = sum(confidences) / \
                len(confidences) if confidences else 0.0

            return combined_text, avg_confidence

        except Exception as e:
            logger.warning(f"General content extraction failed: {e}")
            return "", 0.0

    def _post_process_text(self, text: str) -> str:
        """Post-process extracted text to clean up common issues."""
        if not text:
            return text

        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()

        # Fix common mathematical notation issues
        text = self._fix_mathematical_notation(text)

        return text

    def _fix_mathematical_notation(self, text: str) -> str:
        """Fix common mathematical notation issues in OCR output."""
        # Common OCR corrections for mathematical content
        corrections = [
            # Fraction bars
            (r'(\d+)\s*/\s*(\d+)', r'\1/\2'),
            # Equals signs
            (r'\s*=\s*', ' = '),
            # Plus/minus signs
            (r'\s*\+\s*', ' + '),
            (r'\s*-\s*', ' - '),
            # Multiplication
            (r'\s*\*\s*', ' * '),
            (r'(\d)\s*x\s*(\d)', r'\1 * \2'),  # x as multiplication
            # Parentheses spacing
            (r'\s*\(\s*', '('),
            (r'\s*\)\s*', ') '),
            # Remove double spaces
            (r'\s+', ' ')
        ]

        for pattern, replacement in corrections:
            text = re.sub(pattern, replacement, text)

        return text.strip()

    def get_supported_languages(self) -> list:
        """Get list of supported languages."""
        try:
            return pytesseract.get_languages(config='')
        except Exception as e:
            logger.warning(f"Could not get supported languages: {e}")
            return ['eng']  # Default to English

    def test_with_sample_text(self) -> Dict[str, Any]:
        """Test Tesseract with a sample mathematical expression."""
        try:
            # Create test image with mathematical content
            test_image = np.ones((200, 400), dtype=np.uint8) * 255

            # Add mathematical text
            cv2.putText(test_image, "2x + 3 = 7", (20, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, 0, 2)
            cv2.putText(test_image, "x = (7-3)/2", (20, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, 0, 2)
            cv2.putText(test_image, "x = 2", (20, 150),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, 0, 2)

            # Extract text
            text, confidence = self.extract_text(test_image)

            # Check if expected content is found
            expected_elements = ["2x", "+", "3", "=", "7", "x", "2"]
            found_elements = sum(
                1 for element in expected_elements if element in text)

            return {
                'success': True,
                'extracted_text': text,
                'confidence': confidence,
                'expected_elements': len(expected_elements),
                'found_elements': found_elements,
                'recognition_rate': found_elements / len(expected_elements),
                # 70% threshold
                'test_passed': found_elements >= len(expected_elements) * 0.7
            }

        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'extracted_text': '',
                'confidence': 0.0,
                'test_passed': False
            }


def demo_tesseract_ocr():
    """Demo function to test Tesseract OCR capabilities."""
    print("📝 Tesseract OCR Demo")
    print("=" * 30)

    try:
        # Create Tesseract engine
        tesseract = TesseractOCR()

        print("✅ Tesseract OCR engine initialized")

        # Check supported languages
        languages = tesseract.get_supported_languages()
        print(
            f"✅ Supported languages: {', '.join(languages[:5])}{'...' if len(languages) > 5 else ''}")

        # Test with sample mathematical content
        print("\n🧪 Testing with sample mathematical content...")

        test_result = tesseract.test_with_sample_text()

        if test_result['success']:
            print(f"✅ Sample test completed")
            print(f"   Extracted: '{test_result['extracted_text']}'")
            print(f"   Confidence: {test_result['confidence']:.2f}")
            print(
                f"   Recognition rate: {test_result['recognition_rate']:.1%}")
            print(
                f"   Test passed: {'✅' if test_result['test_passed'] else '❌'}")
        else:
            print(f"❌ Sample test failed: {test_result['error']}")

        # Test with different image types
        print("\n🖼️  Testing with different image preprocessing...")

        # Create test image with noise
        noisy_image = np.ones((100, 300), dtype=np.uint8) * 255
        cv2.putText(noisy_image, "x^2 + 5x = 14", (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, 0, 2)

        # Add noise
        noise = np.random.randint(0, 50, noisy_image.shape, dtype=np.uint8)
        noisy_image = cv2.subtract(noisy_image, noise)

        text, confidence = tesseract.extract_text(noisy_image)
        print(f"✅ Noisy image: '{text}' (confidence: {confidence:.2f})")

        print("\n🎉 Tesseract OCR demo completed!")
        return True

    except Exception as e:
        print(f"❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    demo_tesseract_ocr()
