#!/usr/bin/env python3
"""
AutoGrading System - Advanced Image Preprocessing Pipeline
Comprehensive preprocessing to fix image quality issues and optimize for OCR.
"""

import cv2
import numpy as np
from typing import Tuple, Dict, Any, Optional
import logging
from pathlib import Path

# Setup logging
logger = logging.getLogger(__name__)


class AdvancedImageProcessor:
    """Advanced image preprocessing pipeline for OCR optimization."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize advanced image processor.

        Args:
            config: Configuration dictionary for preprocessing parameters
        """
        self.config = config or self._default_config()

    def _default_config(self) -> Dict[str, Any]:
        """Get default configuration for advanced preprocessing."""
        return {
            'quality_analysis': {
                'brightness_optimal_range': (80, 200),
                'contrast_minimum': 30,
                'sharpness_minimum': 50,
                'noise_threshold': 15
            },
            'enhancement': {
                'auto_brightness_contrast': True,
                'auto_sharpening': True,
                'noise_reduction': True,
                'contrast_enhancement': True,
                'adaptive_processing': True
            },
            'mathematical_optimization': {
                'enhance_text_regions': True,
                'improve_symbol_clarity': True,
                'optimize_line_thickness': True,
                'preserve_spacing': True
            },
            'output_optimization': {
                'target_dpi': 300,
                'optimal_width_range': (1200, 2400),
                'ensure_binary_clarity': True
            }
        }

    def process_for_ocr(self, image: np.ndarray, image_path: str = None) -> Dict[str, Any]:
        """Process image with advanced pipeline for optimal OCR results.

        Args:
            image: Input image as numpy array
            image_path: Optional path for logging

        Returns:
            Dictionary containing processed image and analysis data
        """
        try:
            # Step 1: Analyze image quality
            quality_analysis = self._analyze_image_quality(image)
            logger.info(
                f"Image quality analysis: {quality_analysis['overall_score']}/100")

            # Step 2: Create processing plan based on quality issues
            processing_plan = self._create_processing_plan(quality_analysis)

            # Step 3: Apply targeted enhancements
            enhanced_image = self._apply_enhancements(image, processing_plan)

            # Step 4: Mathematical content optimization
            if self.config['mathematical_optimization']['enhance_text_regions']:
                enhanced_image = self._optimize_for_mathematical_content(
                    enhanced_image)

            # Step 5: Final OCR optimization
            ocr_optimized = self._final_ocr_optimization(enhanced_image)

            # Step 6: Quality verification
            final_quality = self._analyze_image_quality(ocr_optimized)

            improvement_score = final_quality['overall_score'] - \
                quality_analysis['overall_score']

            return {
                'success': True,
                'original_image': image,
                'processed_image': ocr_optimized,
                'original_quality': quality_analysis,
                'final_quality': final_quality,
                'improvement_score': improvement_score,
                'processing_steps': processing_plan['steps_applied'],
                'recommended_for_ocr': final_quality['overall_score'] >= 70,
                'processing_metadata': {
                    'source_path': image_path,
                    'enhancement_applied': processing_plan['enhancements_needed'],
                    'mathematical_optimization': True
                }
            }

        except Exception as e:
            logger.error(f"Advanced preprocessing failed: {e}")
            return {
                'success': False,
                'original_image': image,
                'processed_image': image,  # Fallback to original
                'error_message': str(e)
            }

    def _analyze_image_quality(self, image: np.ndarray) -> Dict[str, Any]:
        """Comprehensive image quality analysis."""
        try:
            # Convert to grayscale for analysis
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            else:
                gray = image

            height, width = gray.shape

            # Brightness analysis
            brightness = float(np.mean(gray))
            brightness_optimal = self.config['quality_analysis']['brightness_optimal_range']
            brightness_score = 100.0
            if brightness < brightness_optimal[0] or brightness > brightness_optimal[1]:
                # Calculate how far from optimal range
                if brightness < brightness_optimal[0]:
                    brightness_score = (
                        brightness / brightness_optimal[0]) * 100
                else:
                    brightness_score = max(
                        0, 100 - (brightness - brightness_optimal[1]))

            # Contrast analysis
            contrast = float(np.std(gray))
            contrast_min = self.config['quality_analysis']['contrast_minimum']
            contrast_score = min(100, (contrast / contrast_min)
                                 * 100) if contrast_min > 0 else 100

            # Sharpness analysis (Laplacian variance)
            laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
            sharpness_min = self.config['quality_analysis']['sharpness_minimum']
            sharpness_score = min(
                100, (laplacian_var / sharpness_min) * 100) if sharpness_min > 0 else 100

            # Noise analysis
            # Use median filter to estimate noise
            median_filtered = cv2.medianBlur(gray, 5)
            noise_level = float(
                np.mean(np.abs(gray.astype(np.float32) - median_filtered.astype(np.float32))))
            noise_threshold = self.config['quality_analysis']['noise_threshold']
            noise_score = max(0, 100 - (noise_level / noise_threshold) * 100)

            # Resolution analysis
            total_pixels = height * width
            if total_pixels < 300000:  # Less than ~550x550
                resolution_score = 30
            elif total_pixels < 1000000:  # Less than ~1000x1000
                resolution_score = 60
            else:
                resolution_score = 100

            # Text clarity analysis (edge density)
            edges = cv2.Canny(gray, 50, 150)
            edge_density = np.sum(edges > 0) / total_pixels
            # Optimal edge density for text is around 0.1-0.3
            if 0.05 <= edge_density <= 0.4:
                text_clarity_score = 100
            else:
                text_clarity_score = max(
                    0, 100 - abs(edge_density - 0.2) * 200)

            # Calculate overall score
            scores = [brightness_score, contrast_score, sharpness_score,
                     noise_score, resolution_score, text_clarity_score]
            overall_score = sum(scores) / len(scores)

            return {
                'brightness': brightness,
                'brightness_score': brightness_score,
                'brightness_optimal': brightness_optimal[0] <= brightness <= brightness_optimal[1],
                'contrast': contrast,
                'contrast_score': contrast_score,
                'contrast_adequate': contrast >= contrast_min,
                'sharpness': laplacian_var,
                'sharpness_score': sharpness_score,
                'sharpness_adequate': laplacian_var >= sharpness_min,
                'noise_level': noise_level,
                'noise_score': noise_score,
                'noise_acceptable': noise_level <= noise_threshold,
                'resolution': (width, height),
                'resolution_score': resolution_score,
                'text_clarity': edge_density,
                'text_clarity_score': text_clarity_score,
                'overall_score': overall_score,
                'needs_enhancement': overall_score < 70
            }

        except Exception as e:
            logger.error(f"Quality analysis failed: {e}")
            return {'overall_score': 0, 'needs_enhancement': True, 'error': str(e)}

    def _create_processing_plan(self, quality_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Create targeted processing plan based on quality analysis."""
        plan = {
            'enhancements_needed': [],
            'steps_applied': [],
            'priority_order': []
        }

        # Brightness correction
        if not quality_analysis.get('brightness_optimal', True):
            brightness = quality_analysis.get('brightness', 128)
            if brightness < 80:
                plan['enhancements_needed'].append('brighten')
                plan['priority_order'].append(
                    ('brightness_correction', 'high'))
            elif brightness > 200:
                plan['enhancements_needed'].append('darken')
                plan['priority_order'].append(
                    ('brightness_correction', 'high'))

        # Contrast enhancement
        if not quality_analysis.get('contrast_adequate', True):
            plan['enhancements_needed'].append('contrast_enhancement')
            plan['priority_order'].append(('contrast_enhancement', 'high'))

        # Sharpness improvement
        if not quality_analysis.get('sharpness_adequate', True):
            plan['enhancements_needed'].append('sharpening')
            plan['priority_order'].append(('sharpening', 'medium'))

        # Noise reduction
        if not quality_analysis.get('noise_acceptable', True):
            plan['enhancements_needed'].append('denoising')
            plan['priority_order'].append(('denoising', 'medium'))

        # Resolution enhancement
        if quality_analysis.get('resolution_score', 100) < 60:
            plan['enhancements_needed'].append('upscaling')
            plan['priority_order'].append(('upscaling', 'low'))

        # Sort by priority
        priority_map = {'high': 1, 'medium': 2, 'low': 3}
        plan['priority_order'].sort(key=lambda x: priority_map[x[1]])

        return plan

    def _apply_enhancements(self, image: np.ndarray, processing_plan: Dict[str, Any]) -> np.ndarray:
        """Apply targeted enhancements based on processing plan."""
        enhanced = image.copy()

        for step, priority in processing_plan['priority_order']:
            try:
                if step == 'brightness_correction':
                    enhanced = self._adjust_brightness_contrast(enhanced)
                    processing_plan['steps_applied'].append(
                        'brightness_correction')

                elif step == 'contrast_enhancement':
                    enhanced = self._enhance_contrast_advanced(enhanced)
                    processing_plan['steps_applied'].append(
                        'contrast_enhancement')

                elif step == 'sharpening':
                    enhanced = self._apply_adaptive_sharpening(enhanced)
                    processing_plan['steps_applied'].append('sharpening')

                elif step == 'denoising':
                    enhanced = self._advanced_denoising(enhanced)
                    processing_plan['steps_applied'].append('denoising')

                elif step == 'upscaling':
                    enhanced = self._intelligent_upscaling(enhanced)
                    processing_plan['steps_applied'].append('upscaling')

            except Exception as e:
                logger.warning(f"Enhancement step {step} failed: {e}")

        return enhanced

    def _adjust_brightness_contrast(self, image: np.ndarray) -> np.ndarray:
        """Automatically adjust brightness and contrast to optimal levels."""
        try:
            # Convert to grayscale for analysis
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
                is_color = True
            else:
                gray = image
                is_color = False

            # Calculate current brightness and contrast
            current_brightness = np.mean(gray)
            current_contrast = np.std(gray)

            # Target values
            target_brightness = 128  # Mid-range
            target_contrast = 50     # Good contrast for text

            # Calculate adjustment factors
            brightness_adjustment = target_brightness - current_brightness

            # Apply brightness adjustment
            if abs(brightness_adjustment) > 10:  # Only adjust if significant difference
                if is_color:
                    # Adjust in LAB color space for better results
                    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
                    l_channel = lab[:, :, 0].astype(np.float32)
                    l_channel = np.clip(
                        l_channel + brightness_adjustment, 0, 255)
                    lab[:, :, 0] = l_channel.astype(np.uint8)
                    adjusted = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
                else:
                    adjusted = np.clip(image.astype(
                        np.float32) + brightness_adjustment, 0, 255).astype(np.uint8)
            else:
                adjusted = image.copy()

            # Apply contrast adjustment if needed
            if current_contrast < target_contrast:
                contrast_factor = min(
                    2.0, target_contrast / max(current_contrast, 1))

                if is_color:
                    # Apply contrast in LAB space
                    lab = cv2.cvtColor(adjusted, cv2.COLOR_RGB2LAB)
                    l_channel = lab[:, :, 0].astype(np.float32)
                    l_channel = np.clip(
                        (l_channel - 128) * contrast_factor + 128, 0, 255)
                    lab[:, :, 0] = l_channel.astype(np.uint8)
                    adjusted = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
                else:
                    adjusted = np.clip((adjusted.astype(
                        np.float32) - 128) * contrast_factor + 128, 0, 255).astype(np.uint8)

            return adjusted

        except Exception as e:
            logger.warning(f"Brightness/contrast adjustment failed: {e}")
            return image

    def _enhance_contrast_advanced(self, image: np.ndarray) -> np.ndarray:
        """Advanced contrast enhancement using CLAHE and histogram equalization."""
        try:
            if len(image.shape) == 3:
                # Apply CLAHE to each channel in LAB color space
                lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)

                # Apply CLAHE to L channel
                clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
                lab[:, :, 0] = clahe.apply(lab[:, :, 0])

                enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
            else:
                # Apply CLAHE to grayscale image
                clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
                enhanced = clahe.apply(image)

            return enhanced

        except Exception as e:
            logger.warning(f"Contrast enhancement failed: {e}")
            return image

    def _apply_adaptive_sharpening(self, image: np.ndarray) -> np.ndarray:
        """Apply adaptive sharpening based on image content."""
        try:
            # Convert to grayscale for analysis
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            else:
                gray = image

            # Detect edges to guide sharpening intensity
            edges = cv2.Canny(gray, 50, 150)
            edge_density = np.sum(edges > 0) / (gray.shape[0] * gray.shape[1])

            # Adjust sharpening kernel based on edge density
            if edge_density < 0.05:  # Very few edges, strong sharpening
                kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
            elif edge_density < 0.15:  # Moderate edges, medium sharpening
                kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
            else:  # Many edges, gentle sharpening
                kernel = np.array(
                    [[0, -0.5, 0], [-0.5, 3, -0.5], [0, -0.5, 0]])

            # Apply sharpening
            if len(image.shape) == 3:
                sharpened = np.zeros_like(image)
                for i in range(3):
                    sharpened[:, :, i] = cv2.filter2D(
                        image[:, :, i], -1, kernel)
                sharpened = np.clip(sharpened, 0, 255).astype(np.uint8)
            else:
                sharpened = cv2.filter2D(image, -1, kernel)
                sharpened = np.clip(sharpened, 0, 255).astype(np.uint8)

            return sharpened

        except Exception as e:
            logger.warning(f"Adaptive sharpening failed: {e}")
            return image

    def _advanced_denoising(self, image: np.ndarray) -> np.ndarray:
        """Advanced noise reduction while preserving text quality."""
        try:
            if len(image.shape) == 3:
                # Use Non-local Means Denoising for color images
                denoised = cv2.fastNlMeansDenoisingColored(
                    image, None, 10, 10, 7, 21)
            else:
                # Use Non-local Means Denoising for grayscale
                denoised = cv2.fastNlMeansDenoising(image, None, 10, 7, 21)

            return denoised

        except Exception as e:
            logger.warning(f"Advanced denoising failed: {e}")
            return image

    def _intelligent_upscaling(self, image: np.ndarray) -> np.ndarray:
        """Intelligent upscaling for low-resolution images."""
        try:
            height, width = image.shape[:2]

            # Only upscale if image is small
            if width < 1200 or height < 800:
                # Calculate optimal scale factor
                target_width = 1600
                scale_factor = min(3.0, target_width / width)

                new_width = int(width * scale_factor)
                new_height = int(height * scale_factor)

                # Use INTER_CUBIC for better quality
                upscaled = cv2.resize(
                    image, (new_width, new_height), interpolation=cv2.INTER_CUBIC)

                return upscaled

            return image

        except Exception as e:
            logger.warning(f"Intelligent upscaling failed: {e}")
            return image

    def _optimize_for_mathematical_content(self, image: np.ndarray) -> np.ndarray:
        """Specialized optimization for mathematical content."""
        try:
            # Convert to grayscale for text analysis
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            else:
                gray = image.copy()

            # Enhance mathematical symbols
            if self.config['mathematical_optimization']['improve_symbol_clarity']:
                # Morphological operations to improve symbol clarity
                kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))

                # Close small gaps in symbols
                gray = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)

                # Remove small noise while preserving symbol structure
                gray = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel)

            # Optimize line thickness for better OCR
            if self.config['mathematical_optimization']['optimize_line_thickness']:
                # Apply adaptive threshold to get binary image
                binary = cv2.adaptiveThreshold(
                    gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

                # Slightly dilate text to ensure good thickness
                kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1))
                binary = cv2.morphologyEx(
                    binary, cv2.MORPH_DILATE, kernel, iterations=1)

                # Convert back if original was color
                if len(image.shape) == 3:
                    optimized = cv2.cvtColor(binary, cv2.COLOR_GRAY2RGB)
                else:
                    optimized = binary
            else:
                if len(image.shape) == 3:
                    optimized = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
                else:
                    optimized = gray

            return optimized

        except Exception as e:
            logger.warning(f"Mathematical content optimization failed: {e}")
            return image

    def _final_ocr_optimization(self, image: np.ndarray) -> np.ndarray:
        """Final optimization specifically for OCR engines."""
        try:
            # Ensure image is in optimal format for OCR
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            else:
                gray = image

            # Apply final threshold for clean binary output
            if self.config['output_optimization']['ensure_binary_clarity']:
                # Use Otsu's thresholding for optimal binary separation
                _, binary = cv2.threshold(
                    gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

                # Ensure adequate DPI
                target_dpi = self.config['output_optimization']['target_dpi']
                height, width = binary.shape

                # If image is too small for target DPI, upscale
                optimal_range = self.config['output_optimization']['optimal_width_range']
                if width < optimal_range[0]:
                    scale_factor = optimal_range[0] / width
                    new_width = int(width * scale_factor)
                    new_height = int(height * scale_factor)
                    binary = cv2.resize(
                        binary, (new_width, new_height), interpolation=cv2.INTER_CUBIC)

                final_image = binary
            else:
                final_image = gray

            # Convert back to RGB if needed (some OCR engines prefer RGB)
            if len(image.shape) == 3:
                final_image = cv2.cvtColor(final_image, cv2.COLOR_GRAY2RGB)

            return final_image

        except Exception as e:
            logger.warning(f"Final OCR optimization failed: {e}")
            return image


def demo_advanced_preprocessing():
    """Demo function to test advanced preprocessing capabilities."""
    print("🖼️  Advanced Image Preprocessing Demo")
    print("=" * 45)

    try:
        # Create advanced processor
        processor = AdvancedImageProcessor()

        print("✅ Advanced image processor initialized")

        # Test with problematic image (like your test case)
        print("\n🧪 Testing with poor quality image...")

        # Create test image with quality issues (simulating your real test case)
        poor_image = np.ones((400, 600), dtype=np.uint8) * 250  # Too bright

        # Add mathematical content
        cv2.putText(poor_image, "2x + 3 = 7", (50, 100), cv2.FONT_HERSHEY_
