#!/usr/bin/env python3
"""
AutoGrading System - Image Processing Module
Handles preprocessing of scanned assignment images for better OCR results.
"""

import cv2
import numpy as np
from PIL import Image, ImageEnhance
from pathlib import Path
import logging
from typing import Tuple, Optional, Dict, Any
import io

# Setup logging
logger = logging.getLogger(__name__)


class ImageProcessor:
    """Handles image preprocessing for scanned assignments."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize image processor with configuration.
        
        Args:
            config: Configuration dictionary for processing parameters
        """
        self.config = config or self._default_config()
        
    def _default_config(self) -> Dict[str, Any]:
        """Get default configuration settings."""
        return {
            'max_file_size_mb': 50,
            'supported_formats': ['.jpg', '.jpeg', '.png', '.pdf'],
            'preprocessing': {
                'denoise': True,
                'deskew': True,
                'enhance_contrast': True,
                'resize_max_width': 2000
            },
            'quality_thresholds': {
                'sharpness_min': 50.0,
                'contrast_min': 30.0,
                'brightness_min': 80.0,
                'brightness_max': 200.0
            }
        }
    
    def process_image(self, image_path: str) -> Dict[str, Any]:
        """Main image processing pipeline.
        
        Args:
            image_path: Path to input image file
            
        Returns:
            Dictionary containing processed image and metadata
        """
        try:
            # Validate input
            self._validate_image_file(image_path)
            
            # Load image
            original_image = self._load_image(image_path)
            if original_image is None:
                raise ValueError(f"Could not load image: {image_path}")
            
            # Store original for comparison
            processed_image = original_image.copy()
            processing_steps = []
            
            # Apply preprocessing pipeline
            if self.config['preprocessing']['denoise']:
                processed_image = self._denoise_image(processed_image)
                processing_steps.append('denoise')
            
            if self.config['preprocessing']['deskew']:
                processed_image, skew_angle = self._deskew_image(processed_image)
                processing_steps.append(f'deskew_{skew_angle:.2f}deg')
            
            if self.config['preprocessing']['enhance_contrast']:
                processed_image = self._enhance_contrast(processed_image)
                processing_steps.append('enhance_contrast')
            
            # Resize if needed
            max_width = self.config['preprocessing']['resize_max_width']
            if processed_image.shape[1] > max_width:
                processed_image = self._resize_image(processed_image, max_width)
                processing_steps.append(f'resize_to_{max_width}px')
            
            # Calculate quality metrics
            quality_metrics = self._assess_image_quality(processed_image)
            
            return {
                'original_image': original_image,
                'processed_image': processed_image,
                'processing_steps': processing_steps,
                'quality_metrics': quality_metrics,
                'file_info': {
                    'path': image_path,
                    'original_size': original_image.shape,
                    'processed_size': processed_image.shape,
                    'file_size_mb': Path(image_path).stat().st_size / (1024 * 1024)
                }
            }
            
        except Exception as e:
            logger.error(f"Image processing failed for {image_path}: {e}")
            raise
    
    def _validate_image_file(self, image_path: str) -> None:
        """Validate image file before processing."""
        path = Path(image_path)
        
        if not path.exists():
            raise FileNotFoundError(f"Image file not found: {image_path}")
        
        if path.suffix.lower() not in self.config['supported_formats']:
            raise ValueError(f"Unsupported file format: {path.suffix}")
        
        file_size_mb = path.stat().st_size / (1024 * 1024)
        if file_size_mb > self.config['max_file_size_mb']:
            raise ValueError(f"File too large: {file_size_mb:.1f}MB > {self.config['max_file_size_mb']}MB")
    
    def _load_image(self, image_path: str) -> Optional[np.ndarray]:
        """Load image from file, handling different formats."""
        try:
            path = Path(image_path)
            
            if path.suffix.lower() == '.pdf':
                # For PDF files, convert first page to image
                return self._load_pdf_image(image_path)
            else:
                # Load regular image files
                image = cv2.imread(image_path)
                if image is not None:
                    # Convert BGR to RGB for consistent processing
                    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                return None
                
        except Exception as e:
            logger.error(f"Failed to load image {image_path}: {e}")
            return None
    
    def _load_pdf_image(self, pdf_path: str) -> Optional[np.ndarray]:
        """Convert first page of PDF to image."""
        try:
            # Try using PIL for basic PDF support
            from PIL import Image
            with Image.open(pdf_path) as img:
                # Convert to RGB if needed
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                return np.array(img)
        except Exception as e:
            logger.warning(f"PDF loading failed, trying alternative method: {e}")
            # For full PDF support, would need pdf2image library
            # For now, return None to indicate unsupported
            return None
    
    def _denoise_image(self, image: np.ndarray) -> np.ndarray:
        """Remove noise from scanned image."""
        try:
            # Convert to grayscale for denoising
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
                # Apply non-local means denoising
                denoised_gray = cv2.fastNlMeansDenoising(gray, None, 10, 7, 21)
                # Convert back to RGB
                return cv2.cvtColor(denoised_gray, cv2.COLOR_GRAY2RGB)
            else:
                return cv2.fastNlMeansDenoising(image, None, 10, 7, 21)
        except Exception as e:
            logger.warning(f"Denoising failed, using original: {e}")
            return image
    
    def _deskew_image(self, image: np.ndarray) -> Tuple[np.ndarray, float]:
        """Detect and correct skew in scanned image."""
        try:
            # Convert to grayscale
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            else:
                gray = image
            
            # Apply threshold to get binary image
            _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            # Invert if needed (text should be black on white background)
            if np.mean(binary) < 127:
                binary = 255 - binary
            
            # Find contours to detect text regions
            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Filter contours by area to focus on text regions
            min_area = gray.shape[0] * gray.shape[1] * 0.001  # 0.1% of image area
            text_contours = [c for c in contours if cv2.contourArea(c) > min_area]
            
            if not text_contours:
                return image, 0.0
            
            # Find minimum area rectangle for largest contours
            angles = []
            for contour in sorted(text_contours, key=cv2.contourArea, reverse=True)[:10]:
                rect = cv2.minAreaRect(contour)
                angle = rect[2]
                
                # Normalize angle to [-45, 45] range
                if angle < -45:
                    angle += 90
                elif angle > 45:
                    angle -= 90
                
                angles.append(angle)
            
            if not angles:
                return image, 0.0
            
            # Use median angle to avoid outliers
            skew_angle = np.median(angles)
            
            # Only correct if skew is significant (> 0.5 degrees)
            if abs(skew_angle) < 0.5:
                return image, skew_angle
            
            # Apply rotation correction
            h, w = image.shape[:2]
            center = (w // 2, h // 2)
            rotation_matrix = cv2.getRotationMatrix2D(center, skew_angle, 1.0)
            
            # Calculate new image size to avoid cropping
            cos = np.abs(rotation_matrix[0, 0])
            sin = np.abs(rotation_matrix[0, 1])
            new_w = int((h * sin) + (w * cos))
            new_h = int((h * cos) + (w * sin))
            
            # Adjust rotation matrix for new center
            rotation_matrix[0, 2] += (new_w / 2) - center[0]
            rotation_matrix[1, 2] += (new_h / 2) - center[1]
            
            # Apply rotation
            corrected = cv2.warpAffine(image, rotation_matrix, (new_w, new_h), 
                                     flags=cv2.INTER_CUBIC, 
                                     borderMode=cv2.BORDER_CONSTANT, 
                                     borderValue=(255, 255, 255))
            
            return corrected, skew_angle
            
        except Exception as e:
            logger.warning(f"Deskewing failed, using original: {e}")
            return image, 0.0
    
    def _enhance_contrast(self, image: np.ndarray) -> np.ndarray:
        """Enhance contrast for better text readability."""
        try:
            # Convert to LAB color space for better contrast enhancement
            if len(image.shape) == 3:
                lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
                l_channel, a_channel, b_channel = cv2.split(lab)
                
                # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) to L channel
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                enhanced_l = clahe.apply(l_channel)
                
                # Merge channels back
                enhanced_lab = cv2.merge([enhanced_l, a_channel, b_channel])
                enhanced = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2RGB)
                
                return enhanced
            else:
                # For grayscale, apply CLAHE directly
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                return clahe.apply(image)
                
        except Exception as e:
            logger.warning(f"Contrast enhancement failed, using original: {e}")
            return image
    
    def _resize_image(self, image: np.ndarray, max_width: int) -> np.ndarray:
        """Resize image while maintaining aspect ratio."""
        try:
            h, w = image.shape[:2]
            
            if w <= max_width:
                return image
            
            # Calculate new dimensions
            aspect_ratio = h / w
            new_width = max_width
            new_height = int(new_width * aspect_ratio)
            
            # Resize using high-quality interpolation
            resized = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
            
            return resized
            
        except Exception as e:
            logger.warning(f"Resizing failed, using original: {e}")
            return image
    
    def _assess_image_quality(self, image: np.ndarray) -> Dict[str, Any]:
        """Assess various quality metrics of the image."""
        try:
            # Convert to grayscale for analysis
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            else:
                gray = image
            
            # Calculate quality metrics
            quality_metrics = {}
            
            # Sharpness (using Laplacian variance)
            laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
            quality_metrics['sharpness'] = float(laplacian_var)
            
            # Contrast (standard deviation of pixel intensities)
            contrast = float(np.std(gray))
            quality_metrics['contrast'] = contrast
            
            # Brightness (mean pixel intensity)
            brightness = float(np.mean(gray))
            quality_metrics['brightness'] = brightness
            
            # Overall quality assessment
            thresholds = self.config['quality_thresholds']
            
            quality_flags = {
                'is_sharp': laplacian_var >= thresholds['sharpness_min'],
                'has_good_contrast': contrast >= thresholds['contrast_min'],
                'is_well_lit': (thresholds['brightness_min'] <= brightness <= thresholds['brightness_max'])
            }
            
            # Calculate overall quality score (0-100)
            quality_score = 0
            if quality_flags['is_sharp']:
                quality_score += 40
            if quality_flags['has_good_contrast']:
                quality_score += 30
            if quality_flags['is_well_lit']:
                quality_score += 30
            
            quality_metrics.update({
                'quality_flags': quality_flags,
                'overall_score': quality_score,
                'is_suitable_for_ocr': quality_score >= 70
            })
            
            return quality_metrics
            
        except Exception as e:
            logger.error(f"Quality assessment failed: {e}")
            return {
                'sharpness': 0.0,
                'contrast': 0.0,
                'brightness': 0.0,
                'quality_flags': {'is_sharp': False, 'has_good_contrast': False, 'is_well_lit': False},
                'overall_score': 0,
                'is_suitable_for_ocr': False
            }
    
    def save_processed_image(self, processed_image: np.ndarray, output_path: str) -> bool:
        """Save processed image to file."""
        try:
            # Convert RGB to BGR for OpenCV
            if len(processed_image.shape) == 3:
                bgr_image = cv2.cvtColor(processed_image, cv2.COLOR_RGB2BGR)
            else:
                bgr_image = processed_image
            
            success = cv2.imwrite(output_path, bgr_image)
            
            if success:
                logger.info(f"Processed image saved to: {output_path}")
            else:
                logger.error(f"Failed to save processed image to: {output_path}")
            
            return success
            
        except Exception as e:
            logger.error(f"Error saving processed image: {e}")
            return False


def create_sample_test_image() -> np.ndarray:
    """Create a sample test image for development testing."""
    # Create a white background
    height, width = 800, 600
    image = np.ones((height, width, 3), dtype=np.uint8) * 255
    
    # Add some text-like rectangles to simulate a scanned assignment
    cv2.rectangle(image, (50, 50), (550, 100), (0, 0, 0), 2)  # Header
    cv2.rectangle(image, (50, 120), (550, 150), (0, 0, 0), 1)  # Line 1
    cv2.rectangle(image, (50, 170), (450, 200), (0, 0, 0), 1)  # Line 2
    cv2.rectangle(image, (50, 220), (500, 250), (0, 0, 0), 1)  # Line 3
    
    # Add some "handwritten" marks
    cv2.circle(image, (100, 300), 5, (0, 0, 0), -1)
    cv2.circle(image, (200, 320), 3, (0, 0, 0), -1)
    cv2.circle(image, (300, 310), 4, (0, 0, 0), -1)
    
    return image


def demo_image_processing():
    """Demo function to test image processing capabilities."""
    print("🖼️  Image Processing Demo")
    print("=" * 40)
    
    try:
        # Create processor
        processor = ImageProcessor()
        
        # Create sample image
        sample_image = create_sample_test_image()
        
        # Add some noise and skew for testing
        noise = np.random.randint(0, 30, sample_image.shape, dtype=np.uint8)
        noisy_image = cv2.add(sample_image, noise)
        
        # Simulate slight skew
        h, w = noisy_image.shape[:2]
        center = (w // 2, h // 2)
        angle = 2.5  # 2.5 degree skew
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        skewed_image = cv2.warpAffine(noisy_image, rotation_matrix, (w, h))
        
        print("✅ Created sample test image with noise and skew")
        
        # Test individual processing steps
        print("\n🔧 Testing processing steps:")
        
        # Denoising
        denoised = processor._denoise_image(skewed_image)
        print("✅ Denoising")
        
        # Deskewing
        deskewed, detected_angle = processor._deskew_image(denoised)
        print(f"✅ Deskewing (detected angle: {detected_angle:.2f}°)")
        
        # Contrast enhancement
        enhanced = processor._enhance_contrast(deskewed)
        print("✅ Contrast enhancement")
        
        # Quality assessment
        quality = processor._assess_image_quality(enhanced)
        print(f"✅ Quality assessment (score: {quality['overall_score']}/100)")
        
        # Test full pipeline with temporary file
        import tempfile
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
            cv2.imwrite(tmp.name, cv2.cvtColor(skewed_image, cv2.COLOR_RGB2BGR))
            
            try:
                result = processor.process_image(tmp.name)
                print(f"\n📊 Full pipeline results:")
                print(f"   Original size: {result['file_info']['original_size']}")
                print(f"   Processing steps: {', '.join(result['processing_steps'])}")
                print(f"   Quality score: {result['quality_metrics']['overall_score']}/100")
                print(f"   Suitable for OCR: {result['quality_metrics']['is_suitable_for_ocr']}")
                
                return True
                
            finally:
                # Cleanup
                Path(tmp.name).unlink(missing_ok=True)
        
    except ImportError as e:
        print(f"❌ Missing dependencies: {e}")
        print("Install with: pip install opencv-python pillow numpy")
        return False
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    demo_image_processing()
