#!/usr/bin/env python3
"""
AutoGrading System - QR Code Extraction Module
Detects and extracts QR codes from scanned assignment images.
"""

import cv2
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
import logging
import json
from pathlib import Path

# Setup logging
logger = logging.getLogger(__name__)


class QRExtractor:
    """Extracts QR codes and metadata from scanned assignment images."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize QR extractor with configuration.
        
        Args:
            config: Configuration dictionary for extraction parameters
        """
        self.config = config or self._default_config()
        self._detector = None
        self._init_detector()
    
    def _default_config(self) -> Dict[str, Any]:
        """Get default configuration settings."""
        return {
            'detection_methods': ['pyzbar', 'opencv'],  # Preferred order
            'preprocessing': {
                'apply_threshold': True,
                'blur_reduction': True,
                'contrast_enhancement': True
            },
            'search_regions': {
                'top_right': {'x': 0.6, 'y': 0.0, 'w': 0.4, 'h': 0.3},
                'top_left': {'x': 0.0, 'y': 0.0, 'w': 0.4, 'h': 0.3},
                'bottom_right': {'x': 0.6, 'y': 0.7, 'w': 0.4, 'h': 0.3},
                'full_image': {'x': 0.0, 'y': 0.0, 'w': 1.0, 'h': 1.0}
            },
            'validation': {
                'require_assignment_metadata': True,
                'min_qr_size': 50,  # pixels
                'max_qr_size': 500  # pixels
            }
        }
    
    def _init_detector(self):
        """Initialize QR code detection libraries."""
        self.detectors = {}
        
        # Try to initialize pyzbar
        try:
            import pyzbar.pyzbar as pyzbar
            self.detectors['pyzbar'] = pyzbar
            logger.info("Initialized pyzbar QR detector")
        except ImportError:
            logger.warning("pyzbar not available for QR detection")
        
        # Try to initialize OpenCV QR detector
        try:
            detector = cv2.QRCodeDetector()
            self.detectors['opencv'] = detector
            logger.info("Initialized OpenCV QR detector")
        except AttributeError:
            logger.warning("OpenCV QR detector not available")
        
        if not self.detectors:
            raise ImportError("No QR detection libraries available. Install pyzbar: pip install pyzbar")
    
    def extract_qr_codes(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """Extract all QR codes from an image.
        
        Args:
            image: Input image as numpy array
            
        Returns:
            List of dictionaries containing QR code data and metadata
        """
        try:
            qr_codes = []
            
            # Preprocess image for better detection
            processed_image = self._preprocess_for_qr(image)
            
            # Try detection in different regions
            search_regions = self.config['search_regions']
            
            for region_name, region in search_regions.items():
                logger.debug(f"Searching for QR codes in region: {region_name}")
                
                # Extract region from image
                region_image = self._extract_region(processed_image, region)
                
                # Try different detection methods
                for method in self.config['detection_methods']:
                    if method in self.detectors:
                        detected = self._detect_with_method(region_image, method, region_name)
                        qr_codes.extend(detected)
                
                # If we found QR codes, we might not need to search other regions
                if qr_codes and region_name != 'full_image':
                    break
            
            # Remove duplicates based on data content
            unique_codes = self._remove_duplicate_qr_codes(qr_codes)
            
            # Validate and parse detected QR codes
            valid_codes = []
            for qr_code in unique_codes:
                if self._validate_qr_code(qr_code):
                    parsed = self._parse_qr_metadata(qr_code)
                    if parsed:
                        valid_codes.append(parsed)
            
            logger.info(f"Extracted {len(valid_codes)} valid QR codes from image")
            return valid_codes
            
        except Exception as e:
            logger.error(f"QR extraction failed: {e}")
            return []
    
    def _preprocess_for_qr(self, image: np.ndarray) -> np.ndarray:
        """Preprocess image to improve QR code detection."""
        processed = image.copy()
        
        # Convert to grayscale if needed
        if len(processed.shape) == 3:
            processed = cv2.cvtColor(processed, cv2.COLOR_RGB2GRAY)
        
        if self.config['preprocessing']['contrast_enhancement']:
            # Apply CLAHE for better contrast
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
            processed = clahe.apply(processed)
        
        if self.config['preprocessing']['blur_reduction']:
            # Apply slight sharpening
            kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
            processed = cv2.filter2D(processed, -1, kernel)
        
        if self.config['preprocessing']['apply_threshold']:
            # Apply adaptive threshold
            processed = cv2.adaptiveThreshold(
                processed, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                cv2.THRESH_BINARY, 11, 2
            )
        
        return processed
    
    def _extract_region(self, image: np.ndarray, region: Dict[str, float]) -> np.ndarray:
        """Extract a specific region from the image."""
        h, w = image.shape[:2]
        
        x1 = int(region['x'] * w)
        y1 = int(region['y'] * h)
        x2 = int((region['x'] + region['w']) * w)
        y2 = int((region['y'] + region['h']) * h)
        
        # Ensure coordinates are within image bounds
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        
        return image[y1:y2, x1:x2]
    
    def _detect_with_method(self, image: np.ndarray, method: str, region_name: str) -> List[Dict[str, Any]]:
        """Detect QR codes using specified method."""
        detected_codes = []
        
        try:
            if method == 'pyzbar' and 'pyzbar' in self.detectors:
                pyzbar = self.detectors['pyzbar']
                
                # Detect QR codes
                qr_codes = pyzbar.decode(image)
                
                for qr in qr_codes:
                    # Extract bounding box
                    x, y, w, h = qr.rect
                    
                    # Validate QR code size
                    if not self._validate_qr_size(w, h):
                        continue
                    
                    detected_codes.append({
                        'data': qr.data.decode('utf-8'),
                        'type': qr.type,
                        'bbox': {'x': x, 'y': y, 'width': w, 'height': h},
                        'detection_method': method,
                        'search_region': region_name,
                        'confidence': 1.0  # pyzbar doesn't provide confidence
                    })
            
            elif method == 'opencv' and 'opencv' in self.detectors:
                detector = self.detectors['opencv']
                
                # Detect and decode QR codes
                data, bbox, _ = detector.detectAndDecode(image)
                
                if data and bbox is not None:
                    # Calculate bounding box
                    points = bbox[0]
                    x_coords = points[:, 0]
                    y_coords = points[:, 1]
                    x, y = int(np.min(x_coords)), int(np.min(y_coords))
                    w = int(np.max(x_coords) - np.min(x_coords))
                    h = int(np.max(y_coords) - np.min(y_coords))
                    
                    # Validate QR code size
                    if self._validate_qr_size(w, h):
                        detected_codes.append({
                            'data': data,
                            'type': 'QRCODE',
                            'bbox': {'x': x, 'y': y, 'width': w, 'height': h},
                            'detection_method': method,
                            'search_region': region_name,
                            'confidence': 0.9  # OpenCV doesn't provide confidence
                        })
        
        except Exception as e:
            logger.debug(f"Detection with {method} failed: {e}")
        
        return detected_codes
    
    def _validate_qr_size(self, width: int, height: int) -> bool:
        """Validate QR code size is within acceptable range."""
        min_size = self.config['validation']['min_qr_size']
        max_size = self.config['validation']['max_qr_size']
        
        size = max(width, height)
        return min_size <= size <= max_size
    
    def _remove_duplicate_qr_codes(self, qr_codes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Remove duplicate QR codes based on data content."""
        seen_data = set()
        unique_codes = []
        
        for qr_code in qr_codes:
            data = qr_code['data']
            if data not in seen_data:
                seen_data.add(data)
                unique_codes.append(qr_code)
        
        return unique_codes
    
    def _validate_qr_code(self, qr_code: Dict[str, Any]) -> bool:
        """Validate that QR code contains expected assignment metadata."""
        try:
            data = qr_code['data']
            
            # Try to parse as JSON
            try:
                metadata = json.loads(data)
            except json.JSONDecodeError:
                logger.debug(f"QR code data is not valid JSON: {data[:50]}...")
                return False
            
            # Check for required fields if validation is enabled
            if self.config['validation']['require_assignment_metadata']:
                required_fields = ['assignment_id', 'assignment_name', 'subject']
                
                for field in required_fields:
                    if field not in metadata:
                        logger.debug(f"QR code missing required field: {field}")
                        return False
            
            return True
            
        except Exception as e:
            logger.debug(f"QR code validation failed: {e}")
            return False
    
    def _parse_qr_metadata(self, qr_code: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Parse and structure QR code metadata."""
        try:
            # Parse JSON data
            metadata = json.loads(qr_code['data'])
            
            # Create structured result
            result = {
                'qr_detection': {
                    'bbox': qr_code['bbox'],
                    'detection_method': qr_code['detection_method'],
                    'search_region': qr_code['search_region'],
                    'confidence': qr_code['confidence']
                },
                'assignment_metadata': metadata,
                'extraction_success': True,
                'extraction_timestamp': None  # Would be set by calling code
            }
            
            # Add convenience fields
            result['assignment_id'] = metadata.get('assignment_id')
            result['assignment_name'] = metadata.get('assignment_name')
            result['subject'] = metadata.get('subject')
            result['total_points'] = metadata.get('total_points')
            
            return result
            
        except Exception as e:
            logger.error(f"Failed to parse QR metadata: {e}")
            return None
    
    def extract_from_file(self, image_path: str) -> List[Dict[str, Any]]:
        """Extract QR codes from an image file.
        
        Args:
            image_path: Path to image file
            
        Returns:
            List of extracted QR code data
        """
        try:
            # Load image
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"Could not load image: {image_path}")
            
            # Convert BGR to RGB
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Extract QR codes
            qr_codes = self.extract_qr_codes(image_rgb)
            
            # Add file information
            for qr_code in qr_codes:
                qr_code['source_file'] = image_path
                qr_code['extraction_timestamp'] = Path(image_path).stat().st_mtime
            
            return qr_codes
            
        except Exception as e:
            logger.error(f"Failed to extract QR codes from {image_path}: {e}")
            return []
    
    def visualize_detection(self, image: np.ndarray, qr_codes: List[Dict[str, Any]]) -> np.ndarray:
        """Create visualization of detected QR codes on image.
        
        Args:
            image: Original image
            qr_codes: List of detected QR codes
            
        Returns:
            Image with QR code bounding boxes drawn
        """
        vis_image = image.copy()
        
        for i, qr_code in enumerate(qr_codes):
            bbox = qr_code['qr_detection']['bbox']
            x, y, w, h = bbox['x'], bbox['y'], bbox['width'], bbox['height']
            
            # Draw bounding box
            color = (0, 255, 0) if qr_code['extraction_success'] else (255, 0, 0)
            cv2.rectangle(vis_image, (x, y), (x + w, y + h), color, 2)
            
            # Add label
            label = f"QR {i+1}: {qr_code.get('assignment_id', 'Unknown')}"
            cv2.putText(vis_image, label, (x, y - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        return vis_image


class QRExtractionResult:
    """Container for QR extraction results."""
    
    def __init__(self, image_path: str):
        self.image_path = image_path
        self.qr_codes = []
        self.extraction_time = 0.0
        self.success = False
        self.error_message = None
    
    def add_qr_code(self, qr_data: Dict[str, Any]):
        """Add detected QR code data."""
        self.qr_codes.append(qr_data)
    
    def get_primary_assignment(self) -> Optional[Dict[str, Any]]:
        """Get the primary assignment metadata from QR codes."""
        if not self.qr_codes:
            return None
        
        # Return the first successfully parsed QR code
        for qr_code in self.qr_codes:
            if qr_code.get('extraction_success', False):
                return qr_code.get('assignment_metadata')
        
        return None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary format."""
        return {
            'image_path': self.image_path,
            'qr_codes_found': len(self.qr_codes),
            'qr_codes': self.qr_codes,
            'extraction_time': self.extraction_time,
            'success': self.success,
            'error_message': self.error_message,
            'primary_assignment': self.get_primary_assignment()
        }


def create_test_image_with_qr():
    """Create a test image with embedded QR code for testing."""
    try:
        # Import QR generator from Week 2
        from qr_generator import QRCodeGenerator, create_sample_assignment
        
        # Create sample assignment
        metadata = create_sample_assignment()
        
        # Generate QR code
        generator = QRCodeGenerator()
        qr_image = generator.generate_qr_image(metadata)
        
        if qr_image is None:
            return None, None
        
        # Create background image (simulating scanned assignment)
        bg_height, bg_width = 800, 600
        background = np.ones((bg_height, bg_width, 3), dtype=np.uint8) * 255
        
        # Add some content to simulate assignment
        cv2.rectangle(background, (50, 50), (550, 100), (0, 0, 0), 2)  # Header
        cv2.rectangle(background, (50, 120), (550, 150), (0, 0, 0), 1)  # Content
        cv2.rectangle(background, (50, 170), (450, 200), (0, 0, 0), 1)
        
        # Convert PIL QR image to OpenCV format
        qr_array = np.array(qr_image.convert('RGB'))
        qr_height, qr_width = qr_array.shape[:2]
        
        # Place QR code in top-right corner
        margin = 20
        y_pos = margin
        x_pos = bg_width - qr_width - margin
        
        # Ensure QR code fits
        if x_pos > 0 and y_pos + qr_height < bg_height:
            background[y_pos:y_pos+qr_height, x_pos:x_pos+qr_width] = qr_array
        
        return background, metadata
        
    except ImportError:
        print("QR generator not available for test image creation")
        return None, None
    except Exception as e:
        print(f"Test image creation failed: {e}")
        return None, None


def demo_qr_extraction():
    """Demo function to test QR code extraction capabilities."""
    print("🔍 QR Code Extraction Demo")
    print("=" * 40)
    
    try:
        # Create extractor
        extractor = QRExtractor()
        
        print(f"✅ Initialized QR extractor with {len(extractor.detectors)} detection methods")
        print(f"   Available methods: {list(extractor.detectors.keys())}")
        
        # Test with sample image
        test_image, original_metadata = create_test_image_with_qr()
        
        if test_image is not None:
            print("\n🧪 Testing QR extraction on sample image...")
            
            # Extract QR codes
            qr_codes = extractor.extract_qr_codes(test_image)
            
            print(f"✅ Found {len(qr_codes)} QR codes")
            
            if qr_codes:
                for i, qr_code in enumerate(qr_codes):
                    print(f"\n📋 QR Code {i+1}:")
                    print(f"   Assignment ID: {qr_code.get('assignment_id')}")
                    print(f"   Assignment Name: {qr_code.get('assignment_name')}")
                    print(f"   Subject: {qr_code.get('subject')}")
                    print(f"   Detection method: {qr_code['qr_detection']['detection_method']}")
                    print(f"   Search region: {qr_code['qr_detection']['search_region']}")
                    print(f"   Confidence: {qr_code['qr_detection']['confidence']}")
                    
                    # Verify metadata matches original
                    extracted_metadata = qr_code['assignment_metadata']
                    if original_metadata:
                        matches = extracted_metadata.get('assignment_id') == original_metadata.assignment_id
                        print(f"   Metadata verification: {'✅' if matches else '❌'}")
            else:
                print("⚠️  No QR codes detected in test image")
            
            # Test visualization
            try:
                vis_image = extractor.visualize_detection(test_image, qr_codes)
                print("✅ Generated detection visualization")
            except Exception as e:
                print(f"⚠️  Visualization failed: {e}")
            
            # Test extraction result container
            result = QRExtractionResult("test_image.jpg")
            for qr_code in qr_codes:
                result.add_qr_code(qr_code)
            result.success = len(qr_codes) > 0
            
            primary = result.get_primary_assignment()
            if primary:
                print(f"✅ Primary assignment extracted: {primary.get('assignment_name')}")
            
            print("\n🎉 QR extraction demo completed successfully!")
            return True
        
        else:
            print("⚠️  Could not create test image - testing basic functionality only")
            
            # Test basic configuration
            print(f"✅ Detection methods: {extractor.config['detection_methods']}")
            print(f"✅ Search regions: {list(extractor.config['search_regions'].keys())}")
            print(f"✅ Validation enabled: {extractor.config['validation']['require_assignment_metadata']}")
            
            return True
    
    except ImportError as e:
        print(f"❌ Missing dependencies: {e}")
        print("Install with: pip install pyzbar opencv-python")
        return False
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    demo_qr_extraction()
    
    