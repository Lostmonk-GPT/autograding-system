#!/usr/bin/env python3
"""
AutoGrading System - Complete QR Code Generator
Complete implementation with all required methods for OCR pipeline.
"""

import json
import uuid
import hashlib
from datetime import datetime, date
from typing import Optional, Dict, Any
from dataclasses import dataclass, asdict
import logging

# Setup logging
logger = logging.getLogger(__name__)


@dataclass
class AssignmentMetadata:
    """Metadata structure for assignments with QR code embedding."""

    assignment_id: str
    assignment_name: str
    subject: str
    class_period: str
    teacher_name: str
    due_date: str
    total_points: int
    created_date: str
    qr_version: str = "1.0"

    def to_json(self) -> str:
        """Convert metadata to JSON string."""
        return json.dumps(asdict(self), indent=None, separators=(',', ':'))

    @classmethod
    def from_json(cls, json_str: str) -> 'AssignmentMetadata':
        """Create AssignmentMetadata from JSON string."""
        data = json.loads(json_str)
        return cls(**data)


class QRCodeGenerator:
    """Generates QR codes for assignment identification and metadata tracking."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize QR code generator.

        Args:
            config: Configuration dictionary for QR generation settings
        """
        self.config = config or self._default_config()
        logger.info("QRCodeGenerator initialized")

    def _default_config(self) -> Dict[str, Any]:
        """Get default configuration for QR code generation."""
        return {
            'qr_version': 1,
            'error_correction': 'M',  # Medium error correction
            'box_size': 10,
            'border': 4,
            'fill_color': 'black',
            'back_color': 'white',
            'id_prefix': 'AG',  # AutoGrading prefix
            'hash_length': 8
        }

    def generate_assignment_id(self, assignment_name: str) -> str:
        """Generate unique assignment ID.

        Args:
            assignment_name: Name of the assignment

        Returns:
            Unique assignment ID string
        """
        # Create base from assignment name and timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_string = f"{assignment_name}_{timestamp}"

        # Generate hash
        hash_object = hashlib.md5(base_string.encode())
        hash_hex = hash_object.hexdigest()[:self.config['hash_length']]

        # Combine with prefix
        assignment_id = f"{self.config['id_prefix']}_{hash_hex}_{timestamp[-6:]}"

        logger.debug(f"Generated assignment ID: {assignment_id}")
        return assignment_id

    def generate_qr_code(self, data: str):
        """Generate QR code from string data.

        Args:
            data: String data to encode in QR code

        Returns:
            PIL Image object or None if generation fails
        """
        try:
            import qrcode
            from qrcode.constants import ERROR_CORRECT_L, ERROR_CORRECT_M, ERROR_CORRECT_Q, ERROR_CORRECT_H

            # Map error correction levels
            error_correction_map = {
                'L': ERROR_CORRECT_L,
                'M': ERROR_CORRECT_M,
                'Q': ERROR_CORRECT_Q,
                'H': ERROR_CORRECT_H
            }

            error_correction = error_correction_map.get(
                self.config['error_correction'],
                ERROR_CORRECT_M
            )

            # Create QR code instance
            qr = qrcode.QRCode(
                version=self.config['qr_version'],
                error_correction=error_correction,
                box_size=self.config['box_size'],
                border=self.config['border'],
            )

            # Add data and generate
            qr.add_data(data)
            qr.make(fit=True)

            # Create image
            qr_image = qr.make_image(
                fill_color=self.config['fill_color'],
                back_color=self.config['back_color']
            )

            logger.debug(
                f"Generated QR code for {len(data)} characters of data")
            return qr_image

        except ImportError:
            logger.error(
                "qrcode library not available. Install with: pip install qrcode[pil]")
            return None
        except Exception as e:
            logger.error(f"QR code generation failed: {e}")
            return None

    def generate_qr_image(self, metadata):
        """Generate QR code image from assignment metadata.

        Args:
            metadata: AssignmentMetadata object, dict, or string

        Returns:
            PIL Image object or None if generation fails
        """
        try:
            # Handle different types of metadata input
            json_data = None

            if isinstance(metadata, AssignmentMetadata):
                json_data = metadata.to_json()
                logger.debug("Using AssignmentMetadata.to_json()")
            elif hasattr(metadata, 'to_json') and callable(getattr(metadata, 'to_json')):
                json_data = metadata.to_json()
                logger.debug("Using metadata.to_json()")
            elif isinstance(metadata, dict):
                json_data = json.dumps(metadata, separators=(',', ':'))
                logger.debug("Converting dict to JSON")
            elif hasattr(metadata, '__dict__'):
                json_data = json.dumps(
                    metadata.__dict__, separators=(',', ':'))
                logger.debug("Converting object.__dict__ to JSON")
            elif isinstance(metadata, str):
                # If it's already a string, try to validate it as JSON
                try:
                    json.loads(metadata)  # Test if it's valid JSON
                    json_data = metadata
                    logger.debug("Using string as-is (valid JSON)")
                except json.JSONDecodeError:
                    # Not valid JSON, wrap it in a simple structure
                    json_data = json.dumps(
                        {"data": metadata}, separators=(',', ':'))
                    logger.debug("Wrapping string in JSON structure")
            else:
                # Last resort: convert to string and wrap
                json_data = json.dumps(
                    {"data": str(metadata)}, separators=(',', ':'))
                logger.debug(f"Converting {type(metadata)} to JSON")

            if not json_data:
                logger.error("Failed to convert metadata to JSON")
                return None

            # Generate QR code image
            qr_image = self.generate_qr_code(json_data)

            if qr_image:
                logger.info(f"Generated QR image for assignment metadata")
                return qr_image
            else:
                logger.error("Failed to generate QR image")
                return None

        except Exception as e:
            logger.error(f"QR image generation failed: {e}")
            import traceback
            traceback.print_exc()
            return None

    def create_assignment_with_qr(self, assignment_name: str, subject: str,
                                  teacher_name: str, class_period: str = "Unknown",
                                  total_points: int = 100) -> tuple:
        """Create assignment metadata and generate QR code.

        Args:
            assignment_name: Name of the assignment
            subject: Subject area (algebra1, ap_calculus, etc.)
            teacher_name: Teacher's name
            class_period: Class period or section
            total_points: Total points for assignment

        Returns:
            Tuple of (AssignmentMetadata, PIL_Image) or (metadata, None) if QR fails
        """
        try:
            # Generate assignment ID
            assignment_id = self.generate_assignment_id(assignment_name)

            # Create metadata
            metadata = AssignmentMetadata(
                assignment_id=assignment_id,
                assignment_name=assignment_name,
                subject=subject,
                class_period=class_period,
                teacher_name=teacher_name,
                due_date=datetime.now().strftime("%Y-%m-%d"),
                total_points=total_points,
                created_date=datetime.now().isoformat(),
                qr_version=str(self.config['qr_version'])
            )

            # Generate QR code
            qr_image = self.generate_qr_image(metadata)

            logger.info(f"Created assignment with QR: {assignment_name}")
            return metadata, qr_image

        except Exception as e:
            logger.error(f"Failed to create assignment with QR: {e}")
            return None, None

    def validate_qr_data(self, qr_data: str) -> bool:
        """Validate QR code data format.

        Args:
            qr_data: Raw QR code data string

        Returns:
            True if data appears to be valid assignment metadata
        """
        try:
            # Try to parse as JSON
            data = json.loads(qr_data)

            # Check for required fields
            required_fields = ['assignment_id', 'assignment_name', 'subject']

            for field in required_fields:
                if field not in data:
                    return False

            # Validate assignment ID format
            assignment_id = data['assignment_id']
            if not assignment_id.startswith(self.config['id_prefix']):
                return False

            return True

        except (json.JSONDecodeError, KeyError, TypeError):
            return False

    def get_qr_info(self) -> Dict[str, Any]:
        """Get information about QR generation settings."""
        return {
            'generator_version': '1.0.0',
            'qr_version': self.config['qr_version'],
            'error_correction': self.config['error_correction'],
            'box_size': self.config['box_size'],
            'border': self.config['border'],
            'id_prefix': self.config['id_prefix']
        }


def create_sample_assignment() -> AssignmentMetadata:
    """Create sample assignment metadata for testing."""
    return AssignmentMetadata(
        assignment_id="AG_abc12345_123456",
        assignment_name="Linear Equations Practice",
        subject="algebra1",
        class_period="Period 3",
        teacher_name="Ms. Johnson",
        due_date="2025-06-20",
        total_points=100,
        created_date=datetime.now().isoformat(),
        qr_version="1.0"
    )


def demo_qr_generation() -> bool:
    """Demo function to test QR code generation."""
    print("🔲 QR Code Generation Demo")
    print("=" * 35)

    try:
        # Create generator
        generator = QRCodeGenerator()

        print("✅ QR Code Generator initialized")

        # Show configuration
        qr_info = generator.get_qr_info()
        print(f"✅ Configuration: Version {qr_info['qr_version']}, "
              f"Error correction: {qr_info['error_correction']}")

        # Test 1: Create sample assignment
        print("\n🧪 Test 1: Sample Assignment Creation")
        metadata = create_sample_assignment()
        print(f"✅ Sample metadata: {metadata.assignment_name}")
        print(f"   ID: {metadata.assignment_id}")
        print(f"   Subject: {metadata.subject}")

        # Test 2: Generate QR image
        print("\n🧪 Test 2: QR Image Generation")
        qr_image = generator.generate_qr_image(metadata)

        if qr_image:
            print(f"✅ QR image generated successfully")
            print(f"   Size: {qr_image.size}")
            print(f"   Mode: {qr_image.mode}")
        else:
            print("❌ QR image generation failed")
            return False

        # Test 3: Create complete assignment with QR
        print("\n🧪 Test 3: Complete Assignment Creation")
        new_metadata, new_qr = generator.create_assignment_with_qr(
            assignment_name="Quadratic Equations Test",
            subject="algebra1",
            teacher_name="Mr. Smith",
            class_period="Period 2",
            total_points=85
        )

        if new_metadata and new_qr:
            print(f"✅ Complete assignment created")
            print(f"   Name: {new_metadata.assignment_name}")
            print(f"   ID: {new_metadata.assignment_id}")
            print(f"   QR Size: {new_qr.size}")
        else:
            print("❌ Complete assignment creation failed")
            return False

        # Test 4: JSON serialization
        print("\n🧪 Test 4: JSON Serialization")
        json_data = metadata.to_json()
        print(f"✅ JSON size: {len(json_data)} characters")

        # Test round-trip
        restored_metadata = AssignmentMetadata.from_json(json_data)
        if restored_metadata.assignment_id == metadata.assignment_id:
            print("✅ JSON round-trip successful")
        else:
            print("❌ JSON round-trip failed")
            return False

        # Test 5: QR validation
        print("\n🧪 Test 5: QR Data Validation")
        valid = generator.validate_qr_data(json_data)
        if valid:
            print("✅ QR data validation passed")
        else:
            print("❌ QR data validation failed")
            return False

        print("\n🎉 All QR generation tests passed!")
        return True

    except ImportError:
        print("❌ QR code library not available")
        print("Install with: pip install qrcode[pil] pillow")
        return False
    except Exception as e:
        print(f"❌ QR generation demo failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    # Run demo when script is executed directly
    success = demo_qr_generation()

    if success:
        print("\n✨ QR Generator is ready for use!")
        print("Key methods available:")
        print("  - generate_qr_image(metadata)")
        print("  - generate_qr_code(data)")
        print("  - create_assignment_with_qr(...)")
    else:
        print("\n❌ QR Generator setup needs attention")

    exit(0 if success else 1)
