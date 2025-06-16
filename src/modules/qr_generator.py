#!/usr/bin/env python3
"""
AutoGrading System - QR Code Generator (Fixed Version)
Simple but complete QR generator with all required methods.
"""

import json
import uuid
import hashlib
from datetime import datetime
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
    """Simple QR code generator with all required methods."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize QR code generator."""
        self.config = config or {}
        logger.info("QRCodeGenerator initialized")
    
    def generate_assignment_id(self, assignment_name: str) -> str:
        """Generate unique assignment ID."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        hash_part = hashlib.md5(f"{assignment_name}_{timestamp}".encode()).hexdigest()[:8]
        return f"AG_{hash_part}_{timestamp[-6:]}"
    
    def generate_qr_code(self, data: str):
        """Generate QR code from string data."""
        try:
            import qrcode
            
            qr = qrcode.QRCode(
                version=1,
                error_correction=qrcode.constants.ERROR_CORRECT_L,
                box_size=10,
                border=4,
            )
            qr.add_data(data)
            qr.make(fit=True)
            
            return qr.make_image(fill_color="black", back_color="white")
            
        except ImportError:
            print("❌ qrcode library not available. Install with: pip install qrcode[pil]")
            return None
        except Exception as e:
            print(f"❌ QR code generation failed: {e}")
            return None
    
    def generate_qr_image(self, metadata):
        """Generate QR code image from assignment metadata."""
        try:
            print(f"🔧 generate_qr_image called with: {type(metadata)}")
            
            # Handle different input types
            if hasattr(metadata, 'to_json') and callable(getattr(metadata, 'to_json')):
                json_data = metadata.to_json()
                print("✅ Used metadata.to_json()")
            elif isinstance(metadata, dict):
                json_data = json.dumps(metadata, separators=(',', ':'))
                print("✅ Converted dict to JSON")
            elif isinstance(metadata, str):
                # Wrap string in simple JSON structure
                json_data = json.dumps({"assignment_name": metadata}, separators=(',', ':'))
                print("✅ Wrapped string in JSON")
            elif hasattr(metadata, '__dict__'):
                json_data = json.dumps(metadata.__dict__, separators=(',', ':'))
                print("✅ Used object.__dict__")
            else:
                json_data = json.dumps({"data": str(metadata)}, separators=(',', ':'))
                print(f"✅ Converted {type(metadata)} to JSON")
            
            print(f"📄 JSON data: {json_data[:100]}...")
            
            # Generate QR code
            qr_image = self.generate_qr_code(json_data)
            
            if qr_image:
                print(f"✅ QR image generated: {qr_image.size}")
                return qr_image
            else:
                print("❌ QR image generation failed")
                return None
                
        except Exception as e:
            print(f"❌ generate_qr_image failed: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def create_assignment_with_qr(self, assignment_name: str, subject: str, 
                                  teacher_name: str, class_period: str = "Unknown",
                                  total_points: int = 100):
        """Create assignment metadata and generate QR code."""
        try:
            assignment_id = self.generate_assignment_id(assignment_name)
            
            metadata = AssignmentMetadata(
                assignment_id=assignment_id,
                assignment_name=assignment_name,
                subject=subject,
                class_period=class_period,
                teacher_name=teacher_name,
                due_date=datetime.now().strftime("%Y-%m-%d"),
                total_points=total_points,
                created_date=datetime.now().isoformat(),
                qr_version="1.0"
            )
            
            qr_image = self.generate_qr_image(metadata)
            return metadata, qr_image
            
        except Exception as e:
            print(f"❌ Failed to create assignment with QR: {e}")
            return None, None


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
        generator = QRCodeGenerator()
        print("✅ QR Code Generator initialized")
        
        # Test with sample assignment
        metadata = create_sample_assignment()
        print(f"✅ Sample metadata: {metadata.assignment_name}")
        
        # Test QR image generation
        qr_image = generator.generate_qr_image(metadata)
        
        if qr_image:
            print(f"✅ QR image generated: {qr_image.size}")
            return True
        else:
            print("❌ QR image generation failed")
            return False
        
    except Exception as e:
        print(f"❌ QR generation demo failed: {e}")
        return False


if __name__ == "__main__":
    success = demo_qr_generation()
    exit(0 if success else 1)
