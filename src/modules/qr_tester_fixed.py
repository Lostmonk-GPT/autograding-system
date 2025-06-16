"""
QR Code Generation Module for AutoGrading System
Generates QR codes with assignment metadata for tracking and identification.
"""

import json
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
import qrcode
from qrcode.constants import ERROR_CORRECT_L, ERROR_CORRECT_M, ERROR_CORRECT_Q, ERROR_CORRECT_H
from PIL import Image, ImageDraw, ImageFont
import logging

from config.config import config

logger = logging.getLogger(__name__)


class AssignmentMetadata:
    """Container for assignment metadata to be embedded in QR codes."""
    
    def __init__(
        self,
        assignment_id: str,
        subject: str,
        assignment_name: str,
        due_date: str,
        total_points: int,
        class_period: Optional[str] = None,
        teacher_name: Optional[str] = None,
        academic_year: Optional[str] = None,
        additional_data: Optional[Dict[str, Any]] = None
    ):
        """Initialize assignment metadata.
        
        Args:
            assignment_id: Unique identifier for the assignment
            subject: Subject area (e.g., 'algebra1', 'ap_calculus')
            assignment_name: Human-readable assignment name
            due_date: Due date in ISO format (YYYY-MM-DD)
            total_points: Total points possible for the assignment
            class_period: Class period identifier
            teacher_name: Teacher's name
            academic_year: Academic year (e.g., '2024-2025')
            additional_data: Additional metadata as needed
        """
        self.assignment_id = assignment_id
        self.subject = subject
        self.assignment_name = assignment_name
        self.due_date = due_date
        self.total_points = total_points
        self.class_period = class_period
        self.teacher_name = teacher_name
        self.academic_year = academic_year or self._get_current_academic_year()
        self.additional_data = additional_data or {}
        
        # Add system metadata
        self.created_at = datetime.now().isoformat()
        self.qr_version = "1.0"
    
    def _get_current_academic_year(self) -> str:
        """Generate current academic year string."""
        now = datetime.now()
        if now.month >= 8:  # August onwards is new academic year
            return f"{now.year}-{now.year + 1}"
        else:
            return f"{now.year - 1}-{now.year}"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metadata to dictionary for JSON serialization."""
        return {
            "assignment_id": self.assignment_id,
            "subject": self.subject,
            "assignment_name": self.assignment_name,
            "due_date": self.due_date,
            "total_points": self.total_points,
            "class_period": self.class_period,
            "teacher_name": self.teacher_name,
            "academic_year": self.academic_year,
            "created_at": self.created_at,
            "qr_version": self.qr_version,
            "additional_data": self.additional_data
        }
    
    def to_json(self) -> str:
        """Convert metadata to JSON string."""
        return json.dumps(self.to_dict(), separators=(',', ':'))
    
    @classmethod
    def from_json(cls, json_str: str) -> 'AssignmentMetadata':
        """Create AssignmentMetadata from JSON string."""
        data = json.loads(json_str)
        return cls(
            assignment_id=data["assignment_id"],
            subject=data["subject"],
            assignment_name=data["assignment_name"],
            due_date=data["due_date"],
            total_points=data["total_points"],
            class_period=data.get("class_period"),
            teacher_name=data.get("teacher_name"),
            academic_year=data.get("academic_year"),
            additional_data=data.get("additional_data", {})
        )


class QRCodeGenerator:
    """Generates QR codes with assignment metadata."""
    
    def __init__(self):
        """Initialize QR code generator with configuration."""
        self.qr_config = config.get('qr', {})
        self.error_correction_map = {
            'L': ERROR_CORRECT_L,
            'M': ERROR_CORRECT_M,
            'Q': ERROR_CORRECT_Q,
            'H': ERROR_CORRECT_H
        }
    
    def generate_qr_code(
        self,
        metadata: AssignmentMetadata,
        size: Tuple[int, int] = (200, 200),
        include_label: bool = True
    ) -> Image.Image:
        """Generate QR code image with assignment metadata.
        
        Args:
            metadata: Assignment metadata to embed
            size: Target size for the QR code image
            include_label: Whether to include text label below QR code
            
        Returns:
            PIL Image containing the QR code
        """
        try:
            # Get QR code configuration
            version = self.qr_config.get('version', 1)
            error_correction = self.qr_config.get('error_correction', 'M')
            border = self.qr_config.get('border', 4)
            box_size = self.qr_config.get('box_size', 10)
            
            # Create QR code instance
            qr = qrcode.QRCode(
                version=version,
                error_correction=self.error_correction_map[error_correction],
                box_size=box_size,
                border=border,
            )
            
            # Add metadata as JSON
            json_data = metadata.to_json()
            qr.add_data(json_data)
            qr.make(fit=True)
            
            # Generate QR code image
            qr_img = qr.make_image(fill_color="black", back_color="white")
            
            # Convert to RGB if needed
            if qr_img.mode != 'RGB':
                qr_img = qr_img.convert('RGB')
            
            # Resize to requested size
            if qr_img.size != size:
                # Calculate scaling to fit the size while maintaining aspect ratio
                qr_w, qr_h = qr_img.size
                target_w, target_h = size
                
                # Create a new image with the target size and white background
                final_img = Image.new('RGB', size, 'white')
                
                # Scale QR code to fit within the target size
                scale = min(target_w / qr_w, target_h / qr_h) * 0.9  # 90% to leave some margin
                new_w, new_h = int(qr_w * scale), int(qr_h * scale)
                
                qr_resized = qr_img.resize((new_w, new_h), Image.Resampling.LANCZOS)
                
                # Center the QR code
                x = (target_w - new_w) // 2
                y = (target_h - new_h) // 2
                
                final_img.paste(qr_resized, (x, y))
                qr_img = final_img
            
            # Add label if requested
            if include_label:
                qr_img = self._add_label(qr_img, metadata)
            
            logger.info(f"Generated QR code for assignment: {metadata.assignment_id}")
            return qr_img
            
        except Exception as e:
            logger.error(f"Failed to generate QR code: {e}")
            raise
    
    def _add_label(self, qr_img: Image.Image, metadata: AssignmentMetadata) -> Image.Image:
        """Add text label below QR code.
        
        Args:
            qr_img: QR code image
            metadata: Assignment metadata for label text
            
        Returns:
            Image with QR code and label
        """
        # Calculate label dimensions
        label_height = 60
        total_height = qr_img.height + label_height
        
        # Create new image with space for label
        labeled_img = Image.new('RGB', (qr_img.width, total_height), 'white')
        labeled_img.paste(qr_img, (0, 0))
        
        # Add text
        draw = ImageDraw.Draw(labeled_img)
        
        try:
            # Try to load a font
            font_size = 12
            try:
                font = ImageFont.truetype("arial.ttf", font_size)
            except (OSError, IOError):
                try:
                    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", font_size)
                except (OSError, IOError):
                    font = ImageFont.load_default()
        except Exception:
            # Fallback to default font
            font = ImageFont.load_default()
        
        # Create label text
        label_text = f"{metadata.assignment_name}\n{metadata.subject.upper()} - {metadata.due_date}"
        
        # Center the text
        try:
            bbox = draw.textbbox((0, 0), label_text, font=font)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]
        except AttributeError:
            # Fallback for older PIL versions
            text_width, text_height = draw.textsize(label_text, font=font)
        
        x = (qr_img.width - text_width) // 2
        y = qr_img.height + (label_height - text_height) // 2
        
        draw.text((x, y), label_text, fill='black', font=font, align='center')
        
        return labeled_img
    
    def save_qr_code(
        self,
        metadata: AssignmentMetadata,
        output_path: Path,
        size: Tuple[int, int] = (200, 200),
        include_label: bool = True
    ) -> Path:
        """Generate and save QR code to file.
        
        Args:
            metadata: Assignment metadata to embed
            output_path: Path to save the QR code image
            size: Target size for the QR code image
            include_label: Whether to include text label
            
        Returns:
            Path to the saved file
        """
        qr_img = self.generate_qr_code(metadata, size, include_label)
        
        # Ensure output directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save image
        qr_img.save(output_path, format='PNG', optimize=True)
        logger.info(f"Saved QR code to: {output_path}")
        
        return output_path
    
    def generate_assignment_id(self, prefix: str = "ASN") -> str:
        """Generate a unique assignment ID.
        
        Args:
            prefix: Prefix for the assignment ID
            
        Returns:
            Unique assignment ID
        """
        timestamp = datetime.now().strftime("%Y%m%d")
        unique_id = str(uuid.uuid4())[:8].upper()
        return f"{prefix}-{timestamp}-{unique_id}"


class AssignmentTemplate:
    """Template for creating assignments with embedded QR codes."""
    
    def __init__(self, template_config: Dict[str, Any]):
        """Initialize assignment template.
        
        Args:
            template_config: Template configuration dictionary
        """
        self.name = template_config.get('name', 'Untitled Assignment')
        self.subject = template_config.get('subject', 'mathematics')
        self.total_points = template_config.get('total_points', 100)
        self.instructions = template_config.get('instructions', '')
        self.problems = template_config.get('problems', [])
        self.page_layout = template_config.get('page_layout', {})
        self.qr_position = template_config.get('qr_position', 'top-right')
    
    def create_assignment(
        self,
        assignment_name: str,
        due_date: str,
        class_period: Optional[str] = None,
        teacher_name: Optional[str] = None,
        assignment_id: Optional[str] = None
    ) -> AssignmentMetadata:
        """Create assignment metadata from template.
        
        Args:
            assignment_name: Name for this specific assignment
            due_date: Due date in ISO format
            class_period: Class period identifier
            teacher_name: Teacher's name
            assignment_id: Optional custom assignment ID
            
        Returns:
            AssignmentMetadata object
        """
        if assignment_id is None:
            generator = QRCodeGenerator()
            assignment_id = generator.generate_assignment_id()
        
        return AssignmentMetadata(
            assignment_id=assignment_id,
            subject=self.subject,
            assignment_name=assignment_name,
            due_date=due_date,
            total_points=self.total_points,
            class_period=class_period,
            teacher_name=teacher_name,
            additional_data={
                'template_name': self.name,
                'problems_count': len(self.problems),
                'instructions': self.instructions
            }
        )


def create_sample_assignment() -> AssignmentMetadata:
    """Create a sample assignment for testing purposes.
    
    Returns:
        Sample AssignmentMetadata object
    """
    generator = QRCodeGenerator()
    assignment_id = generator.generate_assignment_id("SAMPLE")
    
    return AssignmentMetadata(
        assignment_id=assignment_id,
        subject="algebra1",
        assignment_name="Quadratic Equations Practice",
        due_date="2025-06-20",
        total_points=100,
        class_period="Period 3",
        teacher_name="Ms. Johnson",
        additional_data={
            "difficulty_level": "intermediate",
            "estimated_time_minutes": 45,
            "topic": "quadratic_equations"
        }
    )


def demo_qr_generation():
    """Demonstrate QR code generation functionality."""
    print("🔲 QR Code Generation Demo")
    print("=" * 40)
    
    # Create sample assignment
    metadata = create_sample_assignment()
    print(f"📝 Assignment: {metadata.assignment_name}")
    print(f"🆔 ID: {metadata.assignment_id}")
    print(f"📚 Subject: {metadata.subject}")
    print(f"📅 Due: {metadata.due_date}")
    
    # Generate QR code
    generator = QRCodeGenerator()
    
    try:
        # Create output directory
        output_dir = Path("data/output/qr_codes")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate and save QR code
        output_path = output_dir / f"{metadata.assignment_id}.png"
        saved_path = generator.save_qr_code(metadata, output_path)
        
        print(f"✅ QR code generated successfully")
        print(f"💾 Saved to: {saved_path}")
        
        # Test metadata extraction
        json_data = metadata.to_json()
        restored_metadata = AssignmentMetadata.from_json(json_data)
        
        print(f"🔄 Metadata round-trip test: {'✅ PASSED' if restored_metadata.assignment_id == metadata.assignment_id else '❌ FAILED'}")
        
        # Test different sizes
        print("\n🔍 Testing different QR code sizes:")
        test_sizes = [(100, 100), (200, 200), (300, 300)]
        
        for size in test_sizes:
            try:
                qr_img = generator.generate_qr_code(metadata, size=size, include_label=False)
                size_test_path = output_dir / f"{metadata.assignment_id}_{size[0]}x{size[1]}.png"
                qr_img.save(size_test_path)
                print(f"   {size[0]}x{size[1]}: ✅ Generated (actual: {qr_img.size})")
            except Exception as e:
                print(f"   {size[0]}x{size[1]}: ❌ Error - {e}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False


if __name__ == "__main__":
    # Run demo if executed directly
    demo_qr_generation()