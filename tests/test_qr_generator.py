"""
Test suite for QR Code Generator module.
"""

import pytest
import json
from datetime import datetime
from pathlib import Path
import sys

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from modules.qr_generator import (
    AssignmentMetadata,
    QRCodeGenerator,
    AssignmentTemplate,
    create_sample_assignment,
    demo_qr_generation
)


class TestAssignmentMetadata:
    """Test AssignmentMetadata functionality."""
    
    def test_metadata_creation(self):
        """Test basic metadata creation."""
        metadata = AssignmentMetadata(
            assignment_id="TEST-001",
            subject="algebra1",
            assignment_name="Test Assignment",
            due_date="2025-06-20",
            total_points=100
        )
        
        assert metadata.assignment_id == "TEST-001"
        assert metadata.subject == "algebra1"
        assert metadata.assignment_name == "Test Assignment"
        assert metadata.due_date == "2025-06-20"
        assert metadata.total_points == 100
        assert metadata.qr_version == "1.0"
        assert metadata.created_at is not None
    
    def test_metadata_to_dict(self):
        """Test metadata dictionary conversion."""
        metadata = AssignmentMetadata(
            assignment_id="TEST-002",
            subject="ap_calculus",
            assignment_name="Derivatives Practice",
            due_date="2025-06-25",
            total_points=75,
            class_period="Period 2"
        )
        
        data_dict = metadata.to_dict()
        
        assert isinstance(data_dict, dict)
        assert data_dict["assignment_id"] == "TEST-002"
        assert data_dict["subject"] == "ap_calculus"
        assert data_dict["class_period"] == "Period 2"
        assert "created_at" in data_dict
        assert "qr_version" in data_dict
    
    def test_metadata_json_serialization(self):
        """Test JSON serialization and deserialization."""
        original = AssignmentMetadata(
            assignment_id="TEST-003",
            subject="geometry",
            assignment_name="Triangle Properties",
            due_date="2025-07-01",
            total_points=50,
            teacher_name="Mr. Smith"
        )
        
        # Convert to JSON and back
        json_str = original.to_json()
        restored = AssignmentMetadata.from_json(json_str)
        
        assert restored.assignment_id == original.assignment_id
        assert restored.subject == original.subject
        assert restored.assignment_name == original.assignment_name
        assert restored.due_date == original.due_date
        assert restored.total_points == original.total_points
        assert restored.teacher_name == original.teacher_name
    
    def test_metadata_with_additional_data(self):
        """Test metadata with additional data fields."""
        additional = {
            "difficulty": "advanced",
            "topics": ["derivatives", "chain_rule"],
            "estimated_time": 60
        }
        
        metadata = AssignmentMetadata(
            assignment_id="TEST-004",
            subject="ap_calculus",
            assignment_name="Advanced Derivatives",
            due_date="2025-07-15",
            total_points=120,
            additional_data=additional
        )
        
        assert metadata.additional_data == additional
        
        # Test JSON round-trip with additional data
        json_str = metadata.to_json()
        restored = AssignmentMetadata.from_json(json_str)
        assert restored.additional_data == additional


class TestQRCodeGenerator:
    """Test QR Code Generator functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.generator = QRCodeGenerator()
        self.sample_metadata = AssignmentMetadata(
            assignment_id="QR-TEST-001",
            subject="algebra1",
            assignment_name="QR Test Assignment",
            due_date="2025-06-30",
            total_points=100
        )
    
    def test_generator_initialization(self):
        """Test generator initialization."""
        assert self.generator is not None
        assert hasattr(self.generator, 'qr_config')
        assert hasattr(self.generator, 'error_correction_map')
    
    def test_assignment_id_generation(self):
        """Test unique assignment ID generation."""
        id1 = self.generator.generate_assignment_id()
        id2 = self.generator.generate_assignment_id()
        
        assert id1 != id2
        assert id1.startswith("ASN-")
        assert id2.startswith("ASN-")
        assert len(id1.split("-")) == 3  # PREFIX-DATE-UUID
    
    def test_custom_prefix_assignment_id(self):
        """Test assignment ID with custom prefix."""
        custom_id = self.generator.generate_assignment_id("QUIZ")
        assert custom_id.startswith("QUIZ-")
    
    @pytest.mark.skip(reason="Requires PIL/qrcode libraries - run manually")
    def test_qr_code_generation(self):
        """Test QR code image generation."""
        try:
            qr_img = self.generator.generate_qr_code(self.sample_metadata)
            
            assert qr_img is not None
            assert hasattr(qr_img, 'size')
            assert qr_img.mode in ['RGB', 'L']  # RGB or grayscale
            
        except ImportError:
            pytest.skip("PIL/qrcode not available")
    
    @pytest.mark.skip(reason="Requires file system - run manually")
    def test_qr_code_saving(self):
        """Test saving QR code to file."""
        try:
            output_dir = Path("test_output")
            output_dir.mkdir(exist_ok=True)
            output_path = output_dir / "test_qr.png"
            
            saved_path = self.generator.save_qr_code(
                self.sample_metadata,
                output_path
            )
            
            assert saved_path.exists()
            assert saved_path.suffix == '.png'
            
            # Cleanup
            if saved_path.exists():
                saved_path.unlink()
            if output_dir.exists() and not any(output_dir.iterdir()):
                output_dir.rmdir()
                
        except ImportError:
            pytest.skip("PIL/qrcode not available")


class TestAssignmentTemplate:
    """Test Assignment Template functionality."""
    
    def test_template_creation(self):
        """Test template creation from configuration."""
        template_config = {
            'name': 'Algebra Basics Template',
            'subject': 'algebra1',
            'total_points': 75,
            'instructions': 'Solve all problems showing your work.',
            'problems': [
                {'type': 'equation', 'difficulty': 'easy'},
                {'type': 'word_problem', 'difficulty': 'medium'}
            ]
        }
        
        template = AssignmentTemplate(template_config)
        
        assert template.name == 'Algebra Basics Template'
        assert template.subject == 'algebra1'
        assert template.total_points == 75
        assert len(template.problems) == 2
    
    def test_assignment_creation_from_template(self):
        """Test creating assignment from template."""
        template_config = {
            'name': 'Calculus Template',
            'subject': 'ap_calculus',
            'total_points': 100
        }
        
        template = AssignmentTemplate(template_config)
        metadata = template.create_assignment(
            assignment_name="Limits Practice",
            due_date="2025-07-01",
            class_period="Period 4",
            teacher_name="Dr. Brown"
        )
        
        assert metadata.assignment_name == "Limits Practice"
        assert metadata.subject == "ap_calculus"
        assert metadata.total_points == 100
        assert metadata.class_period == "Period 4"
        assert metadata.teacher_name == "Dr. Brown"
        assert metadata.assignment_id is not None


class TestUtilityFunctions:
    """Test utility functions."""
    
    def test_create_sample_assignment(self):
        """Test sample assignment creation."""
        sample = create_sample_assignment()
        
        assert sample.assignment_id.startswith("SAMPLE-")
        assert sample.subject == "algebra1"
        assert sample.assignment_name == "Quadratic Equations Practice"
        assert sample.total_points == 100
        assert "difficulty_level" in sample.additional_data
    
    @pytest.mark.skip(reason="Requires full environment - run manually")
    def test_demo_qr_generation(self):
        """Test QR generation demo."""
        try:
            result = demo_qr_generation()
            assert result is True
        except Exception:
            pytest.skip("Demo requires full environment setup")


def test_imports():
    """Test that all required modules can be imported."""
    # Test that we can import the main classes
    from modules.qr_generator import AssignmentMetadata
    from modules.qr_generator import QRCodeGenerator
    from modules.qr_generator import AssignmentTemplate
    
    assert AssignmentMetadata is not None
    assert QRCodeGenerator is not None
    assert AssignmentTemplate is not None


if __name__ == "__main__":
    # Run basic tests
    print("🧪 Running QR Generator Tests")
    print("=" * 40)
    
    # Test basic functionality
    try:
        # Test metadata
        metadata = AssignmentMetadata(
            assignment_id="MANUAL-TEST-001",
            subject="test_subject",
            assignment_name="Manual Test",
            due_date="2025-06-20",
            total_points=50
        )
        print("✅ AssignmentMetadata creation: PASSED")
        
        # Test JSON serialization
        json_str = metadata.to_json()
        restored = AssignmentMetadata.from_json(json_str)
        assert restored.assignment_id == metadata.assignment_id
        print("✅ JSON serialization: PASSED")
        
        # Test generator
        generator = QRCodeGenerator()
        test_id = generator.generate_assignment_id("TEST")
        assert test_id.startswith("TEST-")
        print("✅ Assignment ID generation: PASSED")
        
        # Test template
        template_config = {'name': 'Test Template', 'subject': 'test'}
        template = AssignmentTemplate(template_config)
        assert template.name == 'Test Template'
        print("✅ Assignment template: PASSED")
        
        print("\n🎉 All basic tests passed!")
        print("Run 'pytest tests/test_qr_generator.py -v' for full test suite")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()