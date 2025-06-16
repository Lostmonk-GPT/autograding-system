"""
Assignment Template Manager for AutoGrading System
"""

import yaml
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime
import logging

from .qr_generator import AssignmentMetadata, QRCodeGenerator

logger = logging.getLogger(__name__)


class TemplateManager:
    """Manages assignment templates."""
    
    def __init__(self, templates_file: Optional[Path] = None):
        if templates_file is None:
            templates_file = Path("config/assignment_templates.yaml")
        
        self.templates_file = templates_file
        self.templates = self._load_templates()
        self.qr_generator = QRCodeGenerator()
    
    def _load_templates(self) -> Dict[str, Any]:
        """Load templates from YAML file."""
        try:
            if self.templates_file.exists():
                with open(self.templates_file, 'r') as f:
                    data = yaml.safe_load(f)
                return data.get('templates', {})
            else:
                logger.warning(f"Templates file not found: {self.templates_file}")
                return self._get_default_templates()
        except yaml.YAMLError as e:
            logger.error(f"Error loading templates: {e}")
            return self._get_default_templates()
    
    def _get_default_templates(self) -> Dict[str, Any]:
        """Return default templates."""
        return {
            "algebra1_basic": {
                "name": "Algebra 1 - Basic Practice",
                "subject": "algebra1",
                "total_points": 100,
                "estimated_time_minutes": 45,
                "difficulty_level": "beginner",
                "instructions": "Solve all problems showing your work clearly.",
                "problem_types": [
                    {"type": "linear_equations", "count": 5, "points_each": 10},
                    {"type": "word_problems", "count": 3, "points_each": 10}
                ],
                "grading_rubric": {
                    "work_shown": 40,
                    "correct_answer": 50,
                    "presentation": 10
                }
            }
        }
    
    def list_templates(self, subject: Optional[str] = None) -> List[Dict[str, Any]]:
        """List available templates."""
        template_list = []
        
        for template_id, template_data in self.templates.items():
            if subject is None or template_data.get('subject') == subject:
                summary = {
                    'id': template_id,
                    'name': template_data.get('name', template_id),
                    'subject': template_data.get('subject', 'unknown'),
                    'total_points': template_data.get('total_points', 100),
                    'difficulty': template_data.get('difficulty_level', 'intermediate'),
                    'estimated_time': template_data.get('estimated_time_minutes', 60)
                }
                template_list.append(summary)
        
        return sorted(template_list, key=lambda x: (x['subject'], x['difficulty'], x['name']))
    
    def get_subjects(self) -> List[str]:
        """Get list of available subjects."""
        subjects = set()
        for template in self.templates.values():
            subject = template.get('subject')
            if subject:
                subjects.add(subject)
        return sorted(list(subjects))
    
    def create_assignment_from_template(
        self,
        template_id: str,
        assignment_name: str,
        due_date: str,
        class_period: Optional[str] = None,
        teacher_name: Optional[str] = None,
        custom_points: Optional[int] = None
    ) -> AssignmentMetadata:
        """Create assignment from template."""
        template = self.templates.get(template_id)
        if template is None:
            raise ValueError(f"Template '{template_id}' not found")
        
        # Validate due date
        try:
            datetime.strptime(due_date, "%Y-%m-%d")
        except ValueError:
            raise ValueError("Due date must be in YYYY-MM-DD format")
        
        # Generate assignment ID
        prefix = template.get('subject', 'ASN').upper()[:3]
        assignment_id = self.qr_generator.generate_assignment_id(prefix)
        
        total_points = custom_points or template.get('total_points', 100)
        
        additional_data = {
            'template_id': template_id,
            'template_name': template.get('name', template_id),
            'difficulty_level': template.get('difficulty_level', 'intermediate'),
            'estimated_time_minutes': template.get('estimated_time_minutes', 60),
            'instructions': template.get('instructions', ''),
            'problem_types': template.get('problem_types', []),
            'grading_rubric': template.get('grading_rubric', {})
        }
        
        metadata = AssignmentMetadata(
            assignment_id=assignment_id,
            subject=template.get('subject', 'mathematics'),
            assignment_name=assignment_name,
            due_date=due_date,
            total_points=total_points,
            class_period=class_period,
            teacher_name=teacher_name,
            additional_data=additional_data
        )
        
        logger.info(f"Created assignment from template '{template_id}': {assignment_name}")
        return metadata


class AssignmentBuilder:
    """Helper class for building assignments."""
    
    def __init__(self, template_manager: TemplateManager):
        self.template_manager = template_manager
        self.reset()
    
    def reset(self):
        """Reset builder."""
        self._template_id = None
        self._assignment_name = None
        self._due_date = None
        self._class_period = None
        self._teacher_name = None
        self._custom_points = None
    
    def use_template(self, template_id: str):
        """Set template to use."""
        self._template_id = template_id
        return self
    
    def set_name(self, assignment_name: str):
        """Set assignment name."""
        self._assignment_name = assignment_name
        return self
    
    def set_due_date(self, due_date: str):
        """Set due date."""
        self._due_date = due_date
        return self
    
    def set_class_period(self, class_period: str):
        """Set class period."""
        self._class_period = class_period
        return self
    
    def set_teacher(self, teacher_name: str):
        """Set teacher name."""
        self._teacher_name = teacher_name
        return self
    
    def set_points(self, total_points: int):
        """Set total points."""
        self._custom_points = total_points
        return self
    
    def build(self) -> AssignmentMetadata:
        """Build the assignment."""
        if self._template_id is None:
            raise ValueError("Template ID is required")
        if self._assignment_name is None:
            raise ValueError("Assignment name is required")
        if self._due_date is None:
            raise ValueError("Due date is required")
        
        return self.template_manager.create_assignment_from_template(
            template_id=self._template_id,
            assignment_name=self._assignment_name,
            due_date=self._due_date,
            class_period=self._class_period,
            teacher_name=self._teacher_name,
            custom_points=self._custom_points
        )


def demo_template_system():
    """Demo the template system."""
    print("📋 Assignment Template System Demo")
    print("=" * 50)
    
    template_manager = TemplateManager()
    
    print("📚 Available Templates:")
    templates = template_manager.list_templates()
    for template in templates:
        print(f"   {template['id']}: {template['name']} ({template['subject']})")
    
    print(f"\n📊 Subjects: {', '.join(template_manager.get_subjects())}")
    
    try:
        builder = AssignmentBuilder(template_manager)
        assignment = (builder
                     .use_template("algebra1_basic")
                     .set_name("Linear Equations Test")
                     .set_due_date("2025-06-25")
                     .set_class_period("Period 3")
                     .set_teacher("Ms. Johnson")
                     .build())
        
        print(f"\n✅ Created Assignment:")
        print(f"   ID: {assignment.assignment_id}")
        print(f"   Name: {assignment.assignment_name}")
        print(f"   Subject: {assignment.subject}")
        print(f"   Points: {assignment.total_points}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False


if __name__ == "__main__":
    demo_template_system()
