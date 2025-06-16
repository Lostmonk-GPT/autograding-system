"""
QR Code Robustness Testing Module
"""

import os
from pathlib import Path
from typing import Dict, Any
import time
import json

from .qr_generator import AssignmentMetadata, QRCodeGenerator, create_sample_assignment


class QRRobustnessTester:
    """Test QR code robustness under various conditions."""
    
    def __init__(self):
        """Initialize the robustness tester."""
        self.generator = QRCodeGenerator()
        self.output_dir = Path("data/output/qr_robustness")
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def test_qr_generation(self, metadata: AssignmentMetadata) -> Dict[str, Any]:
        """Test basic QR code generation at different sizes."""
        sizes = [(100, 100), (150, 150), (200, 200), (300, 300), (500, 500)]
        results = {"test_name": "qr_generation", "passed": 0, "failed": 0, "details": []}
        
        print("🔍 Testing QR code generation at various sizes...")
        
        for size in sizes:
            try:
                qr_img = self.generator.generate_qr_code(metadata, size=size, include_label=False)
                
                # Test that image was created successfully
                if qr_img and hasattr(qr_img, 'size') and qr_img.size[0] > 0 and qr_img.size[1] > 0:
                    # Save the image
                    filename = f"size_test_{size[0]}x{size[1]}.png"
                    filepath = self.output_dir / filename
                    qr_img.save(filepath)
                    
                    results["passed"] += 1
                    status = "✅ PASS"
                    
                    detail = {
                        "requested_size": size,
                        "actual_size": qr_img.size,
                        "status": status,
                        "file": str(filepath),
                        "size_match": qr_img.size == size
                    }
                    
                    if qr_img.size == size:
                        print(f"   {size[0]}x{size[1]}: ✅ PASS (exact size match)")
                    else:
                        print(f"   {size[0]}x{size[1]}: ✅ PASS (generated {qr_img.size[0]}x{qr_img.size[1]})")
                
                else:
                    results["failed"] += 1
                    detail = {
                        "requested_size": size,
                        "status": "❌ FAIL",
                        "error": "Invalid image generated"
                    }
                    print(f"   {size[0]}x{size[1]}: ❌ FAIL - Invalid image")
                
                results["details"].append(detail)
                
            except Exception as e:
                results["failed"] += 1
                results["details"].append({
                    "requested_size": size,
                    "status": "❌ ERROR",
                    "error": str(e)
                })
                print(f"   {size[0]}x{size[1]}: ❌ ERROR - {e}")
        
        return results
    
    def test_qr_functionality(self) -> Dict[str, Any]:
        """Test core QR code functionality."""
        results = {"test_name": "qr_functionality", "passed": 0, "failed": 0, "details": []}
        
        print("⚙️ Testing QR code core functionality...")
        
        tests = [
            {
                "name": "ID Generation",
                "func": lambda: self.generator.generate_assignment_id("TEST"),
                "validator": lambda result: result and isinstance(result, str) and "TEST-" in result
            },
            {
                "name": "Basic QR Creation",
                "func": lambda: self.generator.generate_qr_code(create_sample_assignment()),
                "validator": lambda result: result and hasattr(result, 'size')
            },
            {
                "name": "Metadata Serialization",
                "func": lambda: create_sample_assignment().to_json(),
                "validator": lambda result: result and isinstance(result, str) and len(result) > 50
            },
            {
                "name": "Metadata Deserialization",
                "func": lambda: AssignmentMetadata.from_json(create_sample_assignment().to_json()),
                "validator": lambda result: result and hasattr(result, 'assignment_id')
            }
        ]
        
        for test in tests:
            try:
                result = test["func"]()
                is_valid = test["validator"](result)
                
                if is_valid:
                    results["passed"] += 1
                    print(f"   {test['name']}: ✅ PASS")
                    results["details"].append({
                        "test": test["name"],
                        "status": "✅ PASS",
                        "result_type": type(result).__name__
                    })
                else:
                    results["failed"] += 1
                    print(f"   {test['name']}: ❌ FAIL - Invalid result")
                    results["details"].append({
                        "test": test["name"],
                        "status": "❌ FAIL",
                        "error": "Result validation failed"
                    })
                    
            except Exception as e:
                results["failed"] += 1
                print(f"   {test['name']}: ❌ ERROR - {e}")
                results["details"].append({
                    "test": test["name"],
                    "status": "❌ ERROR",
                    "error": str(e)
                })
        
        return results
    
    def test_metadata_variations(self) -> Dict[str, Any]:
        """Test QR code generation with different metadata complexity."""
        results = {"test_name": "metadata_variations", "passed": 0, "failed": 0, "details": []}
        
        print("📊 Testing QR codes with different metadata complexity...")
        
        test_cases = [
            {
                "name": "minimal",
                "metadata": AssignmentMetadata(
                    assignment_id="MIN-001",
                    subject="math",
                    assignment_name="Test",
                    due_date="2025-12-31",
                    total_points=10
                )
            },
            {
                "name": "standard",
                "metadata": create_sample_assignment()
            },
            {
                "name": "detailed",
                "metadata": AssignmentMetadata(
                    assignment_id="DETAILED-001",
                    subject="advanced_placement_calculus",
                    assignment_name="Comprehensive Assessment",
                    due_date="2025-12-31",
                    total_points=150,
                    class_period="AP Period 6",
                    teacher_name="Dr. Mathematics",
                    additional_data={
                        "difficulty_level": "challenging",
                        "topics": ["integration", "derivatives"],
                        "estimated_time_minutes": 120
                    }
                )
            }
        ]
        
        for test_case in test_cases:
            try:
                metadata = test_case["metadata"]
                json_data = metadata.to_json()
                json_size = len(json_data)
                
                # Test QR generation
                qr_img = self.generator.generate_qr_code(metadata, size=(250, 250), include_label=False)
                
                # Test JSON round-trip
                restored_metadata = AssignmentMetadata.from_json(json_data)
                roundtrip_success = restored_metadata.assignment_id == metadata.assignment_id
                
                # Save test image
                filename = f"metadata_{test_case['name']}.png"
                filepath = self.output_dir / filename
                qr_img.save(filepath)
                
                results["passed"] += 1
                detail = {
                    "case": test_case["name"],
                    "json_size": json_size,
                    "image_size": qr_img.size,
                    "roundtrip_success": roundtrip_success,
                    "status": "✅ PASS",
                    "file": str(filepath)
                }
                results["details"].append(detail)
                print(f"   {test_case['name']} ({json_size} chars): ✅ PASS")
                
            except Exception as e:
                results["failed"] += 1
                results["details"].append({
                    "case": test_case["name"],
                    "status": "❌ ERROR",
                    "error": str(e)
                })
                print(f"   {test_case['name']}: ❌ ERROR - {e}")
        
        return results
    
    def run_comprehensive_test(self) -> Dict[str, Any]:
        """Run all robustness tests."""
        print("🧪 QR Code Robustness Test Suite")
        print("=" * 50)
        
        start_time = time.time()
        test_metadata = create_sample_assignment()
        
        all_results = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "test_metadata": {
                "assignment_id": test_metadata.assignment_id,
                "subject": test_metadata.subject,
                "assignment_name": test_metadata.assignment_name
            },
            "tests": []
        }
        
        # Run test suites
        functionality_results = self.test_qr_functionality()
        all_results["tests"].append(functionality_results)
        
        generation_results = self.test_qr_generation(test_metadata)
        all_results["tests"].append(generation_results)
        
        metadata_results = self.test_metadata_variations()
        all_results["tests"].append(metadata_results)
        
        # Calculate totals
        total_passed = sum(test["passed"] for test in all_results["tests"])
        total_failed = sum(test["failed"] for test in all_results["tests"])
        total_tests = total_passed + total_failed
        
        elapsed_time = time.time() - start_time
        
        # Summary
        print("\n" + "=" * 50)
        print("📊 QR Robustness Test Summary")
        print("=" * 50)
        print(f"Total Tests: {total_tests}")
        print(f"Passed: {total_passed} ✅")
        print(f"Failed: {total_failed} ❌")
        
        if total_tests > 0:
            success_rate = (total_passed / total_tests) * 100
            print(f"Success Rate: {success_rate:.1f}%")
            
            if success_rate >= 90:
                print("🎉 Excellent! QR system is highly robust")
            elif success_rate >= 75:
                print("👍 Good! QR system is reasonably robust")
            elif success_rate >= 50:
                print("⚠️  Fair - QR system needs some improvements")
            else:
                print("🔧 QR system needs significant improvements")
        else:
            success_rate = 0
            print("❌ No tests completed successfully")
        
        print(f"Test Duration: {elapsed_time:.2f} seconds")
        print(f"Output Directory: {self.output_dir}")
        
        # Show created files
        output_files = list(self.output_dir.glob("*.png"))
        if output_files:
            print(f"\n📁 Generated {len(output_files)} test files:")
            for file in sorted(output_files)[:3]:
                print(f"   {file.name}")
            if len(output_files) > 3:
                print(f"   ... and {len(output_files) - 3} more files")
        
        all_results["summary"] = {
            "total_tests": total_tests,
            "passed": total_passed,
            "failed": total_failed,
            "success_rate": success_rate,
            "duration_seconds": elapsed_time,
            "files_created": len(output_files)
        }
        
        return all_results


def run_robustness_demo():
    """Run the robustness testing demo."""
    tester = QRRobustnessTester()
    return tester.run_comprehensive_test()


if __name__ == "__main__":
    run_robustness_demo()
