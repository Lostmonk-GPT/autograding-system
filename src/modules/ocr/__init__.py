#!/usr/bin/env python3
"""
AutoGrading System - OCR Module
Multi-engine OCR system for mathematical content recognition.
"""

from .ocr_manager import OCRManager, OCRResult
from .tesseract_ocr import TesseractOCR
from .easy_ocr import EasyOCREngine
from .math_corrector import MathCorrector
from .mock_pix2text import MockPix2TextEngine

__all__ = [
    'OCRManager',
    'OCRResult',
    'TesseractOCR',
    'EasyOCREngine',
    'MathCorrector',
    'MockPix2TextEngine'
]

# Version info
__version__ = '1.0.0'
__author__ = 'AutoGrading System'

# OCR Engine information
SUPPORTED_ENGINES = {
    'tesseract': {
        'name': 'Tesseract OCR',
        'description': 'Google Tesseract OCR optimized for mathematical content',
        'best_for': 'Printed mathematical expressions',
        'dependencies': ['pytesseract'],
        'system_requirements': ['tesseract binary']
    },
    'easyocr': {
        'name': 'EasyOCR',
        'description': 'Deep learning-based OCR for handwritten content',
        'best_for': 'Handwritten mathematical expressions',
        'dependencies': ['easyocr'],
        'system_requirements': []
    },
    'trocr': {
        'name': 'TrOCR (Transformer OCR)',
        'description': 'Microsoft TrOCR transformer model',
        'best_for': 'Complex document layouts',
        'dependencies': ['transformers', 'torch'],
        'system_requirements': []
    },
    'pix2text': {
        'name': 'Pix2Text (Mock)',
        'description': 'Mathematical formula OCR engine (mock implementation)',
        'best_for': 'Mathematical formulas and equations',
        'dependencies': [],
        'system_requirements': []
    }
}


def get_available_engines():
    """Get list of available OCR engines based on installed dependencies."""
    available = []

    # Check Tesseract
    try:
        import pytesseract
        available.append('tesseract')
    except ImportError:
        pass

    # Check EasyOCR
    try:
        import easyocr
        available.append('easyocr')
    except ImportError:
        pass

    # Check TrOCR
    try:
        from transformers import TrOCRProcessor
        available.append('trocr')
    except ImportError:
        pass

    # Mock Pix2Text is always available
    available.append('pix2text')

    return available


def get_engine_info(engine_name: str) -> dict:
    """Get information about a specific OCR engine."""
    return SUPPORTED_ENGINES.get(engine_name, {})


def create_default_ocr_manager():
    """Create OCR manager with default configuration."""
    return OCRManager()


def demo_all_engines():
    """Run demo for all available OCR engines."""
    print("🔍 OCR Module Demo - All Engines")
    print("=" * 40)

    available_engines = get_available_engines()
    print(f"Available engines: {', '.join(available_engines)}")

    # Create sample test image
    import numpy as np
    import cv2

    test_image = np.ones((100, 300), dtype=np.uint8) * 255
    cv2.putText(test_image, "2x + 3 = 7", (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, 0, 2)

    # Test each available engine
    results = {}

    for engine_name in available_engines:
        print(f"\n🧪 Testing {engine_name}...")

        try:
            if engine_name == 'tesseract':
                engine = TesseractOCR()
            elif engine_name == 'easyocr':
                engine = EasyOCREngine()
            elif engine_name == 'pix2text':
                engine = MockPix2TextEngine()
            else:
                continue

            text, confidence = engine.extract_text(test_image)
            results[engine_name] = {
                'text': text,
                'confidence': confidence,
                'success': True
            }

            print(f"✅ {engine_name}: '{text}' (confidence: {confidence:.2f})")

        except Exception as e:
            results[engine_name] = {
                'error': str(e),
                'success': False
            }
            print(f"❌ {engine_name}: {e}")

    # Test complete OCR manager
    print(f"\n🔄 Testing complete OCR manager...")

    try:
        manager = OCRManager()
        ocr_result = manager.extract_text(test_image)

        if ocr_result['success']:
            print(
                f"✅ OCR Manager: '{ocr_result['final_text']}' (confidence: {ocr_result['confidence']:.2f})")
            print(f"   Engines used: {', '.join(ocr_result['engines_used'])}")
        else:
            print(f"❌ OCR Manager failed: {ocr_result.get('error_message')}")

    except Exception as e:
        print(f"❌ OCR Manager error: {e}")

    # Test mathematical corrector
    print(f"\n🔧 Testing mathematical symbol corrector...")

    try:
        corrector = MathCorrector()
        test_texts = ["2x + 3 = l", "O + 5 = 5", "x^Z = 4"]

        for test_text in test_texts:
            corrected = corrector.correct_mathematical_text(test_text)
            print(f"   '{test_text}' → '{corrected}'")

    except Exception as e:
        print(f"❌ Math corrector error: {e}")

    print(f"\n🎉 OCR module demo completed!")
    return results


# Module-level constants
DEFAULT_OCR_CONFIG = {
    'engines': {
        'tesseract': {
            'enabled': True,
            'priority': 1,
            'math_config': '--psm 6 -c tessedit_char_whitelist=0123456789+-*/=().,ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'
        },
        'easyocr': {
            'enabled': True,
            'priority': 2,
            'languages': ['en']
        },
        'pix2text': {
            'enabled': True,
            'priority': 1
        }
    },
    'processing': {
        'parallel_execution': True,
        'max_workers': 3,
        'confidence_threshold': 0.7,
        'aggregation_method': 'weighted_vote',
        'symbol_correction': True
    }
}
