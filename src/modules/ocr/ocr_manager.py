#!/usr/bin/env python3
"""
AutoGrading System - OCR Engine Manager
Orchestrates multiple OCR engines for optimal text extraction from mathematical assignments.
"""

import cv2
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import json

# Setup logging
logger = logging.getLogger(__name__)


class OCRResult:
    """Container for OCR engine results."""

    def __init__(self, engine_name: str, text: str, confidence: float = 0.0,
                 processing_time: float = 0.0, error_message: str = None):
        self.engine_name = engine_name
        self.text = text
        self.confidence = confidence
        self.processing_time = processing_time
        self.error_message = error_message
        self.success = error_message is None
        self.character_count = len(text) if text else 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary format."""
        return {
            'engine_name': self.engine_name,
            'text': self.text,
            'confidence': self.confidence,
            'processing_time': self.processing_time,
            'success': self.success,
            'error_message': self.error_message,
            'character_count': self.character_count
        }


class OCRManager:
    """Manages multiple OCR engines for mathematical text extraction."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize OCR manager with configuration.

        Args:
            config: Configuration dictionary for OCR engines and processing
        """
        self.config = config or self._default_config()
        self.engines = {}
        self.engine_stats = {}
        self._init_engines()

    def _default_config(self) -> Dict[str, Any]:
        """Get default configuration settings."""
        return {
            'engines': {
                'tesseract': {
                    'enabled': True,
                    'priority': 1,
                    'math_config': '--psm 6 -c tessedit_char_whitelist=0123456789+-*/=().,ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz',
                    'timeout': 30
                },
                'easyocr': {
                    'enabled': True,
                    'priority': 2,
                    'languages': ['en'],
                    'timeout': 45
                },
                'trocr': {
                    'enabled': False,  # Heavy model, enable when needed
                    'priority': 3,
                    'model': 'microsoft/trocr-base-printed',
                    'timeout': 60
                },
                'pix2text': {
                    'enabled': False,  # Disabled for real testing
                    'priority': 1,  # Best for mathematical formulas
                    'timeout': 45
                }
            },
            'processing': {
                'parallel_execution': True,
                'max_workers': 3,
                'confidence_threshold': 0.7,
                'aggregation_method': 'weighted_vote',
                'symbol_correction': True
            },
            'quality_thresholds': {
                'min_confidence': 0.5,
                'min_character_count': 10,
                'max_processing_time': 120
            }
        }

    def _init_engines(self):
        """Initialize available OCR engines."""
        engine_configs = self.config['engines']

        for engine_name, engine_config in engine_configs.items():
            if not engine_config.get('enabled', False):
                continue

            try:
                if engine_name == 'tesseract':
                    self._init_tesseract()
                elif engine_name == 'easyocr':
                    self._init_easyocr()
                elif engine_name == 'trocr':
                    self._init_trocr()
                elif engine_name == 'pix2text':
                    self._init_pix2text()

                self.engine_stats[engine_name] = {
                    'total_calls': 0,
                    'successful_calls': 0,
                    'total_time': 0.0,
                    'average_confidence': 0.0
                }

                logger.info(f"Initialized OCR engine: {engine_name}")

            except Exception as e:
                logger.warning(f"Failed to initialize {engine_name}: {e}")

    def _init_tesseract(self):
        """Initialize Tesseract OCR engine."""
        try:
            import pytesseract

            # Test Tesseract installation
            version = pytesseract.get_tesseract_version()
            logger.info(f"Tesseract version: {version}")

            from .tesseract_ocr import TesseractOCR
            self.engines['tesseract'] = TesseractOCR(
                self.config['engines']['tesseract'])

        except ImportError:
            raise ImportError(
                "pytesseract not installed. Install with: pip install pytesseract")
        except Exception as e:
            raise RuntimeError(f"Tesseract initialization failed: {e}")

    def _init_easyocr(self):
        """Initialize EasyOCR engine."""
        try:
            import easyocr

            from .easy_ocr import EasyOCREngine
            self.engines['easyocr'] = EasyOCREngine(
                self.config['engines']['easyocr'])

        except ImportError:
            raise ImportError(
                "easyocr not installed. Install with: pip install easyocr")
        except Exception as e:
            raise RuntimeError(f"EasyOCR initialization failed: {e}")

    def _init_trocr(self):
        """Initialize TrOCR transformer model."""
        try:
            from transformers import TrOCRProcessor, VisionEncoderDecoderModel

            from .transformer_ocr import TrOCREngine
            self.engines['trocr'] = TrOCREngine(
                self.config['engines']['trocr'])

        except ImportError:
            raise ImportError(
                "transformers not installed. Install with: pip install transformers torch")
        except Exception as e:
            raise RuntimeError(f"TrOCR initialization failed: {e}")

    def _init_pix2text(self):
        """Initialize Pix2Text mathematical formula OCR."""
        try:
            from .pix2text_ocr import Pix2TextEngine
            self.engines['pix2text'] = Pix2TextEngine(
                self.config['engines']['pix2text'])

        except ImportError:
            logger.warning(
                "Pix2Text not available. Mathematical formula OCR will be limited.")
            # Create a mock engine for development
            from .mock_pix2text import MockPix2TextEngine
            self.engines['pix2text'] = MockPix2TextEngine(
                self.config['engines']['pix2text'])
        except Exception as e:
            raise RuntimeError(f"Pix2Text initialization failed: {e}")

    def extract_text(self, image: np.ndarray, image_path: str = None) -> Dict[str, Any]:
        """Extract text from image using multiple OCR engines.

        Args:
            image: Input image as numpy array
            image_path: Optional path to source image for logging

        Returns:
            Dictionary containing aggregated OCR results
        """
        start_time = time.time()

        try:
            # Validate input
            if image is None or image.size == 0:
                raise ValueError("Invalid input image")

            # Get enabled engines sorted by priority
            enabled_engines = self._get_enabled_engines()

            if not enabled_engines:
                raise RuntimeError("No OCR engines available")

            # Process with engines
            if self.config['processing']['parallel_execution'] and len(enabled_engines) > 1:
                results = self._process_parallel(image, enabled_engines)
            else:
                results = self._process_sequential(image, enabled_engines)

            # Aggregate results
            aggregated = self._aggregate_results(results)

            # Apply symbol correction if enabled
            if self.config['processing']['symbol_correction']:
                aggregated = self._apply_symbol_correction(aggregated)

            # Calculate overall metrics
            total_time = time.time() - start_time

            return {
                'success': True,
                'final_text': aggregated.get('text', ''),
                'confidence': aggregated.get('confidence', 0.0),
                'processing_time': total_time,
                'engine_results': [r.to_dict() for r in results],
                'aggregation_method': self.config['processing']['aggregation_method'],
                'engines_used': [r.engine_name for r in results if r.success],
                'source_file': image_path,
                'character_count': len(aggregated.get('text', '')),
                'quality_metrics': self._calculate_quality_metrics(results, aggregated)
            }

        except Exception as e:
            logger.error(f"OCR extraction failed: {e}")
            return {
                'success': False,
                'final_text': '',
                'confidence': 0.0,
                'processing_time': time.time() - start_time,
                'engine_results': [],
                'error_message': str(e),
                'source_file': image_path
            }

    def _get_enabled_engines(self) -> List[str]:
        """Get list of enabled engines sorted by priority."""
        engine_configs = self.config['engines']
        enabled = []

        for engine_name, config in engine_configs.items():
            if config.get('enabled', False) and engine_name in self.engines:
                priority = config.get('priority', 999)
                enabled.append((priority, engine_name))

        # Sort by priority (lower number = higher priority)
        enabled.sort(key=lambda x: x[0])
        return [name for _, name in enabled]

    def _process_parallel(self, image: np.ndarray, engine_names: List[str]) -> List[OCRResult]:
        """Process image with multiple engines in parallel."""
        results = []
        max_workers = min(len(engine_names),
                          self.config['processing']['max_workers'])

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit tasks
            future_to_engine = {}
            for engine_name in engine_names:
                future = executor.submit(
                    self._run_single_engine, image, engine_name)
                future_to_engine[future] = engine_name

            # Collect results
            for future in as_completed(future_to_engine):
                engine_name = future_to_engine[future]
                try:
                    result = future.result()
                    results.append(result)
                    logger.debug(
                        f"Completed OCR with {engine_name}: {result.confidence:.2f} confidence")
                except Exception as e:
                    error_result = OCRResult(engine_name, '', 0.0, 0.0, str(e))
                    results.append(error_result)
                    logger.warning(f"OCR engine {engine_name} failed: {e}")

        return results

    def _process_sequential(self, image: np.ndarray, engine_names: List[str]) -> List[OCRResult]:
        """Process image with engines sequentially."""
        results = []

        for engine_name in engine_names:
            try:
                result = self._run_single_engine(image, engine_name)
                results.append(result)
                logger.debug(
                    f"Completed OCR with {engine_name}: {result.confidence:.2f} confidence")

                # Early exit if we get high confidence result
                if result.confidence >= 0.9:
                    logger.info(
                        f"High confidence result from {engine_name}, skipping remaining engines")
                    break

            except Exception as e:
                error_result = OCRResult(engine_name, '', 0.0, 0.0, str(e))
                results.append(error_result)
                logger.warning(f"OCR engine {engine_name} failed: {e}")

        return results

    def _run_single_engine(self, image: np.ndarray, engine_name: str) -> OCRResult:
        """Run OCR with a single engine."""
        start_time = time.time()

        try:
            engine = self.engines[engine_name]
            timeout = self.config['engines'][engine_name].get('timeout', 60)

            # Update stats
            self.engine_stats[engine_name]['total_calls'] += 1

            # Run OCR
            text, confidence = engine.extract_text(image)

            processing_time = time.time() - start_time

            # Update stats
            if text:
                self.engine_stats[engine_name]['successful_calls'] += 1
            self.engine_stats[engine_name]['total_time'] += processing_time

            # Calculate average confidence
            stats = self.engine_stats[engine_name]
            if stats['successful_calls'] > 0:
                old_avg = stats['average_confidence']
                stats['average_confidence'] = (
                    old_avg * (stats['successful_calls'] - 1) + confidence) / stats['successful_calls']

            return OCRResult(engine_name, text, confidence, processing_time)

        except Exception as e:
            processing_time = time.time() - start_time
            return OCRResult(engine_name, '', 0.0, processing_time, str(e))

    def _aggregate_results(self, results: List[OCRResult]) -> Dict[str, Any]:
        """Aggregate results from multiple OCR engines."""
        successful_results = [
            r for r in results if r.success and r.text.strip()]

        if not successful_results:
            return {'text': '', 'confidence': 0.0, 'method': 'no_results'}

        method = self.config['processing']['aggregation_method']

        if method == 'highest_confidence':
            best_result = max(successful_results, key=lambda r: r.confidence)
            return {
                'text': best_result.text,
                'confidence': best_result.confidence,
                'method': 'highest_confidence',
                'source_engine': best_result.engine_name
            }

        elif method == 'weighted_vote':
            return self._weighted_vote_aggregation(successful_results)

        elif method == 'consensus':
            return self._consensus_aggregation(successful_results)

        else:
            # Default to first successful result
            first_result = successful_results[0]
            return {
                'text': first_result.text,
                'confidence': first_result.confidence,
                'method': 'first_success',
                'source_engine': first_result.engine_name
            }

    def _weighted_vote_aggregation(self, results: List[OCRResult]) -> Dict[str, Any]:
        """Aggregate results using weighted voting based on confidence."""
        if len(results) == 1:
            result = results[0]
            return {
                'text': result.text,
                'confidence': result.confidence,
                'method': 'single_result',
                'source_engine': result.engine_name
            }

        # For now, use highest confidence result
        # TODO: Implement proper weighted voting with text similarity
        best_result = max(results, key=lambda r: r.confidence)

        # Calculate weighted confidence
        total_weight = sum(r.confidence for r in results)
        weighted_confidence = best_result.confidence

        if total_weight > 0:
            weighted_confidence = sum(
                r.confidence * r.confidence for r in results) / total_weight

        return {
            'text': best_result.text,
            'confidence': weighted_confidence,
            'method': 'weighted_vote',
            'source_engine': best_result.engine_name,
            'contributing_engines': [r.engine_name for r in results]
        }

    def _consensus_aggregation(self, results: List[OCRResult]) -> Dict[str, Any]:
        """Aggregate results using consensus method."""
        # Simple consensus: if multiple engines agree, boost confidence
        text_counts = {}

        for result in results:
            text = result.text.strip()
            if text:
                if text not in text_counts:
                    text_counts[text] = []
                text_counts[text].append(result)

        if not text_counts:
            return {'text': '', 'confidence': 0.0, 'method': 'consensus_no_text'}

        # Find most common text
        most_common_text = max(
            text_counts.keys(), key=lambda t: len(text_counts[t]))
        consensus_results = text_counts[most_common_text]

        # Calculate consensus confidence
        avg_confidence = sum(
            r.confidence for r in consensus_results) / len(consensus_results)
        consensus_boost = min(len(consensus_results) *
                              0.1, 0.3)  # Max 30% boost
        final_confidence = min(avg_confidence + consensus_boost, 1.0)

        return {
            'text': most_common_text,
            'confidence': final_confidence,
            'method': 'consensus',
            'consensus_count': len(consensus_results),
            'total_engines': len(results),
            'contributing_engines': [r.engine_name for r in consensus_results]
        }

    def _apply_symbol_correction(self, aggregated: Dict[str, Any]) -> Dict[str, Any]:
        """Apply mathematical symbol correction to aggregated text."""
        try:
            from .math_corrector import MathCorrector

            corrector = MathCorrector()
            original_text = aggregated.get('text', '')

            if original_text:
                corrected_text = corrector.correct_mathematical_text(
                    original_text)

                if corrected_text != original_text:
                    logger.info(
                        f"Applied symbol corrections: {len(original_text)} -> {len(corrected_text)} chars")
                    aggregated['text'] = corrected_text
                    aggregated['symbol_corrections_applied'] = True
                else:
                    aggregated['symbol_corrections_applied'] = False

        except Exception as e:
            logger.warning(f"Symbol correction failed: {e}")
            aggregated['symbol_correction_error'] = str(e)

        return aggregated

    def _calculate_quality_metrics(self, results: List[OCRResult], aggregated: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate quality metrics for the OCR process."""
        successful_results = [r for r in results if r.success]

        if not successful_results:
            return {
                'overall_quality': 'poor',
                'engine_agreement': 0.0,
                'processing_efficiency': 0.0,
                'confidence_consistency': 0.0
            }

        # Engine agreement (simple version)
        unique_texts = len(set(r.text.strip()
                           for r in successful_results if r.text.strip()))
        agreement_score = 1.0 / unique_texts if unique_texts > 0 else 0.0

        # Processing efficiency
        avg_time = sum(
            r.processing_time for r in successful_results) / len(successful_results)
        # 60s = 0 efficiency
        efficiency_score = max(0.0, 1.0 - (avg_time / 60.0))

        # Confidence consistency
        confidences = [r.confidence for r in successful_results]
        if len(confidences) > 1:
            confidence_std = np.std(confidences)
            consistency_score = max(0.0, 1.0 - confidence_std)
        else:
            consistency_score = 1.0

        # Overall quality
        final_confidence = aggregated.get('confidence', 0.0)
        if final_confidence >= 0.8:
            overall_quality = 'excellent'
        elif final_confidence >= 0.6:
            overall_quality = 'good'
        elif final_confidence >= 0.4:
            overall_quality = 'fair'
        else:
            overall_quality = 'poor'

        return {
            'overall_quality': overall_quality,
            'engine_agreement': agreement_score,
            'processing_efficiency': efficiency_score,
            'confidence_consistency': consistency_score,
            'successful_engines': len(successful_results),
            'total_engines': len(results)
        }

    def get_engine_stats(self) -> Dict[str, Any]:
        """Get statistics for all OCR engines."""
        return {
            'engine_stats': self.engine_stats.copy(),
            'available_engines': list(self.engines.keys()),
            'total_engines': len(self.engines)
        }

    def test_all_engines(self, test_image: np.ndarray) -> Dict[str, Any]:
        """Test all available engines with a sample image."""
        results = {}

        for engine_name in self.engines:
            try:
                start_time = time.time()
                engine = self.engines[engine_name]
                text, confidence = engine.extract_text(test_image)
                processing_time = time.time() - start_time

                results[engine_name] = {
                    'success': True,
                    'text': text,
                    'confidence': confidence,
                    'processing_time': processing_time,
                    'character_count': len(text) if text else 0
                }

            except Exception as e:
                results[engine_name] = {
                    'success': False,
                    'error': str(e),
                    'text': '',
                    'confidence': 0.0,
                    'processing_time': 0.0
                }

        return results


def demo_ocr_manager():
    """Demo function to test OCR manager capabilities."""
    print("🔍 OCR Manager Demo")
    print("=" * 40)

    try:
        # Create OCR manager
        manager = OCRManager()

        print(f"✅ Initialized OCR manager with {len(manager.engines)} engines")
        print(f"   Available engines: {list(manager.engines.keys())}")

        # Test with sample image
        try:
            from ..image_processor import create_sample_test_image
            test_image = create_sample_test_image()

            print("\n🧪 Testing all engines individually...")

            # Test each engine
            engine_tests = manager.test_all_engines(test_image)

            for engine_name, result in engine_tests.items():
                if result['success']:
                    print(f"✅ {engine_name}: {result['character_count']} chars, "
                          f"{result['confidence']:.2f} confidence, "
                          f"{result['processing_time']:.2f}s")
                else:
                    print(f"❌ {engine_name}: {result['error']}")

            print("\n🔄 Testing full OCR pipeline...")

            # Test full pipeline
            ocr_result = manager.extract_text(test_image, "test_image.jpg")

            if ocr_result['success']:
                print(f"✅ Pipeline successful!")
                print(
                    f"   Final text: {len(ocr_result['final_text'])} characters")
                print(f"   Confidence: {ocr_result['confidence']:.2f}")
                print(
                    f"   Processing time: {ocr_result['processing_time']:.2f}s")
                print(
                    f"   Engines used: {', '.join(ocr_result['engines_used'])}")
                print(
                    f"   Aggregation method: {ocr_result['aggregation_method']}")
                print(
                    f"   Quality: {ocr_result['quality_metrics']['overall_quality']}")
            else:
                print(f"❌ Pipeline failed: {ocr_result.get('error_message')}")

            # Show engine stats
            stats = manager.get_engine_stats()
            print(f"\n📊 Engine Statistics:")
            for engine_name, engine_stats in stats['engine_stats'].items():
                success_rate = 0.0
                if engine_stats['total_calls'] > 0:
                    success_rate = engine_stats['successful_calls'] / \
                        engine_stats['total_calls']
                print(f"   {engine_name}: {success_rate:.1%} success rate, "
                      f"{engine_stats['average_confidence']:.2f} avg confidence")

            print("\n🎉 OCR manager demo completed successfully!")
            return True

        except ImportError:
            print("⚠️  Image processor not available, testing basic functionality only")

            # Test basic configuration
            print(f"✅ Manager configuration loaded")
            print(f"✅ Engines: {list(manager.engines.keys())}")

            return True

    except Exception as e:
        print(f"❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    demo_ocr_manager()
