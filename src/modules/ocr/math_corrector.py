#!/usr/bin/env python3
"""
AutoGrading System - Mathematical Symbol Corrector
Context-aware correction of common OCR errors in mathematical expressions.
"""

import re
import logging
from typing import Dict, List, Tuple, Optional, Any
import sympy as sp
from sympy.parsing.sympy_parser import parse_expr

# Handle different SymPy versions for exception imports
try:
    from sympy.parsing.exceptions import SympifyError
except ImportError:
    # For older SymPy versions
    try:
        from sympy.core.sympify import SympifyError
    except ImportError:
        # If neither works, use the general SymPy exception
        try:
            SympifyError = sp.SympifyError
        except AttributeError:
            # Create our own exception as last resort
            class SympifyError(Exception):
                pass

# Setup logging
logger = logging.getLogger(__name__)


class MathCorrector:
    """Context-aware mathematical symbol corrector for OCR output."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize mathematical corrector.

        Args:
            config: Configuration dictionary for correction settings
        """
        self.config = config or self._default_config()
        self.correction_stats = {
            'total_corrections': 0,
            'by_type': {},
            'validation_successes': 0,
            'validation_attempts': 0
        }

    def _default_config(self) -> Dict[str, Any]:
        """Get default configuration for mathematical corrections."""
        return {
            'correction_rules': {
                'character_substitutions': True,
                'operator_spacing': True,
                'parentheses_balancing': True,
                'fraction_formatting': True,
                'exponent_notation': True
            },
            'validation': {
                'use_sympy_validation': True,
                'preserve_original_on_failure': True,
                'max_expression_length': 500,
                'timeout_seconds': 5
            },
            'confidence_boost': {
                'successful_validation': 0.1,
                'multiple_corrections': 0.05,
                'pattern_recognition': 0.02
            }
        }

    def correct_mathematical_text(self, text: str) -> str:
        """Apply comprehensive mathematical symbol corrections to text.

        Args:
            text: Raw OCR text output

        Returns:
            Corrected mathematical text
        """
        if not text or not text.strip():
            return text

        try:
            original_text = text
            corrected_text = text
            corrections_made = []

            # Apply correction rules in order
            if self.config['correction_rules']['character_substitutions']:
                corrected_text, char_corrections = self._apply_character_substitutions(
                    corrected_text)
                corrections_made.extend(char_corrections)

            if self.config['correction_rules']['operator_spacing']:
                corrected_text = self._fix_operator_spacing(corrected_text)

            if self.config['correction_rules']['parentheses_balancing']:
                corrected_text = self._balance_parentheses(corrected_text)

            if self.config['correction_rules']['fraction_formatting']:
                corrected_text = self._format_fractions(corrected_text)

            if self.config['correction_rules']['exponent_notation']:
                corrected_text = self._format_exponents(corrected_text)

            # Validate corrections using SymPy if enabled
            if self.config['validation']['use_sympy_validation']:
                corrected_text = self._validate_with_sympy(
                    original_text, corrected_text)

            # Update statistics
            self.correction_stats['total_corrections'] += len(corrections_made)
            for correction_type in corrections_made:
                self.correction_stats['by_type'][correction_type] = \
                    self.correction_stats['by_type'].get(
                        correction_type, 0) + 1

            if corrected_text != original_text:
                logger.debug(
                    f"Applied {len(corrections_made)} corrections: '{original_text}' -> '{corrected_text}'")

            return corrected_text

        except Exception as e:
            logger.warning(f"Mathematical correction failed: {e}")
            return text

    def _apply_character_substitutions(self, text: str) -> Tuple[str, List[str]]:
        """Apply context-aware character substitutions for mathematical content."""
        corrections_made = []

        # Define substitution rules with context patterns
        substitution_rules = [
            # Letter-to-digit substitutions (context-aware)
            {
                'pattern': r'(?<=[0-9+\-*/=\s\(])O(?=[0-9+\-*/=\s\)])',
                'replacement': '0',
                'description': 'Letter O to digit 0',
                'type': 'letter_to_digit'
            },
            {
                'pattern': r'(?<=[0-9+\-*/=\s\(])l(?=[0-9+\-*/=\s\)])',
                'replacement': '1',
                'description': 'Lowercase l to digit 1',
                'type': 'letter_to_digit'
            },
            {
                'pattern': r'(?<=[0-9+\-*/=\s\(])I(?=[0-9+\-*/=\s\)])',
                'replacement': '1',
                'description': 'Uppercase I to digit 1',
                'type': 'letter_to_digit'
            },
            {
                'pattern': r'(?<=[0-9+\-*/=\s\(])S(?=[0-9+\-*/=\s\)])',
                'replacement': '5',
                'description': 'Letter S to digit 5',
                'type': 'letter_to_digit'
            },
            {
                'pattern': r'(?<=[0-9+\-*/=\s\(])Z(?=[0-9+\-*/=\s\)])',
                'replacement': '2',
                'description': 'Letter Z to digit 2',
                'type': 'letter_to_digit'
            },

            # Mathematical operator corrections
            {
                'pattern': r'×',
                'replacement': '*',
                'description': 'Multiplication symbol to asterisk',
                'type': 'operator_normalization'
            },
            {
                'pattern': r'÷',
                'replacement': '/',
                'description': 'Division symbol to slash',
                'type': 'operator_normalization'
            },
            {
                'pattern': r'−',
                'replacement': '-',
                'description': 'Unicode minus to hyphen',
                'type': 'operator_normalization'
            },

            # Variable-digit separation (x2 -> x*2)
            {
                'pattern': r'([a-zA-Z])(\d)',
                'replacement': r'\1*\2',
                'description': 'Variable-digit multiplication',
                'type': 'implicit_multiplication'
            },
            {
                'pattern': r'(\d)([a-zA-Z])',
                'replacement': r'\1*\2',
                'description': 'Digit-variable multiplication',
                'type': 'implicit_multiplication'
            },

            # Common OCR confusions in mathematical context
            {
                'pattern': r'(?<=\d)\s*[oO]\s*(?=\d)',
                'replacement': '0',
                'description': 'Digit-O-digit to digit-0-digit',
                'type': 'contextual_digit'
            },
        ]

        # Apply substitutions
        for rule in substitution_rules:
            original = text
            text = re.sub(rule['pattern'], rule['replacement'], text)

            if text != original:
                corrections_made.append(rule['type'])
                logger.debug(
                    f"Applied {rule['description']}: {rule['pattern']}")

        return text, corrections_made

    def _fix_operator_spacing(self, text: str) -> str:
        """Fix spacing around mathematical operators."""
        # Standardize spacing around operators
        spacing_rules = [
            # Equals sign
            (r'\s*=\s*', ' = '),
            # Plus sign
            (r'\s*\+\s*', ' + '),
            # Minus sign (but not negative numbers)
            (r'(?<!\d)\s*-\s*(?=\d)', ' -'),  # Negative numbers
            (r'(?<=\d)\s*-\s*(?=\d)', ' - '),  # Subtraction
            (r'(?<=\))\s*-\s*(?=\d)', ' - '),  # After parentheses
            # Multiplication
            (r'\s*\*\s*', ' * '),
            # Division
            (r'\s*/\s*', ' / '),
        ]

        for pattern, replacement in spacing_rules:
            text = re.sub(pattern, replacement, text)

        return text

    def _balance_parentheses(self, text: str) -> str:
        """Balance parentheses in mathematical expressions."""
        # Count opening and closing parentheses
        open_count = text.count('(')
        close_count = text.count(')')

        # Add missing closing parentheses at the end
        if open_count > close_count:
            missing_close = open_count - close_count
            text += ')' * missing_close
            logger.debug(f"Added {missing_close} closing parentheses")

        # Remove extra closing parentheses from the end
        elif close_count > open_count:
            extra_close = close_count - open_count
            # Remove from the end
            for _ in range(extra_close):
                last_close = text.rfind(')')
                if last_close != -1:
                    text = text[:last_close] + text[last_close + 1:]
            logger.debug(f"Removed {extra_close} extra closing parentheses")

        return text

    def _format_fractions(self, text: str) -> str:
        """Format fraction notation for better readability."""
        # Fix spacing in fractions
        fraction_patterns = [
            # Standard fractions (digit/digit)
            (r'(\d+)\s*/\s*(\d+)', r'\1/\2'),
            # Fractions with parentheses
            (r'\(\s*(\d+)\s*/\s*(\d+)\s*\)', r'(\1/\2)'),
            # Variable fractions
            (r'([a-zA-Z]+)\s*/\s*([a-zA-Z0-9]+)', r'\1/\2'),
        ]

        for pattern, replacement in fraction_patterns:
            text = re.sub(pattern, replacement, text)

        return text

    def _format_exponents(self, text: str) -> str:
        """Format exponent notation."""
        # Convert common exponent patterns
        exponent_patterns = [
            # x^2 format
            (r'([a-zA-Z0-9]+)\s*\^\s*([0-9]+)', r'\1^\2'),
            # Superscript patterns (if detected)
            (r'([a-zA-Z0-9]+)\s*\*\*\s*([0-9]+)', r'\1^\2'),
        ]

        for pattern, replacement in exponent_patterns:
            text = re.sub(pattern, replacement, text)

        return text

    def _validate_with_sympy(self, original_text: str, corrected_text: str) -> str:
        """Validate mathematical expressions using SymPy."""
        self.correction_stats['validation_attempts'] += 1

        try:
            # Skip validation for very long expressions
            if len(corrected_text) > self.config['validation']['max_expression_length']:
                return corrected_text

            # Try to parse the corrected expression
            parsed_expr = self._parse_mathematical_expression(corrected_text)

            if parsed_expr is not None:
                self.correction_stats['validation_successes'] += 1
                logger.debug(f"SymPy validation successful: {corrected_text}")
                return corrected_text
            else:
                # If corrected version fails but we should preserve original
                if self.config['validation']['preserve_original_on_failure']:
                    logger.debug(
                        f"SymPy validation failed, preserving original: {original_text}")
                    return original_text
                else:
                    return corrected_text

        except Exception as e:
            logger.debug(f"SymPy validation error: {e}")

            if self.config['validation']['preserve_original_on_failure']:
                return original_text
            else:
                return corrected_text

    def _parse_mathematical_expression(self, text: str) -> Optional[sp.Basic]:
        """Try to parse a mathematical expression with SymPy."""
        try:
            # Clean the text for SymPy parsing
            cleaned_text = self._clean_for_sympy(text)

            if not cleaned_text.strip():
                return None

            # Try different parsing strategies
            parsing_strategies = [
                # Standard parsing
                lambda t: parse_expr(t),
                # With transformations
                lambda t: parse_expr(t, transformations='all'),
                # Individual equation sides
                lambda t: self._parse_equation_sides(t)
            ]

            for strategy in parsing_strategies:
                try:
                    result = strategy(cleaned_text)
                    if result is not None:
                        return result
                except:
                    continue

            return None

        except Exception as e:
            logger.debug(f"SymPy parsing failed for '{text}': {e}")
            return None

    def _clean_for_sympy(self, text: str) -> str:
        """Clean text for SymPy parsing."""
        # Remove extra whitespace
        cleaned = re.sub(r'\s+', ' ', text).strip()

        # Handle equals signs in equations
        if '=' in cleaned:
            # For equations, we'll validate each side separately
            return cleaned

        # Replace ** with ^ for exponents (SymPy prefers ^ in some contexts)
        cleaned = cleaned.replace('**', '^')

        return cleaned

    def _parse_equation_sides(self, text: str) -> Optional[sp.Basic]:
        """Parse both sides of an equation separately."""
        if '=' not in text:
            return parse_expr(text)

        try:
            left, right = text.split('=', 1)
            left_expr = parse_expr(left.strip())
            right_expr = parse_expr(right.strip())

            # If both sides parse successfully, return an equation
            return sp.Eq(left_expr, right_expr)

        except:
            return None

    def analyze_corrections(self, text: str) -> Dict[str, Any]:
        """Analyze potential corrections without applying them."""
        analysis = {
            'original_text': text,
            'potential_corrections': [],
            'confidence_factors': [],
            'sympy_parseable': False,
            'estimated_improvement': 0.0
        }

        try:
            # Test if original is already parseable
            original_parseable = self._parse_mathematical_expression(
                text) is not None
            analysis['sympy_parseable'] = original_parseable

            # Apply corrections and test
            corrected = self.correct_mathematical_text(text)
            corrected_parseable = self._parse_mathematical_expression(
                corrected) is not None

            # Calculate potential improvements
            if not original_parseable and corrected_parseable:
                # Significant improvement
                analysis['estimated_improvement'] = 0.3
            elif original_parseable and corrected_parseable:
                analysis['estimated_improvement'] = 0.1  # Minor improvement
            elif len(corrected) != len(text):
                # Some formatting improvement
                analysis['estimated_improvement'] = 0.05

            analysis['corrected_text'] = corrected
            analysis['corrected_parseable'] = corrected_parseable

        except Exception as e:
            logger.warning(f"Correction analysis failed: {e}")

        return analysis

    def get_correction_stats(self) -> Dict[str, Any]:
        """Get statistics about corrections made."""
        stats = self.correction_stats.copy()

        if stats['validation_attempts'] > 0:
            stats['validation_success_rate'] = stats['validation_successes'] / \
                stats['validation_attempts']
        else:
            stats['validation_success_rate'] = 0.0

        return stats

    def reset_stats(self):
        """Reset correction statistics."""
        self.correction_stats = {
            'total_corrections': 0,
            'by_type': {},
            'validation_successes': 0,
            'validation_attempts': 0
        }


def demo_math_corrector():
    """Demo function to test mathematical symbol corrector."""
    print("🔧 Mathematical Symbol Corrector Demo")
    print("=" * 45)

    try:
        # Create corrector
        corrector = MathCorrector()

        print("✅ Mathematical corrector initialized")

        # Test cases with common OCR errors
        test_cases = [
            "2x + 3 = 7",  # Clean case
            "2x + 3 = l",  # l instead of 1
            "O + 5 = 5",   # O instead of 0
            "x^Z = 4",     # Z instead of 2
            "3×4÷2",       # Unicode operators
            "y=mx+b",      # No spacing
            "2x+3=7",      # No spacing
            "x^2-4x+3=O",  # Multiple errors
            "(2+3×4",      # Missing parenthesis
            "a/b + c/d",   # Fractions
        ]

        print("\n🧪 Testing correction cases:")

        for i, test_case in enumerate(test_cases, 1):
            print(f"\n{i}. Testing: '{test_case}'")

            # Analyze first
            analysis = corrector.analyze_corrections(test_case)

            # Apply corrections
            corrected = corrector.correct_mathematical_text(test_case)

            print(f"   Corrected: '{corrected}'")
            print(
                f"   Original parseable: {'✅' if analysis['sympy_parseable'] else '❌'}")
            print(
                f"   Corrected parseable: {'✅' if analysis.get('corrected_parseable', False) else '❌'}")
            print(f"   Improvement: {analysis['estimated_improvement']:.1%}")

        # Test SymPy validation
        print("\n🔬 Testing SymPy validation:")

        sympy_tests = [
            "x + 2 = 5",
            "x^2 + 3*x - 4",
            "sin(x) + cos(x)",
            "invalid expression $$"
        ]

        for test_expr in sympy_tests:
            parsed = corrector._parse_mathematical_expression(test_expr)
            status = "✅ Valid" if parsed is not None else "❌ Invalid"
            print(f"   '{test_expr}': {status}")

        # Show statistics
        stats = corrector.get_correction_stats()
        print(f"\n📊 Correction Statistics:")
        print(f"   Total corrections: {stats['total_corrections']}")
        print(
            f"   Validation success rate: {stats['validation_success_rate']:.1%}")
        print(f"   Correction types: {dict(stats['by_type'])}")

        print("\n🎉 Mathematical corrector demo completed!")
        return True

    except Exception as e:
        print(f"❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    demo_math_corrector()
