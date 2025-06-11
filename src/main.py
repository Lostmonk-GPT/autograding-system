#!/usr/bin/env python3
"""
AutoGrading System - Main Entry Point
"""
import click
from config.config import config

@click.group()
@click.version_option(version="1.0.0")
def cli():
    """AutoGrading System - AI-powered mathematics assignment grading."""
    pass

@cli.command()
def test():
    """Test the installation and configuration."""
    click.echo("🎯 AutoGrading System Test")
    click.echo(f"✅ Configuration loaded: {config.get('application.name')}")
    click.echo(f"✅ Debug mode: {config.debug}")
    click.echo(f"✅ Offline mode: {config.get('application.offline_mode')}")
    
    # Test OCR libraries
    try:
        import cv2
        click.echo("✅ OpenCV imported successfully")
    except ImportError as e:
        click.echo(f"❌ OpenCV import failed: {e}")
    
    try:
        import pytesseract
        click.echo("✅ Tesseract imported successfully")
    except ImportError as e:
        click.echo(f"❌ Tesseract import failed: {e}")
    
    try:
        import easyocr
        click.echo("✅ EasyOCR imported successfully")
    except ImportError as e:
        click.echo(f"❌ EasyOCR import failed: {e}")
    
    click.echo("\n🎉 Installation test complete!")

if __name__ == "__main__":
    cli()
