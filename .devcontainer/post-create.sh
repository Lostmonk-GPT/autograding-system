#!/bin/bash

echo "🚀 Setting up AutoGrading development environment..."

# Update system packages
sudo apt-get update

# Install system dependencies for OCR and image processing
sudo apt-get install -y \
    tesseract-ocr \
    tesseract-ocr-eng \
    libtesseract-dev \
    libzbar0 \
    libzbar-dev \
    poppler-utils \
    libgl1-mesa-glx

# Install Python dependencies
pip install --upgrade pip
pip install -r requirements.txt

# Create necessary directories
mkdir -p data/{input,output,templates,test_data}
mkdir -p logs

# Copy configuration templates
cp config/settings.example.yaml config/settings.yaml
cp .env.example .env

echo "✅ Environment setup complete!"
echo "📝 Next steps:"
echo "   1. Review .env file for any needed configurations"
echo "   2. Run 'python src/main.py --help' to test installation"
