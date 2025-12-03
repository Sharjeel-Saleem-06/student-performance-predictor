#!/bin/bash
# Quick activation script for ML_PROJECT environment
# Usage: source activate_env.sh

cd /Users/muhammadsharjeel/Documents/ML_PROJECT
source venv/bin/activate

echo "✅ Virtual environment activated!"
echo "📍 Project: ML_PROJECT"
echo "🐍 Python: $(python --version)"
echo "📦 Packages: $(pip list | wc -l | tr -d ' ') installed"
echo ""
echo "Quick commands:"
echo "  - Run script: python src/pipeline/train_pipeline.py"
echo "  - Install package: pip install package_name"
echo "  - List packages: pip list"
echo "  - Deactivate: deactivate"
echo ""

