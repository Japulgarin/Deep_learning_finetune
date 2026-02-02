#!/bin/bash

# Quick setup script for the Airbnb fine-tuning project

echo "Setting up Airbnb Fine-Tuning Project..."

# Install dependencies
echo "Installing dependencies..."
pip install -q torch torchvision pandas numpy scikit-learn matplotlib seaborn tqdm

# Set PYTHONPATH
export PYTHONPATH="${PWD}:${PYTHONPATH}"

echo "✓ Setup complete!"
echo ""
echo "You can now run:"
echo "  python examples/simple_training.py"
echo "  python examples/finetune_example.py"
echo "  python train.py --help"
echo ""
echo "Or start Jupyter notebook:"
echo "  jupyter notebook notebooks/airbnb_finetune_example.ipynb"
