#!/bin/bash

echo "=================================="
echo "Project Structure Verification"
echo "=================================="
echo ""

echo "✓ Core modules:"
ls -1 src/*/*.py 2>/dev/null | grep -v __pycache__ | head -10

echo ""
echo "✓ Example scripts:"
ls -1 examples/*.py 2>/dev/null

echo ""
echo "✓ Notebooks:"
ls -1 notebooks/*.ipynb 2>/dev/null

echo ""
echo "✓ Configuration files:"
ls -1 *.txt *.py setup.sh run_example.sh 2>/dev/null | grep -v test

echo ""
echo "✓ Documentation:"
ls -1 *.md 2>/dev/null

echo ""
echo "=================================="
echo "Quick Feature Test"
echo "=================================="
python << 'PYEOF'
import sys
sys.path.insert(0, '.')

# Test imports
print("\n✓ Testing imports...")
from src.data import AirbnbDataLoader
from src.models import create_model
from src.utils import Trainer
print("  All imports successful!")

# Test data generation
print("\n✓ Testing data generation...")
loader = AirbnbDataLoader()
df = loader.create_sample_data(50)
print(f"  Generated {len(df)} samples with {len(df.columns)} features")

# Test model creation
print("\n✓ Testing model creation...")
model = create_model(input_dim=12, model_type='simple')
params = sum(p.numel() for p in model.parameters())
print(f"  Model created with {params:,} parameters")

print("\n✓ All verification tests passed!")
print("  Project is ready to use!")
PYEOF

echo ""
echo "=================================="
echo "Ready to run examples!"
echo "=================================="
echo "Try: ./run_example.sh examples/simple_training.py"
