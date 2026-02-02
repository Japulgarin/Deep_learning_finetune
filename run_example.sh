#!/bin/bash

# Helper script to run examples with proper PYTHONPATH

# Set the project root directory
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Set PYTHONPATH to include project root
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH}"

# Run the provided script
if [ $# -eq 0 ]; then
    echo "Usage: ./run_example.sh <script_name>"
    echo "Examples:"
    echo "  ./run_example.sh examples/simple_training.py"
    echo "  ./run_example.sh examples/finetune_example.py"
    echo "  ./run_example.sh train.py --epochs 20"
    exit 1
fi

python "$@"
