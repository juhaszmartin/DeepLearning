#!/bin/bash

# Script to run the transformer training pipeline
# Usage: ./run.sh or bash run.sh

set -e  # Exit on error

echo "========================================="
echo "Bull Flag Detector - Training Pipeline"
echo "========================================="
echo ""

# Check if Python is available
if ! command -v python &> /dev/null && ! command -v python3 &> /dev/null; then
    echo "Error: Python is not installed or not in PATH"
    exit 1
fi

# Use python3 if available, otherwise python
if command -v python3 &> /dev/null; then
    PYTHON_CMD=python3
else
    PYTHON_CMD=python
fi

# Check if main.py exists
if [ ! -f "main.py" ]; then
    echo "Error: main.py not found in current directory"
    exit 1
fi

# Run the main script
echo "Starting training pipeline..."
echo ""
$PYTHON_CMD main.py

echo ""
echo "========================================="
echo "Training pipeline completed!"
echo "========================================="

