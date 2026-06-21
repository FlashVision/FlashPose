#!/bin/bash
# FlashPose — Environment Setup Script
# Creates a virtual environment and installs all dependencies.

set -e

PYTHON=${PYTHON:-python3}
VENV_DIR=${VENV_DIR:-.venv}

echo "╔════════════════════════════════════════════════╗"
echo "║         FlashPose Environment Setup            ║"
echo "╚════════════════════════════════════════════════╝"
echo ""

# Check Python version
PYVER=$($PYTHON --version 2>&1 | grep -oP '\d+\.\d+')
echo "Python: $PYVER"

if [[ $(echo "$PYVER < 3.8" | bc -l) -eq 1 ]]; then
    echo "Error: Python 3.8+ is required (found $PYVER)"
    exit 1
fi

# Create virtual environment
if [ ! -d "$VENV_DIR" ]; then
    echo "Creating virtual environment in $VENV_DIR..."
    $PYTHON -m venv "$VENV_DIR"
else
    echo "Virtual environment already exists: $VENV_DIR"
fi

# Activate
source "$VENV_DIR/bin/activate"
echo "Activated: $(which python)"

# Upgrade pip
pip install --upgrade pip setuptools wheel

# Install FlashPose
echo ""
echo "Installing FlashPose..."
pip install -e ".[all,dev]"

# Verify
echo ""
echo "Verifying installation..."
flashpose version

echo ""
echo "╔════════════════════════════════════════════════╗"
echo "║           Setup Complete!                      ║"
echo "╠════════════════════════════════════════════════╣"
echo "║  Activate: source $VENV_DIR/bin/activate       ║"
echo "║  Verify:   flashpose check                     ║"
echo "║  Train:    flashpose train --config configs/   ║"
echo "╚════════════════════════════════════════════════╝"
