#!/usr/bin/bash

# Create a separate virtual environment for deepeval since there's a conflicting package version
# R2R expects a lower version of openai
python3 -m venv deepeval_venv

source deepeval_venv/bin/activate

# Upgrade tools
python3 -m pip install --upgrade pip setuptools wheel

pip3 install deepeval==2.6.5 ipykernel==6.29.5

echo "DeepEval installation complete"