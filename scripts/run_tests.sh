#!/bin/bash
set -e

source /opt/anaconda3/etc/profile.d/conda.sh
conda activate snpio-dev

# Install SNPio and all required deps (only if not already installed)
pip install -e ".[dev]"

pytest tests/
