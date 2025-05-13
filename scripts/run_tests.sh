#!/bin/bash
set -e

PYTHON_PATH="/opt/anaconda3/envs/snpio-dev/bin/python"

echo "Using Python at: $PYTHON_PATH"

"$PYTHON_PATH" -m pip install -e ".[dev]"
"$PYTHON_PATH" -m pytest tests/
