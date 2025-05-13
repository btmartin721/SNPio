#!/bin/bash
# Fail early on errors
set -e

# Ensure login shell to activate conda env correctly
source /opt/anaconda3/etc/profile.d/conda.sh
conda activate snpio-dev

echo "✅ Conda environment activated: $CONDA_DEFAULT_ENV"

# Optional: confirm correct python path
which python

# Install snpio in editable mode with dev dependencies
pip install -e ".[dev]"

# Now run tests
pytest tests/
