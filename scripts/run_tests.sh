#!/bin/bash
set -e
source /opt/anaconda3/etc/profile.d/conda.sh
conda activate snpio-dev-gha
pytest tests/
