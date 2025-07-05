#!/bin/bash
set -e
source /opt/miniconda3/etc/profile.d/conda.sh
conda activate /Users/btm002/miniconda3/envs/snpio-dev-2
pytest tests/
