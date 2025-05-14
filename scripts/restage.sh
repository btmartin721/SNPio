#!/bin/bash

set -euo pipefail # Exit on error, undefined variable, or pipe failure

git add pyproject.toml recipe/meta.yaml snpio/docs/source/conf.py snpio/docs/HEADER.yaml template.tex
