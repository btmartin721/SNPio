#!/bin/bash
set -e

PYTHON="/opt/anaconda3/envs/snpio-dev/bin/python"

echo "Python path: $PYTHON"
"$PYTHON" --version

# Show which pip is used
"$PYTHON" -m pip --version

# Show site-packages location
"$PYTHON" -c "import site; print(site.getsitepackages())"

# Show installed packages
"$PYTHON" -m pip list | grep pytz

# Attempt a clean editable install
"$PYTHON" -m pip install -e .[dev]

# Confirm snpio is importable
"$PYTHON" -c "import snpio; print('✅ SNPio import successful')"

# Now run tests
"$PYTHON" -m pytest tests/
