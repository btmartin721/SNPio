#!/bin/bash
set -euo pipefail # Exit on error, undefined variable, or pipe failure

# --- Configuration ---
# Python version to use for executing the conda-build process (meta.yaml's build/host requirements)
# For a noarch: python package, this version runs the build script.
PYTHON_FOR_BUILD="3.12"

# Python version for the separate test environment (should be compatible with your package's run requirements)
PYTHON_FOR_TEST_ENV="3.12"

# Package name (as defined in your meta.yaml)
PACKAGE_NAME="snpio"

# Project structure (assumes script is run from project root)
PROJECT_ROOT="$(pwd)"
RECIPE_DIR="recipe"
OUTPUT_DIR="conda_build_artifacts"

# Generate a unique name for the test environment to avoid conflicts
TEST_ENV_NAME="${PACKAGE_NAME}_test_$(date +%s)"

# --- Helper Functions ---
info() {
    echo "[INFO] $1"
}

success() {
    echo "[SUCCESS] $1"
}

error_exit() {
    echo "[ERROR] $1" >&2
    exit 1
}

# --- Main Script ---
info "Starting local Conda package build and test script for '${PACKAGE_NAME}'."
info "Project root: ${PROJECT_ROOT}"
info "Using Python ${PYTHON_FOR_BUILD} for the build process."
info "Using Python ${PYTHON_FOR_TEST_ENV} for the test environment."

# 1. Check for Conda
if ! command -v conda &> /dev/null; then
    error_exit "Conda command not found. Please ensure Miniconda or Anaconda is installed and in your PATH."
fi
info "Conda found: $(command -v conda)"
if ! conda info --envs | grep -q "conda-build"; then
    # This is a basic check; a more robust one would be `conda list -n base conda-build`
    # or checking if `conda build --help` works.
    info "Reminder: 'conda-build' package should be installed in your base or a dedicated build environment."
    info "You can install it with: conda install -c conda-forge conda-build"
    sleep 3 # Give user time to read
fi


# 2. Setup local .condarc file
# This ensures consistent channel configuration for this project without altering global settings.
info "Setting up local .condarc file in ${PROJECT_ROOT}..."
cat << EOF > "${PROJECT_ROOT}/.condarc"
channel_priority: flexible
auto_activate_base: false
channels:
  - conda-forge
  - bioconda
  - defaults
EOF
info ".condarc created/updated successfully."

# 3. Build the Conda package
info "Building Conda package '${PACKAGE_NAME}'..."
info "Build artifacts will be placed in ./${OUTPUT_DIR}/"
# Clean previous build artifacts from the output directory if it exists
if [ -d "${PROJECT_ROOT}/${OUTPUT_DIR}" ]; then
    info "Removing previous build artifacts from ./${OUTPUT_DIR}/"
    rm -rf "${PROJECT_ROOT}/${OUTPUT_DIR}"
fi

conda build "${RECIPE_DIR}/" \
    --python "${PYTHON_FOR_BUILD}" \
    --output-folder "${OUTPUT_DIR}" --no-test || error_exit "Conda build failed."

success "Conda build completed. Package should be in ./${OUTPUT_DIR}/noarch/"

# 4. Test the locally built package
info "--- Starting Package Installation Test ---"

# 4.1 Create a new, clean Conda test environment
info "Creating new Conda test environment: '${TEST_ENV_NAME}' with Python ${PYTHON_FOR_TEST_ENV}..."
conda create -n "${TEST_ENV_NAME}" python="${PYTHON_FOR_TEST_ENV}" -y || error_exit "Failed to create test Conda environment."
info "Test environment '${TEST_ENV_NAME}' created."

# 4.2 Install the package into the test environment
# The local build artifacts folder is used as a primary channel.
LOCAL_CHANNEL_PATH="file://${PROJECT_ROOT}/${OUTPUT_DIR}"
info "Installing '${PACKAGE_NAME}' into '${TEST_ENV_NAME}' from local channel: ${LOCAL_CHANNEL_PATH}"
conda install -n "${TEST_ENV_NAME}" \
    -c "${LOCAL_CHANNEL_PATH}" \
    -c conda-forge \
    -c bioconda \
    -c defaults \
    "${PACKAGE_NAME}" -y || error_exit "Failed to install '${PACKAGE_NAME}' into test environment '${TEST_ENV_NAME}'."
success "'${PACKAGE_NAME}' installed successfully into '${TEST_ENV_NAME}'."

# 4.3 Run test commands (e.g., import and print version)
info "Running test command in '${TEST_ENV_NAME}': python -c 'import ${PACKAGE_NAME}; print(${PACKAGE_NAME}.__version__)'"
# Use 'conda run' to execute the command within the specified environment
TEST_COMMAND="import ${PACKAGE_NAME}; print('${PACKAGE_NAME} version:', '${PACKAGE_NAME}.__version__}')"
VERSION_OUTPUT=$(conda run -n "${TEST_ENV_NAME}" python -c "${TEST_COMMAND}") || error_exit "Test command failed in '${TEST_ENV_NAME}'."
info "Test command successful. Output:"
echo "${VERSION_OUTPUT}"

success "--- Package Installation Test Successful ---"

# 5. Cleanup
# Ask the user if they want to remove the test environment and local .condarc
echo # Newline for better readability
read -p "Do you want to remove the test environment ('${TEST_ENV_NAME}') and the local '.condarc' file? (y/N): " REMOVE_CHOICE

if [[ "${REMOVE_CHOICE}" == "y" || "${REMOVE_CHOICE}" == "Y" ]]; then
    info "Cleaning up..."
    if conda env list | grep -q "${TEST_ENV_NAME}"; then
        info "Removing test Conda environment: ${TEST_ENV_NAME}"
        conda env remove -n "${TEST_ENV_NAME}" -y
    else
        info "Test environment '${TEST_ENV_NAME}' not found (already removed or failed creation)."
    fi
    if [ -f "${PROJECT_ROOT}/.condarc" ]; then
        info "Removing local .condarc file from ${PROJECT_ROOT}"
        rm -f "${PROJECT_ROOT}/.condarc"
    fi
    success "Cleanup complete. Build artifacts in ./${OUTPUT_DIR}/ are preserved."
else
    info "Skipping cleanup."
    info "To manually remove the test environment later: conda env remove -n ${TEST_ENV_NAME} -y"
    info "To manually remove the local .condarc file: rm ${PROJECT_ROOT}/.condarc"
    info "Build artifacts are in ./${OUTPUT_DIR}/"
fi

success "Script finished."
