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

ANACONDA_USER="btmartin721"

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

# 4. Locate the built package file
info "Locating built package file..."
# Search for .conda first (newer format), then .tar.bz2.
# -maxdepth 1 ensures we only look in the immediate noarch/ subdir.
PACKAGE_FILE_PATH=$(find "${PROJECT_ROOT}/${OUTPUT_DIR}/noarch/" -maxdepth 1 -name "${PACKAGE_NAME}*.conda" -print -quit 2>/dev/null)

if [ -z "${PACKAGE_FILE_PATH}" ] || [ ! -f "${PACKAGE_FILE_PATH}" ]; then
    info "No .conda package found for ${PACKAGE_NAME} in ./${OUTPUT_DIR}/noarch/, looking for .tar.bz2..."
    PACKAGE_FILE_PATH=$(find "${PROJECT_ROOT}/${OUTPUT_DIR}/noarch/" -maxdepth 1 -name "${PACKAGE_NAME}*.tar.bz2" -print -quit 2>/dev/null)
fi

if [ -z "${PACKAGE_FILE_PATH}" ] || [ ! -f "${PACKAGE_FILE_PATH}" ]; then
    error_exit "Built package file not found in ./${OUTPUT_DIR}/noarch/ matching '${PACKAGE_NAME}*.[conda|tar.bz2]'"
fi
success "Found built package: ${PACKAGE_FILE_PATH}"

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

# 6. Upload Section (Optional)
echo # Newline for better readability
read -p "Do you want to upload the package '${PACKAGE_FILE_PATH##*/}' to Anaconda.org user '${ANACONDA_USER}'? (y/N): " UPLOAD_CHOICE

if [[ "${UPLOAD_CHOICE}" == "y" || "${UPLOAD_CHOICE}" == "Y" ]]; then
    info "Preparing to upload package..."

    if ! command -v anaconda &> /dev/null; then
        error_exit "anaconda-client is not found. Please install it (e.g., 'conda install anaconda-client'). You'll also need to be logged in ('anaconda login') or have the ANACONDA_API_TOKEN environment variable set."
    fi
    info "anaconda-client found: $(command -v anaconda)"

    info "Attempting to upload ${PACKAGE_FILE_PATH##*/} to user '${ANACONDA_USER}' on Anaconda.org."
    info "This will use the ANACONDA_API_TOKEN environment variable if set, or your credentials from 'anaconda login'."
    info "The package will be uploaded to the 'main' label and will overwrite if it already exists (--force)."

    # anaconda-client automatically uses ANACONDA_API_TOKEN env var if set.
    # Otherwise, it uses tokens from `anaconda login`.
    # If neither, it may prompt (if tty) or fail.
    anaconda upload "${PACKAGE_FILE_PATH}" --user "${ANACONDA_USER}" --label main --force || error_exit "Anaconda upload failed. Ensure you are logged in ('anaconda login') or ANACONDA_API_TOKEN is correctly set."
    
    # The package name on anaconda.org will be the 'name' field from your meta.yaml (e.g., 'snpio')
    success "Package uploaded successfully to Anaconda.org! View at: https://anaconda.org/${ANACONDA_USER}/${PACKAGE_NAME}"
else
    info "Skipping upload."
fi

# 7. Cleanup
echo # Newline for better readability
read -p "Do you want to remove the test environment ('${TEST_ENV_NAME}') and the local '.condarc' file? (y/N): " REMOVE_CHOICE

if [[ "${REMOVE_CHOICE}" == "y" || "${REMOVE_CHOICE}" == "Y" ]]; then
    info "Cleaning up..."
    if conda env list | grep -qw "${TEST_ENV_NAME}"; then # -q for quiet, -w for whole word
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
