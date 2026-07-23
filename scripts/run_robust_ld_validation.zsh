#!/usr/bin/env zsh

emulate -LR zsh
setopt errexit nounset pipefail

readonly SCRIPT_DIR="${0:A:h}"
readonly SCRIPT_NAME="${0:t}"
readonly REPO_ROOT="${SCRIPT_DIR:h}"
readonly VALIDATOR="${SCRIPT_DIR}/validate_ld.py"
readonly DEFAULT_VCF="${REPO_ROOT}/snpio/example_data/vcf_files/phylogen_subset14K.vcf.gz"
readonly DEFAULT_POPMAP="${REPO_ROOT}/snpio/example_data/popmaps/phylogen_nomx.popmap"
readonly RUN_STAMP="$(date -u '+%Y%m%dT%H%M%SZ')"

PYTHON_COMMAND="${PYTHON:-python3}"
OUTPUT_DIR="${REPO_ROOT}/validation_results/linkage_disequilibrium/robust_${RUN_STAMP}"
VCF_PATH="${DEFAULT_VCF}"
POPMAP_PATH="${DEFAULT_POPMAP}"
GENEPOP_PATH=""
JOBS=8
SEED=20260715
SIMULATION_REPLICATES=250
SKIP_TESTS=0
SKIP_SIMULATION=0
SKIP_PUBLISHED=0
SKIP_CONVERGENCE=0
DRY_RUN=0

usage() {
    print -r -- "Usage: ${SCRIPT_NAME} [options]"
    print -r -- ""
    print -r -- "Run the complete SNPio LD validation hierarchy and preserve a"
    print -r -- "timestamped result bundle with logs and run metadata."
    print -r -- ""
    print -r -- "Options:"
    print -r -- "  --genepop FILE          Island-fox GP_NO_grays.txt input. Required"
    print -r -- "                          unless --skip-published is supplied."
    print -r -- "  --vcf FILE              Convergence VCF (default: SNPio 14K example)."
    print -r -- "  --popmap FILE           Convergence population map (default: example)."
    print -r -- "  --output DIR            New output directory (default: timestamped)."
    print -r -- "  --python EXECUTABLE     Python interpreter (default: \$PYTHON or python3)."
    print -r -- "  --jobs N                Parallel workers (default: 8)."
    print -r -- "  --seed N                Master random seed (default: 20260715)."
    print -r -- "  --simulation-replicates N"
    print -r -- "                          Replicates per valid simulation cell (default: 250)."
    print -r -- "  --skip-tests            Skip the focused LD unit-test suite."
    print -r -- "  --skip-simulation       Skip fwdpy11/tskit calibration."
    print -r -- "  --skip-published        Skip the published island-fox benchmark."
    print -r -- "  --skip-convergence      Skip target-dataset pair convergence."
    print -r -- "  --dry-run               Validate arguments and print commands only."
    print -r -- "  -h, --help              Show this help message."
}

die() {
    print -u2 -r -- "ERROR: $*"
    exit 2
}

require_value() {
    local option="$1"
    local value="${2-}"
    [[ -n "${value}" ]] || die "${option} requires a value."
}

is_positive_integer() {
    [[ "$1" == <-> ]] && (( $1 > 0 ))
}

is_nonnegative_integer() {
    [[ "$1" == <-> ]]
}

while (( $# > 0 )); do
    case "$1" in
        --genepop)
            require_value "$1" "${2-}"
            GENEPOP_PATH="$2"
            shift 2
            ;;
        --vcf)
            require_value "$1" "${2-}"
            VCF_PATH="$2"
            shift 2
            ;;
        --popmap)
            require_value "$1" "${2-}"
            POPMAP_PATH="$2"
            shift 2
            ;;
        --output)
            require_value "$1" "${2-}"
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --python)
            require_value "$1" "${2-}"
            PYTHON_COMMAND="$2"
            shift 2
            ;;
        --jobs)
            require_value "$1" "${2-}"
            JOBS="$2"
            shift 2
            ;;
        --seed)
            require_value "$1" "${2-}"
            SEED="$2"
            shift 2
            ;;
        --simulation-replicates)
            require_value "$1" "${2-}"
            SIMULATION_REPLICATES="$2"
            shift 2
            ;;
        --skip-tests)
            SKIP_TESTS=1
            shift
            ;;
        --skip-simulation)
            SKIP_SIMULATION=1
            shift
            ;;
        --skip-published)
            SKIP_PUBLISHED=1
            shift
            ;;
        --skip-convergence)
            SKIP_CONVERGENCE=1
            shift
            ;;
        --dry-run)
            DRY_RUN=1
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            die "Unknown option: $1"
            ;;
    esac
done

is_positive_integer "${JOBS}" || die "--jobs must be a positive integer."
is_nonnegative_integer "${SEED}" || die "--seed must be a non-negative integer."
is_positive_integer "${SIMULATION_REPLICATES}" || \
    die "--simulation-replicates must be a positive integer."

OUTPUT_DIR="${OUTPUT_DIR:A}"
VCF_PATH="${VCF_PATH:A}"
POPMAP_PATH="${POPMAP_PATH:A}"
if [[ -n "${GENEPOP_PATH}" ]]; then
    GENEPOP_PATH="${GENEPOP_PATH:A}"
fi

[[ -f "${VALIDATOR}" ]] || die "Validation CLI not found: ${VALIDATOR}"
if (( ! SKIP_CONVERGENCE )); then
    [[ -f "${VCF_PATH}" ]] || die "VCF not found: ${VCF_PATH}"
    [[ -f "${POPMAP_PATH}" ]] || die "Population map not found: ${POPMAP_PATH}"
fi
if (( ! SKIP_PUBLISHED )); then
    [[ -n "${GENEPOP_PATH}" ]] || \
        die "Provide --genepop FILE or explicitly use --skip-published."
    [[ -f "${GENEPOP_PATH}" ]] || die "GenePop input not found: ${GENEPOP_PATH}"
fi

typeset -a FAILED_STEPS=()
typeset -a PASSED_STEPS=()
typeset -a SKIPPED_STEPS=()

if (( ! DRY_RUN )); then
    command -v "${PYTHON_COMMAND}" >/dev/null 2>&1 || \
        die "Python executable not found: ${PYTHON_COMMAND}"

    typeset -a existing_output=("${OUTPUT_DIR}"/*(DN))
    if (( ${#existing_output[@]} > 0 )); then
        die "Output directory is not empty: ${OUTPUT_DIR}"
    fi

    mkdir -p "${OUTPUT_DIR}/logs" "${OUTPUT_DIR}/.cache/matplotlib" \
        "${OUTPUT_DIR}/.cache/numba"
fi

readonly LOG_DIR="${OUTPUT_DIR}/logs"
readonly STEP_SUMMARY="${OUTPUT_DIR}/step_summary.tsv"
readonly RUN_METADATA="${OUTPUT_DIR}/run_metadata.tsv"

export MPLCONFIGDIR="${OUTPUT_DIR}/.cache/matplotlib"
export NUMBA_CACHE_DIR="${OUTPUT_DIR}/.cache/numba"
export PYTHONHASHSEED="${SEED}"

timestamp() {
    date -u '+%Y-%m-%dT%H:%M:%SZ'
}

record_skip() {
    local step_name="$1"
    local reason="$2"
    SKIPPED_STEPS+=("${step_name}")
    print -r -- "SKIP ${step_name}: ${reason}"
    if (( ! DRY_RUN )); then
        print -r -- "${step_name}"$'\t'"SKIP"$'\t'"0"$'\t'"0"$'\t'"${reason}" \
            >> "${STEP_SUMMARY}"
    fi
}

run_step() {
    setopt localoptions noerrexit pipefail

    local step_name="$1"
    shift
    local command_line="${(j: :)${(q)@}}"

    if (( DRY_RUN )); then
        print -r -- "DRY-RUN ${step_name}: ${command_line}"
        return 0
    fi

    local log_file="${LOG_DIR}/${step_name}.log"
    local started_epoch="$(date '+%s')"
    print -r -- "[$(timestamp)] START ${step_name}"
    print -r -- "Command: ${command_line}" | tee "${log_file}"

    "$@" 2>&1 | tee -a "${log_file}"
    local -a pipeline_status=("${pipestatus[@]}")
    local exit_code="${pipeline_status[1]:-1}"
    local tee_status="${pipeline_status[2]:-1}"
    if (( exit_code == 0 && tee_status != 0 )); then
        exit_code="${tee_status}"
    fi

    local elapsed=$(( $(date '+%s') - started_epoch ))
    if (( exit_code == 0 )); then
        PASSED_STEPS+=("${step_name}")
        print -r -- "${step_name}"$'\t'"PASS"$'\t'"0"$'\t'"${elapsed}"$'\t'"${log_file}" \
            >> "${STEP_SUMMARY}"
        print -r -- "[$(timestamp)] PASS ${step_name} (${elapsed}s)"
    else
        FAILED_STEPS+=("${step_name}")
        print -r -- \
            "${step_name}"$'\t'"FAIL"$'\t'"${exit_code}"$'\t'"${elapsed}"$'\t'"${log_file}" \
            >> "${STEP_SUMMARY}"
        print -u2 -r -- "[$(timestamp)] FAIL ${step_name} (exit ${exit_code})"
    fi
    return 0
}

write_metadata() {
    local git_commit="unknown"
    local git_branch="unknown"
    local git_dirty="unknown"
    local package_metadata_code
    package_metadata_code=$'import importlib.metadata as metadata\n'
    package_metadata_code+=$'import snpio\n'
    package_metadata_code+=$'names = ("snpio", "numpy", "pandas", "scipy", "matplotlib", '
    package_metadata_code+=$'"fwdpy11", "msprime", "tskit")\n'
    package_metadata_code+=$'installed = {distribution.metadata["Name"].lower(): '
    package_metadata_code+=$'distribution.version for distribution in metadata.distributions() '
    package_metadata_code+=$'if distribution.metadata["Name"]}\n'
    package_metadata_code+=$'versions = ";".join(f"{name}={installed[name]}" '
    package_metadata_code+=$'for name in names if name in installed)\n'
    package_metadata_code+=$'print("source_snpio_version\\t" + '
    package_metadata_code+=$'getattr(snpio, "__version__", "unknown"))\n'
    package_metadata_code+=$'print("packages\\t" + versions)\n'

    if command -v git >/dev/null 2>&1 && \
        git -C "${REPO_ROOT}" rev-parse --git-dir >/dev/null 2>&1; then
        git_commit="$(git -C "${REPO_ROOT}" rev-parse HEAD)"
        git_branch="$(git -C "${REPO_ROOT}" branch --show-current)"
        if [[ -n "$(git -C "${REPO_ROOT}" status --porcelain --untracked-files=no)" ]]; then
            git_dirty="true"
        else
            git_dirty="false"
        fi
    fi

    {
        print -r -- "key"$'\t'"value"
        print -r -- "started_at"$'\t'"$(timestamp)"
        print -r -- "repository"$'\t'"${REPO_ROOT}"
        print -r -- "git_commit"$'\t'"${git_commit}"
        print -r -- "git_branch"$'\t'"${git_branch}"
        print -r -- "git_dirty"$'\t'"${git_dirty}"
        print -r -- "python"$'\t'"${PYTHON_COMMAND}"
        print -r -- "jobs"$'\t'"${JOBS}"
        print -r -- "seed"$'\t'"${SEED}"
        print -r -- "simulation_replicates"$'\t'"${SIMULATION_REPLICATES}"
        print -r -- "vcf"$'\t'"${VCF_PATH}"
        print -r -- "popmap"$'\t'"${POPMAP_PATH}"
        print -r -- "genepop"$'\t'"${GENEPOP_PATH:-SKIPPED}"
        print -r -- "platform"$'\t'"$(uname -a)"
        "${PYTHON_COMMAND}" -c "${package_metadata_code}" 2>/dev/null
    } > "${RUN_METADATA}"
}

if (( DRY_RUN )); then
    print -r -- "Repository: ${REPO_ROOT}"
    print -r -- "Planned output: ${OUTPUT_DIR}"
else
    cd "${REPO_ROOT}"
    print -r -- $'step\tstatus\texit_code\tseconds\tdetail' > "${STEP_SUMMARY}"

    if ! "${PYTHON_COMMAND}" -c 'import snpio, numpy, pandas, scipy, matplotlib' \
        > "${LOG_DIR}/00_core_dependency_check.log" 2>&1; then
        die "Core Python dependencies are unavailable; see ${LOG_DIR}/00_core_dependency_check.log"
    fi
    if (( ! SKIP_TESTS )) && \
        ! "${PYTHON_COMMAND}" -c 'import pytest' \
        > "${LOG_DIR}/00_test_dependency_check.log" 2>&1; then
        die "pytest is unavailable; install SNPio's dev extra or use --skip-tests."
    fi
    if (( ! SKIP_SIMULATION )) && \
        ! "${PYTHON_COMMAND}" -c 'import fwdpy11, msprime, tskit' \
        > "${LOG_DIR}/00_simulation_dependency_check.log" 2>&1; then
        die "Simulation dependencies are unavailable. Install with " \
            "python -m pip install -e '.[ld-validation]'"
    fi
    write_metadata
fi

if (( SKIP_TESTS )); then
    record_skip "01_unit_tests" "requested with --skip-tests"
else
    run_step "01_unit_tests" \
        "${PYTHON_COMMAND}" -m pytest \
        tests/test_linkage_disequilibrium.py tests/test_ld_validation.py -q
fi

run_step "02_exact_expectations" \
    "${PYTHON_COMMAND}" "${VALIDATOR}" \
    --output "${OUTPUT_DIR}" --plot-formats png pdf --plot-dpi 300 \
    exact --sample-sizes 4 6 --atol 1e-12

run_step "03_golden_reference" \
    "${PYTHON_COMMAND}" "${VALIDATOR}" \
    --output "${OUTPUT_DIR}" --plot-formats png pdf --plot-dpi 300 \
    golden --rtol 1e-12 --atol 1e-14

if (( SKIP_PUBLISHED )); then
    record_skip "04_published_island_fox" "requested with --skip-published"
else
    run_step "04_published_island_fox" \
        "${PYTHON_COMMAND}" "${VALIDATOR}" \
        --output "${OUTPUT_DIR}" --plot-formats png pdf --plot-dpi 300 \
        published --genepop "${GENEPOP_PATH}" --n-jobs "${JOBS}" \
        --seed "${SEED}" --relative-tolerance 0.05
fi

if (( SKIP_SIMULATION )); then
    record_skip "05_forward_simulation" "requested with --skip-simulation"
else
    run_step "05_forward_simulation" \
        "${PYTHON_COMMAND}" "${VALIDATOR}" \
        --output "${OUTPUT_DIR}" --plot-formats png pdf --plot-dpi 300 \
        simulate --population-sizes 10 25 50 100 400 \
        --sample-sizes 4 6 8 20 50 --replicates "${SIMULATION_REPLICATES}" \
        --chromosomes 8 --loci-per-chromosome 100 --n-bootstraps 200 \
        --burnin-multiplier 10 --allow-residual-selfing \
        --minimum-model-population-size 100 \
        --minimum-coverage-sample-size 8 \
        --model-relative-bias-tolerance 0.05 \
        --n-jobs "${JOBS}" --seed "${SEED}"
fi

if (( SKIP_CONVERGENCE )); then
    record_skip "06_pair_convergence" "requested with --skip-convergence"
else
    run_step "06_pair_convergence" \
        "${PYTHON_COMMAND}" "${VALIDATOR}" \
        --output "${OUTPUT_DIR}" --plot-formats png pdf --plot-dpi 300 \
        convergence --vcf "${VCF_PATH}" --popmap "${POPMAP_PATH}" \
        --pair-budgets 250000 1000000 4000000 \
        --seeds 101 203 307 409 503 --n-jobs "${JOBS}" --n-bootstraps 0
fi

if (( DRY_RUN )); then
    print -r -- "Dry run completed; no commands were executed."
    exit 0
fi

{
    print -r -- "completed_at"$'\t'"$(timestamp)"
    print -r -- "passed_steps"$'\t'"${(j:,:)PASSED_STEPS}"
    print -r -- "failed_steps"$'\t'"${(j:,:)FAILED_STEPS}"
    print -r -- "skipped_steps"$'\t'"${(j:,:)SKIPPED_STEPS}"
} >> "${RUN_METADATA}"

print -r -- ""
print -r -- "Validation bundle: ${OUTPUT_DIR}"
print -r -- "Step summary: ${STEP_SUMMARY}"
if (( ${#FAILED_STEPS[@]} > 0 )); then
    print -u2 -r -- "Validation completed with failed steps: ${(j:, :)FAILED_STEPS}"
    exit 1
fi

print -r -- "Validation completed successfully."
