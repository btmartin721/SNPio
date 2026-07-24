#!/usr/bin/env zsh

emulate -LR zsh
setopt errexit nounset pipefail

readonly SCRIPT_NAME="${0:t}"
readonly SCRIPT_DIR="${0:A:h}"
readonly REPO_ROOT="${SCRIPT_DIR:h}"
readonly GITHUB_REPOSITORY="btmartin721/SNPio"
readonly RELEASE_WORKFLOW="version-and-release.yml"
readonly PYPI_DOCKER_WORKFLOW="pypi-docker-publish.yml"
readonly CONDA_WORKFLOW="conda-publish.yml"

VERSION=""
ASSUME_YES=0
DRY_RUN=0
WAIT_FOR_RELEASE=1

usage() {
    print -r -- "Usage: ${SCRIPT_NAME} [options] [VERSION]"
    print -r -- ""
    print -r -- "Trigger SNPio's guarded GitHub Actions release workflow."
    print -r -- "VERSION must use MAJOR.MINOR.PATCH format; a leading v is accepted"
    print -r -- "and removed. If VERSION is omitted, the script prompts for it."
    print -r -- ""
    print -r -- "Examples:"
    print -r -- "  ./scripts/${SCRIPT_NAME} 1.7.4"
    print -r -- "  ./scripts/${SCRIPT_NAME} --dry-run v1.7.4"
    print -r -- "  ./scripts/${SCRIPT_NAME} --yes --no-wait 1.7.4"
    print -r -- ""
    print -r -- "Options:"
    print -r -- "  -y, --yes       Skip the interactive confirmation."
    print -r -- "  --dry-run       Run preflight checks without triggering a release."
    print -r -- "  --no-wait       Exit after dispatching the release workflow."
    print -r -- "  -h, --help      Show this help message."
}

die() {
    print -u2 -r -- "ERROR: $*"
    exit 2
}

warn() {
    print -u2 -r -- "WARNING: $*"
}

timestamp() {
    date -u '+%Y-%m-%dT%H:%M:%SZ'
}

require_command() {
    command -v "$1" >/dev/null 2>&1 || die "Required command not found: $1"
}

active_runs_for_workflow() {
    local workflow="$1"

    gh run list \
        --repo "${GITHUB_REPOSITORY}" \
        --workflow "${workflow}" \
        --limit 20 \
        --json databaseId,status,headBranch,url \
        --jq '
            .[]
            | select(
                .status == "queued"
                or .status == "in_progress"
                or .status == "requested"
                or .status == "waiting"
                or .status == "pending"
            )
            | [.databaseId, .status, .headBranch, .url]
            | @tsv
        '
}

assert_no_active_release_runs() {
    local workflow active_runs

    for workflow in \
        "${RELEASE_WORKFLOW}" \
        "${PYPI_DOCKER_WORKFLOW}" \
        "${CONDA_WORKFLOW}"
    do
        active_runs="$(active_runs_for_workflow "${workflow}")"
        if [[ -n "${active_runs}" ]]; then
            print -u2 -r -- "Active run(s) found for ${workflow}:"
            print -u2 -r -- "${active_runs}"
            die "Wait for active release or publisher runs before starting another release."
        fi
    done
}

find_release_run_id() {
    local created_after="$1"
    local run_id=""
    local attempt

    for (( attempt = 1; attempt <= 40; ++attempt )); do
        run_id="$(
            gh run list \
                --repo "${GITHUB_REPOSITORY}" \
                --workflow "${RELEASE_WORKFLOW}" \
                --branch master \
                --event workflow_dispatch \
                --limit 10 \
                --json databaseId,createdAt \
                --jq \
                "map(select(.createdAt >= \"${created_after}\"))[0].databaseId // empty"
        )"
        if [[ -n "${run_id}" ]]; then
            print -r -- "${run_id}"
            return 0
        fi
        sleep 3
    done

    return 1
}

find_publisher_run_id() {
    local workflow="$1"
    local tag="$2"
    local run_id=""
    local attempt

    for (( attempt = 1; attempt <= 40; ++attempt )); do
        run_id="$(
            gh run list \
                --repo "${GITHUB_REPOSITORY}" \
                --workflow "${workflow}" \
                --branch "${tag}" \
                --event workflow_dispatch \
                --limit 5 \
                --json databaseId \
                --jq '.[0].databaseId // empty'
        )"
        if [[ -n "${run_id}" ]]; then
            print -r -- "${run_id}"
            return 0
        fi
        sleep 3
    done

    return 1
}

wait_for_publishers() {
    local pypi_docker_run_id="$1"
    local conda_run_id="$2"
    local -a run_ids=("${pypi_docker_run_id}" "${conda_run_id}")
    local -A completed=()
    local failure=0
    local run_id run_info name run_status conclusion url

    while (( ${#completed[@]} < ${#run_ids[@]} )); do
        print -r -- "[$(timestamp)] Publisher status:"

        for run_id in "${run_ids[@]}"; do
            if [[ -n "${completed[${run_id}]-}" ]]; then
                continue
            fi

            run_info="$(
                gh run view "${run_id}" \
                    --repo "${GITHUB_REPOSITORY}" \
                    --json name,status,conclusion,url \
                    --jq '[.name, .status, (.conclusion // ""), .url] | join("|")'
            )"
            IFS='|' read -r name run_status conclusion url <<< "${run_info}"
            print -r -- \
                "  ${name}: ${run_status}${conclusion:+/${conclusion}}"
            print -r -- "    ${url}"

            if [[ "${run_status}" == "completed" ]]; then
                completed[${run_id}]=1
                if [[ "${conclusion}" != "success" ]]; then
                    failure=1
                fi
            fi
        done

        if (( ${#completed[@]} < ${#run_ids[@]} )); then
            sleep 30
        fi
    done

    if (( failure )); then
        print -u2 -r -- "One or more publisher workflows failed."
        print -u2 -r -- "Inspect failed logs with:"
        print -u2 -r -- \
            "  gh run view RUN_ID --repo ${GITHUB_REPOSITORY} --log-failed"
        return 1
    fi

    return 0
}

while (( $# > 0 )); do
    case "$1" in
        -y|--yes)
            ASSUME_YES=1
            shift
            ;;
        --dry-run)
            DRY_RUN=1
            shift
            ;;
        --no-wait)
            WAIT_FOR_RELEASE=0
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        --)
            shift
            if (( $# > 1 )); then
                die "Only one release version may be supplied."
            fi
            if (( $# == 1 )); then
                VERSION="$1"
                shift
            fi
            ;;
        -*)
            die "Unknown option: $1"
            ;;
        *)
            [[ -z "${VERSION}" ]] || die "Only one release version may be supplied."
            VERSION="$1"
            shift
            ;;
    esac
done

if [[ -z "${VERSION}" ]]; then
    [[ -t 0 ]] || die "Provide VERSION when running non-interactively."
    read "VERSION?Release version (for example, 1.7.4): "
fi

VERSION="${VERSION#v}"
[[ "${VERSION}" =~ ^[0-9]+\.[0-9]+\.[0-9]+$ ]] || \
    die "Version must use MAJOR.MINOR.PATCH format: ${VERSION}"

require_command git
require_command gh

cd "${REPO_ROOT}"
git rev-parse --show-toplevel >/dev/null 2>&1 || \
    die "Not inside a Git repository: ${REPO_ROOT}"
gh auth status --hostname github.com >/dev/null 2>&1 || \
    die "GitHub CLI is not authenticated. Run: gh auth login"

print -r -- "Refreshing origin/master and release tags..."
git fetch origin master --tags --quiet
git cat-file -e "origin/master:.github/workflows/${RELEASE_WORKFLOW}" 2>/dev/null || \
    die "Release workflow is missing from origin/master."

if git show-ref --verify --quiet "refs/tags/v${VERSION}"; then
    die "Local tag v${VERSION} already exists."
fi

remote_tag="$(
    git ls-remote --tags origin "refs/tags/v${VERSION}"
)"
[[ -z "${remote_tag}" ]] || die "Remote tag v${VERSION} already exists."

assert_no_active_release_runs

local_changes="$(git status --porcelain)"
if [[ -n "${local_changes}" ]]; then
    warn "Local changes are not part of this release:"
    print -u2 -r -- "${local_changes}"
fi

master_sha="$(git rev-parse --short=12 origin/master)"
print -r -- ""
print -r -- "Release:        v${VERSION}"
print -r -- "Source:         origin/master (${master_sha})"
print -r -- "Repository:     ${GITHUB_REPOSITORY}"
if (( WAIT_FOR_RELEASE )); then
    print -r -- "Wait for jobs:  yes"
else
    print -r -- "Wait for jobs:  no"
fi

if (( DRY_RUN )); then
    print -r -- ""
    print -r -- "DRY-RUN: preflight checks passed."
    print -r -- "Would run:"
    print -r -- \
        "  gh workflow run ${RELEASE_WORKFLOW} --repo ${GITHUB_REPOSITORY}" \
        "--ref master -f manual_tag=${VERSION}"
    exit 0
fi

if (( ! ASSUME_YES )); then
    [[ -t 0 ]] || die "Use --yes when running non-interactively."
    reply=""
    read "reply?Trigger release v${VERSION} from origin/master? [y/N] "
    [[ "${reply:l}" == "y" || "${reply:l}" == "yes" ]] || {
        print -r -- "Release cancelled."
        exit 0
    }
fi

triggered_at="$(date -u '+%Y-%m-%dT%H:%M:%SZ')"
print -r -- "Triggering release v${VERSION}..."
trigger_output="$(
    gh workflow run "${RELEASE_WORKFLOW}" \
        --repo "${GITHUB_REPOSITORY}" \
        --ref master \
        -f "manual_tag=${VERSION}"
)"
if [[ -n "${trigger_output}" ]]; then
    print -r -- "${trigger_output}"
fi

if (( ! WAIT_FOR_RELEASE )); then
    print -r -- "Release dispatched. Monitor it at:"
    print -r -- "https://github.com/${GITHUB_REPOSITORY}/actions"
    exit 0
fi

release_run_id=""
if [[ "${trigger_output}" == *"/actions/runs/"<-> ]]; then
    release_run_id="${trigger_output##*/}"
fi
if [[ -z "${release_run_id}" ]]; then
    if ! release_run_id="$(find_release_run_id "${triggered_at}")"; then
        die "Release was dispatched, but its workflow run could not be located."
    fi
fi

print -r -- "Watching release workflow ${release_run_id}..."
gh run watch "${release_run_id}" \
    --repo "${GITHUB_REPOSITORY}" \
    --exit-status

tag="v${VERSION}"
release_url="$(
    gh release view "${tag}" \
        --repo "${GITHUB_REPOSITORY}" \
        --json url \
        --jq '.url'
)"
print -r -- "GitHub Release: ${release_url}"

pypi_docker_run_id=""
conda_run_id=""
if ! pypi_docker_run_id="$(
    find_publisher_run_id "${PYPI_DOCKER_WORKFLOW}" "${tag}"
)"; then
    die "Could not locate the PyPI/Docker publisher for ${tag}."
fi
if ! conda_run_id="$(
    find_publisher_run_id "${CONDA_WORKFLOW}" "${tag}"
)"; then
    die "Could not locate the Conda publisher for ${tag}."
fi

print -r -- "Monitoring package publishers..."
wait_for_publishers "${pypi_docker_run_id}" "${conda_run_id}"

print -r -- ""
print -r -- "Release v${VERSION} completed successfully."
print -r -- "GitHub: ${release_url}"
print -r -- "PyPI:   https://pypi.org/project/snpio/${VERSION}/"
print -r -- "Conda:  https://anaconda.org/btmartin721/snpio/files?version=${VERSION}"
print -r -- "Docker: https://hub.docker.com/r/btmartin721/snpio/tags?name=${VERSION}"
