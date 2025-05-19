#!/bin/zsh

# -----------------------------
# Configuration
# -----------------------------
DOCKERHUB_USERNAME="btmartin721"
IMAGE_NAME="snpio"
DOCKER_COMPOSE_SERVICE="snpio"
TAG_LATEST="latest"

# -----------------------------
# Utility functions
# -----------------------------
function print_success() {
    print -P "%F{green}[✓]%f $1"
}

function print_error() {
    print -P "%F{red}[✗]%f $1"
    exit 1
}

function print_info() {
    print -P "%F{blue}[i]%f $1"
}

# -----------------------------
# Prompt for version tag
# -----------------------------
VERSION_TAG=$(grep '^version =' pyproject.toml | cut -d '"' -f2)
if [[ -z "$VERSION_TAG" ]]; then
    print_info "No version tag found in pyproject.toml. Please provide a version tag:"
    read -r VERSION_TAG
fi
if [[ -z "$VERSION_TAG" ]]; then
    print_error "No version tag provided."
fi

if [[ "$VERSION_TAG" =~ ^[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
    print_success "Version tag: $VERSION_TAG"
else
    print_error "Invalid version tag format. Expected format: x.y.z"
fi
# -----------------------------
# Check DockerHub login
# -----------------------------
print_info "Checking DockerHub login..."
if ! docker info | grep -q "Username: $DOCKERHUB_USERNAME"; then
    print_error "You are not logged into DockerHub as $DOCKERHUB_USERNAME. Run: docker login"
fi

# -----------------------------
# Build Docker image
# -----------------------------
print_info "Building Docker image using docker-compose..."
docker-compose build $DOCKER_COMPOSE_SERVICE || print_error "Docker build failed."
print_success "Docker image built."

# -----------------------------
# Tag Docker image
# -----------------------------
print_info "Tagging image as $TAG_LATEST and $VERSION_TAG..."
docker tag "$IMAGE_NAME:$TAG_LATEST" "$DOCKERHUB_USERNAME/$IMAGE_NAME:$TAG_LATEST" || print_error "Failed to tag :latest"
docker tag "$IMAGE_NAME:$TAG_LATEST" "$DOCKERHUB_USERNAME/$IMAGE_NAME:$VERSION_TAG" || print_error "Failed to tag :$VERSION_TAG"
print_success "Image tagged successfully."

# -----------------------------
# Push Docker images
# -----------------------------
print_info "Pushing tags to DockerHub..."
docker push "$DOCKERHUB_USERNAME/$IMAGE_NAME:$TAG_LATEST" || print_error "Push failed for :latest"
docker push "$DOCKERHUB_USERNAME/$IMAGE_NAME:$VERSION_TAG" || print_error "Push failed for :$VERSION_TAG"
print_success "Images pushed to DockerHub successfully."

# -----------------------------
# Done
# -----------------------------
print_success "All done! 🐳"
