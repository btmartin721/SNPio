# --- Configurable Variables ---
IMAGE_NAME = snpio
TAG = latest
CONTAINER_NAME = snpio-container
WORKDIR = /app

# Paths for local volume mounts
DATA_DIR = $(PWD)/data
RESULTS_DIR = $(PWD)/results
SCRIPTS_DIR = ${PWD}/scripts_and_notebooks

# --- Targets ---

# Build the Docker image
build:
	docker build -t $(IMAGE_NAME):$(TAG) .

# Run the container interactively
run:
	docker run --rm -it \
		-v $(DATA_DIR):/app/data \
		-v $(RESULTS_DIR):/app/results \
		-v $(SCRIPTS_DIR):/app/scripts_and_notebooks \
		--workdir $(WORKDIR) \
		--name $(CONTAINER_NAME) \
		$(IMAGE_NAME):$(TAG) bash

# Run the default SNPio CLI help command
help:
	docker run --rm \
		$(IMAGE_NAME):$(TAG) --help

# Run test suite inside the container
test:
	docker run --rm \
		--workdir /app \
		$(IMAGE_NAME):$(TAG) pytest tests/

# Clean up Docker artifacts
clean:
	docker image rm $(IMAGE_NAME):$(TAG) || true
	docker container rm $(CONTAINER_NAME) || true

# Full rebuild: clean + build
rebuild: clean build
