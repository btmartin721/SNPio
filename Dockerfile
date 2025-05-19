# Base image
FROM python:3.12-slim

# Environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    git \
    curl \
    wget \
    libbz2-dev \
    liblzma-dev \
    libz-dev \
    libcurl4-openssl-dev \
    libssl-dev \
    libncurses-dev \
    libhdf5-dev \
    ca-certificates \
    unzip \
    procps \
    && rm -rf /var/lib/apt/lists/*


# Upgrade pip and install build tools
RUN pip install --upgrade pip setuptools wheel build

# Create working directory
WORKDIR /app

# Create logical user directories
RUN mkdir -p \
    /app/results \
    /app/docs \
    /app/example_data

# Copy only necessary files early for Docker caching
COPY tests/ tests/
COPY scripts_and_notebooks/ scripts_and_notebooks/
COPY snpio/example_data/ example_data/
COPY UserManual.pdf docs/UserManual.pdf
COPY README.md docs/README.md

# Copy custom bashrc
COPY scripts_and_notebooks/.bashrc_snpio /root/.bashrc

RUN pip install --upgrade pip
RUN pip install jupyterlab

# Install SNPio package and its dependencies
RUN pip install snpio

# Optional: validate installation
RUN pytest tests/ || echo "Tests failed during build; continuing..."

# Default entrypoint
CMD ["bash"]
