# Base image with Conda
FROM continuumio/miniconda3

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    CONDA_ENV=snpioenv

# Install system-level dependencies
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

# Create the Conda environment (as root)
RUN conda create -y -n $CONDA_ENV -c conda-forge -c btmartin721 \
    python=3.12 \
    numpy=2.2.6 \
    pandas=2.2.3 \
    pip && \
    conda clean -afy

# Create a non-root user and set home directory
RUN useradd -ms /bin/bash snpiouser && \
    mkdir -p /home/snpiouser/.config/matplotlib /app/results /app/docs /app/example_data && \
    chown -R snpiouser:snpiouser /app /home/snpiouser

# Ensure HOME and MPLCONFIGDIR are set correctly
ENV HOME=/home/snpiouser
ENV MPLCONFIGDIR=$HOME/.config/matplotlib

# Set working directory
WORKDIR /app

# Copy application files with correct permissions
COPY --chown=snpiouser:snpiouser tests/ tests/
COPY --chown=snpiouser:snpiouser scripts_and_notebooks/ scripts_and_notebooks/
COPY --chown=snpiouser:snpiouser snpio/example_data/ example_data/
COPY --chown=snpiouser:snpiouser README.md docs/README.md
COPY --chown=snpiouser:snpiouser scripts_and_notebooks/.bashrc_snpio /home/snpiouser/.bashrc

# Switch to non-root user
USER snpiouser

# Install packages with pip inside the environment as snpiouser
RUN /bin/bash -c "source activate $CONDA_ENV && \
    pip install --upgrade pip && \
    pip install snpio pytest jupyterlab"

# Run tests (non-blocking; allows image to build even if tests fail)
RUN /bin/bash -c "source activate $CONDA_ENV && pytest tests/ || echo 'Tests failed during build; continuing...'" 

# Default container command
CMD ["/bin/bash"]
