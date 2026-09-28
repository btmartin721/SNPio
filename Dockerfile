# Base image with Conda
FROM continuumio/miniconda3

ARG SNPIO_VERSION

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    CONDA_ENV=snpioenv \
    CONDA_NO_PLUGINS=true \
    PIP_ROOT_USER_ACTION=ignore

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
    xz-utils \
    ca-certificates \
    unzip \
    procps \
    zsh \
    && rm -rf /var/lib/apt/lists/*

# Create a new Conda environment and install dependencies
RUN conda create --yes --solver classic --override-channels \
    --name "$CONDA_ENV" -c conda-forge -c btmartin721 \
    python=3.12 \
    numpy=2.2.6 \
    pandas=2.2.3 \
    pip && \
    conda clean -afy && \
    echo "conda activate $CONDA_ENV" > ~/.bashrc

ENV PATH=/opt/conda/envs/$CONDA_ENV/bin:$PATH

COPY dist/snpio-${SNPIO_VERSION}-py3-none-any.whl /tmp/

RUN conda run -n "$CONDA_ENV" python -m pip install --no-cache-dir \
    /tmp/snpio-${SNPIO_VERSION}-py3-none-any.whl \
    pytest \
    jupyterlab && \
    rm /tmp/snpio-${SNPIO_VERSION}-py3-none-any.whl && \
    conda clean -afy

# Create a non-root user and set home directory. useradd -m makes the home
# private (0700) on current Debian; keep it readable so the image also works
# when run as another user (e.g. docker -u, as Nextflow does).
RUN useradd -ms /bin/bash snpiouser && \
    mkdir -p /home/snpiouser/.cache/numba \
    /home/snpiouser/.config/matplotlib \
    /app/results /app/docs /app/example_data && \
    chown -R snpiouser:snpiouser /app /home/snpiouser && \
    chmod 755 /home/snpiouser

# Set working directory
WORKDIR /app

# Copy application files with correct permissions
COPY --chown=snpiouser:snpiouser tests/ tests/
COPY --chown=snpiouser:snpiouser scripts/ scripts/
COPY --chown=snpiouser:snpiouser scripts_and_notebooks/ scripts_and_notebooks/
COPY --chown=snpiouser:snpiouser snpio/example_data/ snpio/example_data/
COPY --chown=snpiouser:snpiouser multiqc_config.yml pyproject.toml ./
COPY --chown=snpiouser:snpiouser README.md docs/README.md
COPY --chown=snpiouser:snpiouser scripts_and_notebooks/.bashrc_snpio /home/snpiouser/.bashrc

# Switch to non-root user
USER snpiouser
ENV HOME=/home/snpiouser
# numba and matplotlib default to $HOME/.cache/numba and
# $HOME/.config/matplotlib. They are not pinned here, so they follow HOME
# when a runtime sets its own (e.g. Singularity/Apptainer).

# Validate the installed release before publishing the image.
RUN conda run -n "$CONDA_ENV" python -m pytest -q

# Let any user reuse and update the caches written above.
RUN chmod -R a+rwX "$HOME/.cache" "$HOME/.config"

# Default container command
CMD ["bash"]
