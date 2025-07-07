# Base image with Conda
FROM continuumio/miniconda3

# Use bash login shells so conda.sh is auto-sourced
SHELL ["/bin/bash", "-l", "-c"]

# Environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    CONDA_ENV=snpioenv \
    PATH=/opt/conda/envs/${CONDA_ENV}/bin:/opt/conda/bin:$PATH \
    HOME=/home/snpiouser \
    MPLCONFIGDIR=/home/snpiouser/.config/matplotlib

# Install system-level dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential gcc git curl wget \
        libbz2-dev liblzma-dev libz-dev \
        libcurl4-openssl-dev libssl-dev \
        libncurses-dev libhdf5-dev \
        ca-certificates unzip procps \
    && rm -rf /var/lib/apt/lists/*

# Ensure conda.sh is sourced in every bash session
RUN echo ". /opt/conda/etc/profile.d/conda.sh" >> /etc/bash.bashrc

# Update base conda, create our env with Python & core deps
RUN conda update -n base -c defaults conda \
    && conda create -y -n ${CONDA_ENV} -c conda-forge -c btmartin721 \
         python=3.12 numpy=2.2.6 pandas=2.2.3 pip \
    && conda clean -afy

# Create a non-root user and directories
RUN useradd -ms /bin/bash snpiouser \
    && mkdir -p /home/snpiouser/.config/matplotlib /app/{results,docs,example_data} \
    && chown -R snpiouser:snpiouser /home/snpiouser /app

# Switch to non-root
USER snpiouser

# Auto-activate our conda env in interactive shells
RUN echo "conda activate ${CONDA_ENV}" >> /home/snpiouser/.bashrc

# Set working directory
WORKDIR /app

# Copy application code & tests
COPY --chown=snpiouser:snpiouser tests/            tests/
COPY --chown=snpiouser:snpiouser scripts_and_notebooks/ scripts_and_notebooks/
COPY --chown=snpiouser:snpiouser snpio/example_data/    example_data/
COPY --chown=snpiouser:snpiouser README.md           docs/README.md

# Install snpio, pytest, and Jupyter into the env (PATH already set)
RUN pip install --upgrade pip snpio pytest jupyterlab

# Run your tests (failures won’t stop the build)
RUN pytest tests/ \
    || echo 'Tests failed during build; continuing…'

# Default to an interactive bash with your env on PATH
CMD ["bash"]
