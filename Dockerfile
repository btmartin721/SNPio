# Base image with Conda
FROM continuumio/miniconda3

# Use bash login shells so conda.sh is auto-sourced
SHELL ["/bin/bash", "-l", "-c"]

# Environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HOME=/home/snpiouser \
    MPLCONFIGDIR=/home/snpiouser/.config/matplotlib

# Expose the new env's bin directory
ENV PATH=/opt/conda/envs/snpioenv/bin:/opt/conda/bin:$PATH

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

# Update base conda and create our env
RUN conda update -n base -c defaults conda \
    && conda create -y -n snpioenv -c conda-forge -c btmartin721 \
         python=3.12 numpy=2.2.6 pandas=2.2.3 pip \
    && conda clean -afy

# Create a non-root user and app directories
RUN useradd -ms /bin/bash snpiouser \
    && mkdir -p /home/snpiouser/.config/matplotlib /app/{results,docs,example_data} \
    && chown -R snpiouser:snpiouser /home/snpiouser /app

# Switch to non-root
USER snpiouser

# Auto-activate snpioenv in interactive shells
RUN echo "conda activate snpioenv" >> /home/snpiouser/.bashrc

# Set working directory
WORKDIR /app

# Copy in your code & tests
COPY --chown=snpiouser:snpiouser tests/            tests/
COPY --chown=snpiouser:snpiouser scripts_and_notebooks/ scripts_and_notebooks/
COPY --chown=snpiouser:snpiouser snpio/example_data/    example_data/
COPY --chown=snpiouser:snpiouser README.md           docs/README.md

# Install snpio, pytest, and Jupyter into the env (now on PATH)
RUN pip install --upgrade pip snpio pytest jupyterlab

# Run tests (failures won't stop the build)
RUN pytest tests/ \
    || echo 'Tests failed during build; continuing…'

# Default to an interactive bash with your env on PATH
CMD ["bash"]
