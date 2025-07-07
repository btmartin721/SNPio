# Base image with Conda
FROM continuumio/miniconda3

# Use bash login shells (so /etc/profile.d/*.sh get sourced)
SHELL ["/bin/bash", "-l", "-c"]

# Env vars
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    CONDA_ENV=snpioenv \
    HOME=/home/snpiouser \
    MPLCONFIGDIR=/home/snpiouser/.config/matplotlib \
    PATH=/opt/conda/bin:/opt/conda/envs/${CONDA_ENV}/bin:$PATH

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential gcc git curl wget \
        libbz2-dev liblzma-dev libz-dev \
        libcurl4-openssl-dev libssl-dev \
        libncurses-dev libhdf5-dev \
        ca-certificates unzip procps \
    && rm -rf /var/lib/apt/lists/*

# Ensure Conda’s activation logic is always available
RUN echo ". /opt/conda/etc/profile.d/conda.sh" > /etc/profile.d/conda.sh

# Update base conda, create our env (with pip)
RUN conda update -n base -c defaults conda \
 && conda create -y -n ${CONDA_ENV} -c conda-forge -c btmartin721 \
      python=3.12 numpy=2.2.6 pandas=2.2.3 pip \
 && conda clean -afy

# Create non-root user, app dirs
RUN useradd -ms /bin/bash snpiouser \
 && mkdir -p /home/snpiouser/.config/matplotlib /app/{results,docs,example_data} \
 && chown -R snpiouser:snpiouser /home/snpiouser /app

USER snpiouser
WORKDIR /app

# (Optional) auto-activate in interactive shells
RUN echo "conda activate ${CONDA_ENV}" >> /home/snpiouser/.bashrc

# Copy your code & tests
COPY --chown=snpiouser:snpiouser tests/               tests/
COPY --chown=snpiouser:snpiouser scripts_and_notebooks/ scripts_and_notebooks/
COPY --chown=snpiouser:snpiouser snpio/example_data/     example_data/
COPY --chown=snpiouser:snpiouser README.md            docs/README.md

# Install Python packages INTO snpioenv
RUN conda run -n ${CONDA_ENV} pip install --upgrade pip snpio pytest jupyterlab

# Run tests with that same env (failures won’t break the build)
RUN conda run -n ${CONDA_ENV} pytest tests/ || echo 'Tests failed; continuing…'

# Launch into bash with snpioenv on PATH
CMD ["bash"]
