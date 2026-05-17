FROM continuumio/miniconda3:23.5.2-0

# System build deps (gcc for Cython algos2.c extension)
RUN apt-get update -qq && \
    apt-get install -y -qq --no-install-recommends \
      build-essential git curl && \
    rm -rf /var/lib/apt/lists/*

# Install mamba for fast dependency resolution
RUN conda install -n base -c conda-forge mamba -y -q

# Bake the full ms-gen conda environment
COPY nrp/environment.yml /tmp/environment.yml
RUN mamba env create -n ms-gen -f /tmp/environment.yml && \
    conda clean -afy && \
    find /opt/conda/pkgs -mindepth 1 -maxdepth 1 -exec rm -rf {} + 2>/dev/null || true

# Default to the ms-gen environment
ENV PATH=/opt/conda/envs/ms-gen/bin:$PATH
ENV CONDA_DEFAULT_ENV=ms-gen
ENV CONDA_PREFIX=/opt/conda/envs/ms-gen
