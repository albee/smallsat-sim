# syntax=docker/dockerfile:experimental

# Use a CUDA container
FROM nvidia/cuda:12.6.0-devel-ubuntu22.04

ARG USE_CUDA=0
ENV USE_CUDA=${USE_CUDA}

# Set environment variables for CUDA and cuDNN
ENV CUDA_HOME=/usr/local/cuda \
    CUDA_PATH=/usr/local/cuda \
    LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH \
    TZ=US \
    DEBIAN_FRONTEND=noninteractive

# NVIDIA container runtime
ENV NVIDIA_VISIBLE_DEVICES ${NVIDIA_VISIBLE_DEVICES:-all}
ENV NVIDIA_DRIVER_CAPABILITIES ${NVIDIA_DRIVER_CAPABILITIES:+$NVIDIA_DRIVER_CAPABILITIES,}graphics,video,compute,utility

# Manually set timezone  
ENV TZ=US \
    DEBIAN_FRONTEND=noninteractive

# Define the folder /code as the main working directory
WORKDIR /code

# Copy setup scripts needed during image build
COPY ./.setup ./.setup

# ===============================
# Ensure Python 3.10 is available
# ===============================

RUN apt-get update && apt-get install -y \
    python3 \
    python3-distutils \
    python3-dev \
    python3-venv \
    python3-pip \
    && ln -sf /usr/bin/python3 /usr/bin/python \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# ===============================
# Install system packages
# ===============================

# Install some basic dependencies
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y \
    curl \
    wget \
    ccache \
    assimp-utils \
    build-essential \
    cmake \
    git \
    libboost-all-dev \
    libccd-dev \
    libeigen3-dev \
    libglew-dev \
    software-properties-common \
    net-tools \
    virtualenv \
    wget \
    patchelf \
    apt-transport-https \
    liblapack-dev \
    libopenblas-dev \
    libhdf5-dev \
    libegl1 \
    libgles2 \
    libosmesa6 \
    libosmesa6-dev \
    mesa-utils \
    libc6 \
    libgomp1 \
    libglfw3 \
    xvfb \
    x11vnc \
    ffmpeg \
    libsm6 \
    libxext6 \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Default to an off-screen MuJoCo GL backend; override at runtime if needed (e.g. MUJOCO_GL=egl or osmesa)
ENV MUJOCO_GL=egl

# ===============================
# Install native dependencies
# ===============================

# Compile and install acados
RUN ./.setup/ubuntu/install_acados.sh
ENV ACADOS_SOURCE_DIR="/acados"
ENV LD_LIBRARY_PATH="/acados/lib:$LD_LIBRARY_PATH"

# Setup the Tera renderer
RUN ./.setup/ubuntu/install_tera_renderer.sh

# Compile and install l4acados
RUN ./.setup/ubuntu/install_l4acados.sh
# ENV ACADOS_SOURCE_DIR="/acados"
# ENV LD_LIBRARY_PATH="/acados/lib:$LD_LIBRARY_PATH"

# ===============================
# Install Python dependencies
# ===============================

# Install pip
RUN curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py
RUN python get-pip.py
RUN rm get-pip.py

# This is to make sure the newest version of cffi is installed (for some reason it will install 1.14.0, which is an old version).
RUN pip install --upgrade pip && pip install "setuptools<81"

# Install additional dependencies; keep PyTorch only for CPU builds
RUN pip install torch torchvision --extra-index-url https://download.pytorch.org/whl/cpu

# Ensure l4acados Python package is available
RUN pip install -e /l4acados

# Copy the remaining project files
COPY . .

# Remove the OS package that is pinning blinker (which is causing problems with pip)
RUN apt-get update && \
    apt-get remove -y python3-blinker && \
    rm -rf /var/lib/apt/lists/*

# Install dependencies and the package in editable mode
RUN pip install -e .

# Force install CUDA-enabled JAX when requested (same version as in pyproject.toml)
RUN if [ "$USE_CUDA" = "1" ]; then \
        pip install "jax[cuda12]==0.6.2" --force-reinstall --no-cache-dir; \
    fi


# Only set for arm64
# ENV LD_PRELOAD="/usr/local/lib/python3.10/dist-packages/torch.libs/libgomp-f3febf51.so.1.0.0 /lib/aarch64-linux-gnu/libGLdispatch.so.0"
# ENV MUJOCO_GL=glfw
