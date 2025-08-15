# syntax=docker/dockerfile:experimental

# Use ubuntu 20.04 as base image for arm64
FROM ubuntu:20.04

# TODO(dschwartz): verify CUDA/jax support
# how to portably support different CUDA versions on host?
FROM nvidia/cuda:12.4.0-devel-ubuntu20.04

Set environment variables for CUDA and cuDNN
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

# ===============================
# Install Python3.10 
# ===============================
# Install python3.10. Since ubuntu 20.04 only has python3.8 by default, we need to add some addtional ppa's to install python3.10
RUN apt-get update && apt install -y software-properties-common 

# Add deadsnakes ppa which contains python3.10 for ubuntu 20.04
RUN add-apt-repository -y ppa:deadsnakes/ppa
# Install python3.10
RUN apt update && apt install -y python3.10 python3.10-distutils python3.10-dev python3.10-minimal
# Make sure python and python3 point to python3.10, and not the system default python3.8
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1

# ===============================
# Install common dependencies
# ===============================
RUN DEBIAN_FRONTEND=noninteractive apt-get install -y \
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
    libc6 \
    libgomp1 \
    libglfw3 \
    xvfb \
    x11vnc \
    ffmpeg \
    libsm6 \
    libxext6 \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# ===============================
# Install Python dependencies
# Specified in the requirements.txt file
# ===============================
# Install pip
RUN curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py
RUN python get-pip.py
RUN rm get-pip.py

# This is to make sure the newest version of cffi is installed (for some reason it will install 1.14.0, which is an old version).
RUN pip install --upgrade pip setuptools 

# Copy the current directory contents into the container at /code
COPY . .

# Install dependencies and the package in editable mode
RUN pip install --upgrade pip
RUN pip install -e .

# Install additional dependencies and PyTorch
RUN pip install torch torchvision --extra-index-url https://download.pytorch.org/whl/cpu
RUN pip install -U "jax[cuda11]"

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

# Only set for arm64
# ENV LD_PRELOAD="/usr/local/lib/python3.10/dist-packages/torch.libs/libgomp-f3febf51.so.1.0.0 /lib/aarch64-linux-gnu/libGLdispatch.so.0"
# ENV MUJOCO_GL=glfw

