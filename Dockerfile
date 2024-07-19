# Choose a base image
# FROM nvidia/cuda:11.6.2-devel-ubuntu20.04
# FROM nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu20.04
# FROM nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04
FROM nvidia/cuda:12.1.0-devel-ubuntu20.04

# Set environment variables for CUDA and cuDNN
ENV CUDA_HOME=/usr/local/cuda \
    CUDA_PATH=/usr/local/cuda \
    LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

# Choose a Python image
FROM python:3.10-bookworm

# Install the necessary Linux dependencies
RUN apt-get update && \
    apt-get install -y \
    git \ 
    cmake \
    build-essential

# Set the working directory
WORKDIR /acados

# Clone the acados repo
RUN git clone https://github.com/acados/acados.git .
RUN git submodule update --recursive --init

# Build acados
RUN mkdir -p build && \
    cd build && \
    cmake -DACADOS_WITH_QPOASES=ON .. -DACADOS_SILENT=ON && \
    make install -j$(nproc --all)

# Add acados_template as a module
RUN cd /
RUN pip install -e /acados/interfaces/acados_template

# Set the relevant acados variables
ENV ACADOS_SOURCE_DIR="/acados"
ENV LD_LIBRARY_PATH="/acados/lib:$LD_LIBRARY_PATH"

# Install the tera renderer (used by acados)
COPY /.setup/install_tera_renderer.sh /acados/install_tera_renderer.sh
RUN . /acados/install_tera_renderer.sh

# Set the working directory
WORKDIR /smallsat-sim

# Copy the requirements into the working directory
COPY requirements.txt .

# Install the specified dependencies
RUN pip install -U "jax[cuda12]"
RUN pip install --no-cache-dir -r requirements.txt

# Define use case (on which hardware the container should run)
ENV USE_CASE=default

# NVIDIA container runtime
ENV NVIDIA_VISIBLE_DEVICES ${NVIDIA_VISIBLE_DEVICES:-all}
ENV NVIDIA_DRIVER_CAPABILITIES ${NVIDIA_DRIVER_CAPABILITIES:+$NVIDIA_DRIVER_CAPABILITIES,}graphics,video,compute,utility

# Define Python path
ENV PYTHONPATH="/smallsat-sim"

# Copy the entrypoint-docker script for lambda-quad and make it executable
COPY .docker/entrypoint-docker.sh /entrypoint-docker.sh
RUN chmod +x /entrypoint-docker.sh

# Define the command/script to be executed when the container starts
# ENTRYPOINT [ "/entrypoint-docker.sh"]
ENTRYPOINT [ "python"]

# Use a dummy command to keep the container running indefinitely
# CMD ["tail", "-f", "/dev/null"]