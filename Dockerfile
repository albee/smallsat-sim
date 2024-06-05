# Choose a base image
FROM ubuntu:20.04

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
RUN pip install --no-cache-dir -r requirements.txt

# # Define 
ENV PYTHONPATH="/smallsat-sim"
ENTRYPOINT ["python"]

# Use a dummy command to keep the container running indefinitely
# CMD ["tail", "-f", "/dev/null"]