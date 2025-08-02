#!/usr/bin/env bash

set -e  # Exit on error. Important if git clone fails.

# Venv
poetry env use $(pyenv which python)
python --version


# Pip deps
echo "Press any key to continue..."
read -n1
poetry install

# Acados (not on pip or apt: system-wide install!)
echo "Press any key to continue..."
read -n1
printf "Downloading acados...\n"

script_dir=$(pwd)

if [ ! -d "./deps/acados" ]; then
  mkdir -p ./deps/acados
  cd ${script_dir}/deps/acados
  git clone --branch v0.3.2 https://github.com/acados/acados.git --depth=1
  cd acados
  git submodule update --recursive --init --depth=1
else
  cd ${script_dir}/deps/acados/acados
fi 

printf "Building acados...\n"
cd ${script_dir}/deps/acados/acados
if [ ! -d "build" ]; then
  sudo mkdir -p build
fi

cd build
sudo cmake -DACADOS_WITH_QPOASES=ON -DACADOS_SILENT=ON  ..
sudo make install -j$(nproc --all)

# TODO(kalbee): this might not be using the venv properly in shell for some reason
printf "Installing acados_template python package...\n"
cd ${script_dir}/deps/acados
sudo pip3 install acados/interfaces/acados_template