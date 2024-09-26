#!/usr/bin/env bash

set -e      # Exit on error. Important if git clone fails.

printf "downloading l4acados\n"
cd /
git clone https://github.com/IntelligentControlSystems/l4acados.git --depth=1
cd l4acados
# git submodule update --recursive --init external/acados

# printf "building l4acados\n"
# mkdir -p external/acados/build
# cd external/acados/build
# cmake -DACADOS_PYTHON=ON .. # do not forget the ".."

# make install -j4

# printf "install l4acados_template python package\n"
# cd /
# pip install -e  /l4acados/external/acados/interfaces/acados_template
# install l4acados
cd /l4acados
pip install -e .[pytorch,gpytorch,gpytorch-exo]
