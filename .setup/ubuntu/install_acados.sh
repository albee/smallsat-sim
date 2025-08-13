#!/usr/bin/env bash

set -e      # Exit on error. Important if git clone fails.

printf "downloading acados\n"
cd /
git clone --branch v0.4.1 https://github.com/acados/acados.git --depth=1
cd acados
git submodule update --recursive --init --depth=1

printf "building acados\n"
mkdir -p build
cd build
# cmake -DACADOS_WITH_QPOASES=ON -DACADOS_SILENT=ON  ..
# make install -j$(nproc --all)

# cmake .. -DACADOS_WITH_QPOASES=ON -DACADOS_SILENT=ON -DCMAKE_OSX_ARCHITECTURES=arm64 -DBLASFEO_TARGET=GENERIC -DHPIPM_TARGET=GENERIC  -DCMAKE_BUILD_TYPE=Release -DUNIT_TESTS=OFF -DSWIG_MATLAB=0 -DSWIG_PYTHON=1
cmake .. -DACADOS_WITH_QPOASES=ON -DACADOS_SILENT=ON  -DBLASFEO_TARGET=GENERIC -DHPIPM_TARGET=GENERIC  -DCMAKE_BUILD_TYPE=Release -DUNIT_TESTS=OFF -DSWIG_MATLAB=0 -DSWIG_PYTHON=1
make -j 1 && make install

printf "install acados_template python package\n"
cd /
pip3 install acados/interfaces/acados_template
