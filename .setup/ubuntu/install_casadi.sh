#!/usr/bin/env bash

set -e              # Exit on error.

printf "downloading casadi\n"
cd /
git clone --branch 3.6.7 https://github.com/casadi/casadi.git --depth=1
cd casadi
printf "building casadi\n"
mkdir -p build
cd build

cmake -DWITH_IPOPT=ON -DCMAKE_BUILD_TYPE=Debug ..
# cmake .. -DACADOS_WITH_QPOASES=ON -DACADOS_SILENT=ON -DCMAKE_OSX_ARCHITECTURES=arm64 -DBLASFEO_TARGET=GENERIC -DHPIPM_TARGET=GENERIC  -DCMAKE_BUILD_TYPE=Release -DUNIT_TESTS=OFF -DSWIG_MATLAB=0 -DSWIG_PYTHON=1

# make install -j4
make -j 1 && make install
