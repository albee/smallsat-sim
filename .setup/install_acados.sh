# Clone the acados repo
git clone https://github.com/acados/acados.git
git submodule update --recursive --init

# Build acados
cd acados
sudo mkdir -p build
cd build
cmake -DACADOS_WITH_QPOASES=ON .. -DACADOS_SILENT=ON
make install -j$(nproc --all)

# Add acados_template as a module
cd /
sudo pip install -e /acados/interfaces/acados_template
