#!/usr/bin/env bash

machine_arch=$(uname -m)

tera_renderer_dir="/acados/interfaces/acados_template/tera_renderer/"

# Install the tera renderer
cd $tera_renderer_dir
apt-get update 
apt-get install curl -y --no-install-recommends
curl https://sh.rustup.rs -sSf | sh -s -- -y
. "$HOME/.cargo/env"
cargo build --verbose --release

# If it it was already installed, delete and clean up
file_name="/acados/bin/t_renderer"
if [ -e "$file_name" ] || [ -L "$file_name" ]; then
    rm "$file_name"
fi
cp "${tera_renderer_dir}/target/release/t_renderer" "$file_name"
cargo clean
rustup self uninstall -y

exit 0
