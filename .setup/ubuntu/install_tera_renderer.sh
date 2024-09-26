#!/usr/bin/env bash

machine_arch=$(uname -m)

printf "Found architecture %s\n" $machine_arch

echo "Building tera_renderer from source"

tera_renderer_dir="/acados/interfaces/acados_template/tera_renderer/"
# tera_renderer_dir="/l4acados/external/acados/interfaces/acados_template/tera_renderer/"
echo $tera_renderer_dir

cd $tera_renderer_dir
apt-get update && apt-get install curl -y --no-install-recommends
# sudo apt-get update && sudo apt-get install curl -y --no-install-recommends
curl https://sh.rustup.rs -sSf | sh -s -- -y
source "$HOME/.cargo/env"
cargo build --verbose --release

echo "finished building tera_renderer"

# Delete other tera_renderer if already installed
# file_name="/l4acados/external/acados/bin/t_renderer"
file_name="/acados/bin/t_renderer"
echo $file_name
if [ -e "$file_name" ] || [ -L "$file_name" ]; then
    rm "$file_name"
fi

echo "copying executable to correct place"
cp "${tera_renderer_dir}/target/release/t_renderer" "$file_name"

echo "cleaning build"
cargo clean

echo "uninstalling rust compiler"
rustup self uninstall -y

exit 0
