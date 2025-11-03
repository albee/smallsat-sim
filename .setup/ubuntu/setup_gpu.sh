#!/usr/bin/env bash

set -e
printf "[info] this installer is intended for Ubuntu systems with NVIDIA GPUs.\n"
printf "[info] your system runs:\n$(cat /etc/*-release)\n"
read -n 1 -s -r -p "To proceed press any key"

printf "\nchecking for docker...\n"
PKG_OK=$(docker -v || if [ $? == 1 ]; then exit 0; else exit 2; fi)
if [ "$PKG_OK" == "" ]; then
    printf "docker not found! install docker!\n"
    exit 1
else
    printf "docker found, version: $PKG_OK\n"
fi

printf "installing GPU-enabled smallsat docker...\n"
if [[ -f docker-compose-ubuntu-gpu.yaml ]]; then
    sudo install ../smallsat /usr/bin/
    ln -sf $PWD/docker-compose-ubuntu-gpu.yaml ../../docker-compose.yaml
    docker pull cturra/ntp  # used for NTP sync server
else
    printf "Cannot run the setup in this folder! Change to the correct setup sub-folder.\n"
    exit 1
fi

printf "done!\n"
cd ../..
exit 0
