#!/usr/bin/env bash

set -e
printf "[info] this installer is intended for Ubuntu; it might also run on other linux distributions but it was only tested on Ubuntu 18.04/20.04/22.04 LTS\n"
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

printf "installing smallsat docker...\n"
if [[ -f docker-compose-ubuntu.yaml ]]; then
    sudo install ../smallsat /usr/bin/
	ln -sf $PWD/docker-compose-ubuntu.yaml ../../docker-compose.yaml
    docker pull cturra/ntp  # used for NTP sync server
else
    printf "Cannot run the setup in this folder! Change to the correct setup sub-folder.\n"
    exit 1
fi

# printf "creating log folder...\n"
# mkdir -p ../../rosbags
# source ~/.profile

printf "done!\n"
cd ../..
exit 0
