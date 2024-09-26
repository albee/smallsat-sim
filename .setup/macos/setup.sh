# 926 B
#!/bin/bash
set -e
printf "[info] this installer is intended for MacOS; however it was only tested on Sierra/Monterey.\n"
printf "[info] your system runs:\n$(system_profiler SPSoftwareDataType)\n"
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
if [[ -f docker-compose-macos.yaml ]]; then
    sudo install ../smallsat /usr/local/bin
	ln -sf $PWD/docker-compose-macos.yaml ../../docker-compose.yaml
else
    printf "Cannot run the setup in this folder! Change to the correct setup sub-folder.\n"
    exit 1
fi
printf "creating log folder...\n"
mkdir -p ../../rosbags
printf "done! Close the terminal and open a new one!\n"
cd ../..
exit 0