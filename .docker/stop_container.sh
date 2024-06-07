#!/bin/bash

# Stop and remove all containers
docker stop $(docker ps -q --filter "name=smallsat-sim")
docker r, $(docker ps -q --filter "name=smallsa-sim")

# Stop colima (if applicable)
colima stop