#!/bin/bash

# Flags:
#   --no-cache -t if you would like to start completely fresh

colima start
docker build --no-cache -t smallsat-sim .