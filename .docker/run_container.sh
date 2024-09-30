#!/bin/bash

# Flags:
#   -it for interactive session
#   --rm to remove container after exiting
#   -v to mount volume
#   --headless (after .py script) to run the sim in headless mode

# You may need to adapt the paths, the python script that you would like to run, etc.
docker run -it -—rm -v ~/smallsat-sim:/smallsat-sim smallsat-sim experiments/astrobee_CL.py --headless
