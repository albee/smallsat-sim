#! /bin/bash

# Dependency check
check_for_package() {
    PACKAGE_NAME=$1

    # Check if the package is installed  
    if pip show "$PACKAGE_NAME" &> /dev/null; then
      echo "$PACKAGE_NAME is installed, continuing."
    else
      echo "$PACKAGE_NAME is not installed. Install with `pip install $PACKAGE_NAME`."
    fi
}
check_for_package "pylint"

# Generate useful class diagrams for a sim overview
pyreverse -o png -p SmallSatSim ../smallsat_sim/
pyreverse -o png -p Experiments ../experiments/

# Also make .dot
pyreverse -o dot -p SmallSatSim ../smallsat_sim/
pyreverse -o dot -p Experiments ../experiments/
