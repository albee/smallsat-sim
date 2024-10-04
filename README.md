# smallsat-sim

Simulation environment for SmallSat Steward. The simulation environment provides capabilities for:

- Satellite dynamics modeling
  - From a physics simulation engine (MuJoCo)
  - From differentiable custom set of dynamics (CasADi)
- Disturbance/degradation modeling 
- Collision modeling for realstic collision volumes
- Visualization

ROS2 integration is not currently implemented, but may be supported in future updates.


## Dependencies

Python dependencies are specified in `requirements.text` and are installed in the steps below.

*A future update to this repo will likely Docker-ize the project!*


## Installation
First, create a virtual environment with Python 3.10.11. This can be done with Visual Studio's extension *Python Environment Manager*, `pyenv`, outlined [here](https://robiokidenis.medium.com/how-to-install-multiple-python-on-your-mac-d20713740a2d).

```bash
pyenv install 3.10.11
pyenv global 3.10.11  # to globally switch for your user account
```

### Using install script

```bash
./install_deps.sh
```

### Using pyenv and pip
Next, create a virtual environment in `smallsat-sim`:

```bash
python -m venv .venv
```

Activate the virtual environment, where `smallsat-sim` is the name of your virtual environment:

```bash
source .venv/bin/activate && source ~/.bashrc
```

Afterwards install all necessary packages:

```bash
pip install -e .
```

### Using poetry

```bash
poetry env use $(pyenv which python)
poetry install
```

***Note for Ubuntu VMs on Mac M1!***

You will need to load the correct OpenGL interface libraries for visualization to work. Add the following lines to your `~.bashrc`:

```bash
  export LD_PRELOAD=/usr/lib/aarch64-linux-gnu/libGLEW.so:/usr/lib/aarch64-linux-gnu/libGLX_mesa.so.0
```


## Usage 
To check if the installation was successful, try the test script:

```bash
cd experiments
python3 test.py
```

This should show a cube with two thrusters moving periodically from the left to the right. Happy coding!

TODO: need Docker instructions!

### Running the Simulation Headless
TODO (i.e., how do we run the simulation with no visualization when we're training or doing Monte Carlo, etc.)

### Using the MuJoCo Backend
TODO

### Using the CasADi Backend
TODO


## Development Practices

We will be using a "git flow" style and branching off of main for feature adds.

Please develop significant features on separate branches named `feature/FEATURE_NAME`, branching off of `main`. For small changes, committing directly to `main` is fine; for larger changes merqe requests are encouraged.


## License
TODO
