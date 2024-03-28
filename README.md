# smallsat-sim

Simulation environment for SmallSat Steward. The simulation envrionment provides capabilities for:

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
First, create a virtual environment with Python 3.10.11. This can be done with Visual Studio's extension *Python Environment Manager* or `pyenv`, outlined [here](https://robiokidenis.medium.com/how-to-install-multiple-python-on-your-mac-d20713740a2d).

```bash
pyenv install 3.10.11
pyenv global 3.10.11  # to globally switch for your user account
```

Next, create a virtual environment in `<smallsat-sim`:

```bash
python -m venv .venv
```

Activate the virutal environment, where `smallsat-sim` is the name of your virtual environment:

```bash
source .venv/bin/activate` && source ~/.bashrc
```

Afterwards install all necessary packages:

```bash
`pip install -r requirements.txt`.  
```

To check if the installation was successful, try the test script:

```bash
cd experiments
python3 test.py`
```

This should show R2D2 dropping from the sky onto the ground. Happy coding!


## Usage
TODO

### Running the Simulation Headless
TODO (i.e., how do we run the simulation with no visualization when we're training or doing Monte Carlo, etc.)

### Using the MuJoCo Backend
TODO

### Using the CasADi Backend
TODO


## Development Practices

Please develop significant features on separate branches named `feature/FEATURE_NAME`, branching off of `main`. For small changes, committing directly to main is fine; for larger changes merqe requests are encouraged. We will be using a "git flow" style and branching off of main for feature adds.


## License
TODO
