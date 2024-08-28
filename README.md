# smallsat-sim

The future home of SmallSatSim, a MuJoCo-based simulation environment for microgravity robotics research. SmallSatSim is easily repurposable to other robotics research scenarios including, notably, maritime robotics. SmallSatSim provides:

- easy parallelization for reinforcement learning
- system dynamics modification for model perturbations and disturbances
- built-in environments for some common microgravity research scenarios
- interfacing with MuJoCo's high-fidelity dynamics and visualization utilities
- differentiable custom dynamics for some scenarios, such as the free-flyer dynamics using CasADi

SmallSatSim is a part of the SmallSat Steward project, a collaboration between researchers at Caltech's Jet Propulsion Laboratory and the University of Michigan's Space Systems Laboratory. This project is open-sourced under an Apache 2.0 license.

## Dependencies

Python dependencies are specified in `requirements.txt` and are installed in the steps below.

## Installation
First, create a virtual environment with Python 3.10.11. This can be done with Visual Studio's extension *Python Environment Manager* or `pyenv`, outlined [here](https://robiokidenis.medium.com/how-to-install-multiple-python-on-your-mac-d20713740a2d).

```bash
pyenv install 3.10.11
pyenv global 3.10.11  # to globally switch for your user account
```

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

To check if the installation was successful, try the test script:

```bash
cd experiments
python3 test.py
```

This should show a cube with two thrusters moving periodically from the left to the right. Happy coding!

***Note for Ubuntu VMs on Mac M1!***

You will need to load the correct OpenGL interface libraries for visualization to work. Add the following lines to your `~.bashrc`:

```bash
  export LD_PRELOAD=/usr/lib/aarch64-linux-gnu/libGLEW.so:/usr/lib/aarch64-linux-gnu/libGLX_mesa.so.0
```


## Usage
TODO

### Running the Simulation Headless
TODO (i.e., how do we run the simulation with no visualization when we're training or doing Monte Carlo, etc.)

### Using the MuJoCo Backend
TODO

### Using the CasADi Backend
TODO


## Development Practices

The project uses a [git flow](https://www.atlassian.com/git/tutorials/comparing-workflows/gitflow-workflow) style branching off of main for feature adds.

Please develop significant features on separate branches named `feature/FEATURE_NAME`, branching off of `main`. For small changes, committing directly to `main` is fine; for larger changes merqe requests are encouraged.


## License

Copyright (c) 2024, Jet Propulsion Laboratory, California Institute of Technology. All rights reserved. JPL NTR 53088.

This software is licensed under the Apache License, version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0.

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.
