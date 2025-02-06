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

A Docker container containing all necessary packages is also provided.


## Usage
The main files to run simulations can be found in the `experiments` folder. For each smallsat, there is a choice of different planning and control algorithms. This folder also contains a training script for reinforcement learning (RL) agents. Control parameters are set in the respective configuration files of each environment.

All experiments can be run by mounting the `smallsat-sim` directory on the provided Docker container. Running simulations on GPU is only possible for RL controllers and only by using the Docker container. For more information, see the instructions below.

### Running the Simulation Headless
To run the simulation without the viewer, add the `--headless` flag to your command. This will result in shorter runtimes and less memory consumption.

### Logging Data from Experiments
To log various user-defined metrics, add `--log` to your command. To save videos of the simulation runs, add `--video`. The start and end times of the video are defined in the base environment configuration.

If you are using RL, you may also want to monitor the training online using Weights & Biases. To do so, add `--wandb` to your command. An account is needed to use this feature.

### Using the MuJoCo Backend
The RL controller uses MuJoCo XLA (MJX) as a physics engine. This allows for training in multiple environments in parallel. All other applications are built on MuJoCo.

***Note for MacOS users!***

The usual `python` command must be replaced by `mjpython` for the passive viewer to work. `mjpython` is installed as part of the `mujoco`package, and you can select it as the default interpreter in VS Code. More information can be found [here](https://mujoco.readthedocs.io/en/stable/python.html#passive-viewer).

### Using the CasADi Backend
TODO

### Using the Docker Container
A container runtime environment must be installed to build the Docker image and run containers. An alternative to Docker Desktop is [Colima](https://github.com/abiosoft/colima).

To build the docker image, run:
```
docker compose build x86_gpu  # Or x86_cpu depending on the platform
```

To run an experiment on the resulting Docker container, run:
```
docker compose run x86_gpu /path/to/experiment --headless
```
The Docker container is only compatible with headless mode for now.


## Development Practices

The project uses a [git flow](https://www.atlassian.com/git/tutorials/comparing-workflows/gitflow-workflow) style branching off of main for feature adds.

Please develop significant features on separate branches named `feature/FEATURE_NAME`, branching off of `main`. For small changes, committing directly to `main` is fine; for larger changes merqe requests are encouraged.


## License

Copyright (c) 2024, Jet Propulsion Laboratory, California Institute of Technology. All rights reserved. JPL NTR 53088.

This software is licensed under the Apache License, version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0.

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.
