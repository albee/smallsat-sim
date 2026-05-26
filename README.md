# SmallSatSim

SmallSatSim is a MuJoCo-based simulation environment for microgravity robotics research. SmallSatSim is easily repurposable to other robotics research scenarios including, notably, maritime robotics. SmallSatSim provides:

- easy parallelization for reinforcement learning
- system dynamics modification for model perturbations and disturbances
- built-in environments for some common microgravity research scenarios
- interfacing with MuJoCo's high-fidelity dynamics and visualization utilities
- differentiable custom dynamics for some scenarios, such as the free-flyer dynamics using CasADi.

SmallSatSim emerged from the SmallSat Steward project, a collaboration between researchers at Caltech's Jet Propulsion Laboratory (now University of Southern California) and the University of Michigan's Space Systems Laboratory. This project is open-sourced under an Apache 2.0 license.


## Docker

### macOS

Run the setup script
```bash
cd .setup/macos
bash setup.sh
```
### Ubuntu (CPU)

Run the setup script
```bash
cd .setup/ubuntu
bash setup.sh
```

### Ubuntu (GPU)

If you have an NVIDIA GPU with the Container Toolkit installed, run the GPU-enabled setup instead:
```bash
cd .setup/ubuntu
bash setup_gpu.sh
```
This variant passes `USE_CUDA=1` to the image build so that the CUDA-enabled JAX wheels are installed.

*Note: you must have a CUDA version of 12.1 or above installed.*

### Running Docker

Build the Docker:
```bash
smallsat up
```

Run the Docker:
```bash
smallsat run
```

Additional terminals can be attached to the Docker using:
```bash
smallsat attach
```

Make sure to shut down the Docker when you are done developing:
```bash
smallsat down
```

### Viewing MuJoCo with NoVNC

The compose stack starts a NoVNC service that exposes the MuJoCo viewer through your browser.

**Local macOS workflow**

Launch your script (keep the default `MUJOCO_GL=glfw`), e.g.:

```bash
python experiments/astrobee_CL.py
```

In your browser visit `http://localhost:8080` to see the simulation window.

**Remote GPU host**

Forward the NoVNC port to your local machine,

```bash
ssh -L 8080:localhost:8080 <remote_user>@<remote_host>
```

then follow the same steps as above. You must also set `MUJOCO_GL=egl`.

**Headless execution**

If you prefer headless execution, simply use the `--headless` flag. Your experiment will run completely without rendering or GUI context. This is the fastest option (and recommended for RL training).

Alternatively, you could also set `MUJOCO_GL=osmesa` (CPU) or `MUJOCO_GL=egl` (GPU) before running your experiment; no NoVNC session is required in that case.

### Using an existing image

#### Using a base image
Download base image (.tar file) and add it to `code/` directory. \
Load prebuilt image `docker load -i base_image.tar`. \
Allow patching over local Docker image by running the following in your terminal:
```bash
docker context use default
docker buildx use default
```

Patch `Dockerfile` on top of the `base_image` base image by running:
```bash
docker buildx build --platform="linux/arm64" -t smallsat:latest --build-arg BASE_IMAGE=base_image .  --progress=plain
```

If you are not using an `arm64` base, make sure to change the `platform`.

Save built image as .tar: 
```bash
docker save -o smallsat.tar smallsat:latest
```
Run the Docker container: 
```bash
docker run -it --rm -v .:/code smallsat:latest
```

#### Loading a prebuilt image
The `smallsat.tar` file contains the pre-built image. You can load this image by running:

```bash
docker load -i smallsat.tar
``` 

Then run the Docker as usual:
```bash
smallsat run
```

## How to run a standard experiment
In a terminal that is attached to the Docker:

```bash
python experiments/[my_experiment.py]
```

You can try a test experiment with:

```bash
python experiments/test.py
```

You should see a small box appear with a sinusoidal thrust applied. Most experiments also support the `--headless` argument to ignore visualization. A more complex experiment includes inspection of a space station:

```bash
python experiments/astrobee_CL_2d.py
```

### Example experiments

Other example experiments include NASA's Astrobee free-flyer and JPL's air bearing platform. The `experiments` folder contains various examples: 

`astrobee_CL.py` : Uses the 6DoF Astrobee model, MPCC controller and mission planner. \
`astrobee_CL_2d.py` : Uses the 3DoF Astrobee model, MPCC controller and mission planner. \
`udp_controller.py` : 6DoF MPCC controller that is run in a separate thread from the simulation. It listens for messages on a given port (in this case the messages are the 6DoF observations from the simulation environment) and sends the computed 6DoF control inputs on a separate port. \
`udp_controller_2d.py` : 3DoF MPCC controller that is run in a separate thread from the simulation. It listens for messages on a given port (in this case the messages are the 6DoF observations from the simulation environment) and sends the computed 3DoF control inputs on a separate port. \
`astrobee_CL_udp_external.py` : Runs the 6DoF simulation loop. Observations are sent to a given port, and 6DoF control inputs are received from the controller. \
`astrobee_CL_2d_udp_external.py` : Runs the 3DoF simulation loop. Observations are sent to a given port, and 3DoF control inputs are received from the controller. 

## Structure

### Models
Configuration information like geometry and physical layouts of satellite objects are specified for some default objects. Meshes are given in `meshes/`, and physical layouts are provided in `cfg/config.py`.

### Environments
The environments are used to set up the MuJoCo simulations. The environments set up the symbolic model as well as the observations. The main functionalities the environment provides are:

`step()` : Simulates the environment for one timestep using the control inputs \
`set_obs()` : Sets all observations (position, orientation, velocity, angular velocity) \
`get_obs()` : Returns the current (noisy) observations 

Available base environments include `base_env` (6DoF) and `base_env_2d` (3DoF). The main difference is that the `base_env_2d` converts the standard 6DoF observations to 3DoF in the `set_obs()` function. 

In addition to the standard environments that use MuJoCo, a standalone environment (`standalone_env.py`) was added, that is independent from MuJoCo. This environment is used when running a controller independent from the simulator.

### Controllers
The main controller used is the MPCC. There is a 6DoF version (`controller.py`) and a 3DoF version (`controller_2d.py`). Note that the 3DoF version is still formulated as a 6DoF w.r.t the cost function (but not the dynamics). Therefore, the states (which are 3DoF) are extended with zeros and the quaternion corresponding to the 3DoF yaw angle is computed.

### Planners
The main planner is the `mission` planner. There is a 6DoF version (`mission.py`) and a 3DoF version (`mission_2d.py`). The main difference is that for the 3DoF planner, the waypoint coordinates corresponding to the `z-axis` and the angles `roll` and `pitch` are set to zero.

### Communication
The file `src/smallsat_sim/communication/udp.py` provides basic UDP communication capabilities. The main idea is to allow for e.g. the controller and simulator to run in separate threads, i.e. asynchronous threads. The `UdpBuffer` class provides a UDP buffer. It continuously listens for incoming UDP messages and updates the shared data buffer. It can be activated by calling the `.start` method, which asynchronously updates the `.data` property.

### Experiments
The `experiments` folder contains various examples, see [above](### Example experiments). New experiments can be added here to use different combinations of controllers, planners, models, and environments, which are created and operated in the top-level experiment script. For a basic example, see `experiments/astrobee_CL.py`.

## RL support using JAX

### Overview

The RL controller uses the MJX backend, and follows the same structure as the other controllers in SmallSatSim, with a few exceptions. Notably: 
- a training script is provided in `experiments/rl_training/train_astrobee.py`, 
- multiple benchmarking scripts and plotting utilities are provided in `experiments/rl_benchmarking/`,
- the RL training must happen in a vectorized environment defined in `envs/vec_env.py`,
- the environment and config files are in a seperate directory, `astrobee_rl`,
- the RL implementation uses different JAX-based helper functions that can also be found in `utils`,
- the agent trains in a simplified environment defined in `utils/xml_parser_lightweights.py`.

The structure of the core implementation in `controllers/rl` is:

```
.
├── algorithms
│   ├── base_agent.py
│   ├── ppo.py
│   └── vpg.py
├── checkpoints
├── controller.py
├── modules
│   ├── base_network.py
│   ├── base_policy.py
│   ├── cnn_am.py
│   ├── mlp.py
│   └── transformer_am.py
├── runners
│   ├── on_policy_runner.py
│   └── runner_utils.py
└── storage
    └── replay_buffer.py
```

The implementation of the RL agent is heavily geared towards running model-free on-policy actor-critic algorithms such as [Proximal Policy Optimization](https://arxiv.org/pdf/1707.06347) (PPO). However, since the MJX background and the vectorized environment are agnostic of the RL technique used, other methods could be added.

The vectorized environment in `envs/vec_env.py` is crucial training the RL agent: among other necessary features, it contains the JIT-compiled MJX step function and the transition function that applies actions to the environment.

The easiest way to get a good overview is to step through one iteration of the training loop using the debugger. The object that coordinates all RL activities is the `OnPolicyRunner` in `on_policy_runner.py`. In between pretraining, training and evaluation, the neural network weights of the actor and critic are saved as [pickle](https://docs.python.org/3/library/pickle.html) files.

To achieve adaptive policies, the residual wrench (actual - desired) commanded by the smallsat's actuator's is fed to the base policy during training and depolyment. During the latter, the actual wrench (called extrinsics) is estimated by the `CNNAdaptationModule` or `TransformerAdaptationModule`. This is turn is trained with supervised learning, using the state-action history and the ground extrinsics from simulation. The active adaptation architecture can be selected through `EnvConfig.control.RL.am_architecture` (default `"transformer"`).

When failures are enabled, policy training uses a precomputed Astrobee failure-scenario library. After nominal training, failures are sampled online at reset time based on the current regulation wrench so the policy sees feasible but task-relevant authority loss rather than globally hard or impossible faults.

To deploy the trained RL controller, run `experiments/astrobee_RL.py`.

### Training instructions

#### Before training (or deploying) the RL agent

1. Set the controller hyperparameters in `envs/astrobee_rl/cfg/config.py`.
2. Instantiate perturbations and disturbances for the *vectorized* environment in `envs/astrobee_rl/env.py`. They work for an arbitrary number of environments.

Both steps work exactly in the same way as for the other controllers.

#### Training the RL controller

The training script provided in `experiments/rl_training/train_astrobee.py` directs the pretraining, training and evaluation of the actor and critic networks. To skip one of these steps, comment the corresponding function call. The necessary data for pretraining is generated with a vectorized PD controller found in `controllers/pd/vectorized_controller.py`.

The agent should be trained on GPU using a CUDA Docker image. It is also possible to log videos and data from experiments, as well as to monitor training runs online using Weights & Biases. More details can be found in `README.md`.

*Note: to use W&B, add your API key to `wandb_config.py`.*

#### Benchmarking the RL training methods

To benchmark all different variants of the algorithm against each other, run `train_test_all.sh`. To plot the results, run `experiments/rl_benchmarking/rl_plotter.py` (this only works with `python`, not `mjpython`).


## UDP Controller
The experiments folder contains examples to run a controller and simulation environment in separate threads. The file `udp_controller_2d.py` listens for messages on a given port (in this case the messages are the observations from the simulation environment) and sends the computed control inputs on a separate port. The controller is run at a given frequency `Ts` and should compute control inputs at this frequency, regardless of whether new observations are available (hence the simulator thread and controller thread are asynchronous). 

The file `astrobee_CL_2d_udp_external.py` runs the simulation loop. Observations are sent to a given port, and control inputs are received from the controller. This thread will propagate the simulation every time a new control input is received (as opposed to at a given frequency).

`udp_echo.py` provides a UPD receiver. This can be used for testing / development purposes when working with UDP.

### How to run a UDP experiment
We provide an example based on the Astrobee system with 3DoF dynamics. \
Note that while the dynamics are 3Dof, i.e. the Astrobee is constrained to a 2 dimensional space, the simulation environment still provides 6DoF observations. The observations are converted to 3DoF in the `BaseEnv2D`.

Start the controller in a first terminal (this is the controller thread)
```bash
python experiments/udp_controller_2d.py
```

Start the simulation loop in a second terminal (this is the simulation thread)
```bash
python experiments/astrobee_CL_2d_udp_external.py
```

View the simulation under http://localhost:8080/vnc.html.
Note that the simulation will not update the planned trajectory of the MPC controller, since the simulation thread only receives data regarding the current control input.


## Testing

Pytest is included as a project dependency in `pyproject.toml`.

Run all tests (from the repo root):

```bash
PYTHONPATH=src pytest -q src/tests
```

Run a single test file:

```bash
PYTHONPATH=src pytest -q src/tests/test_functional_rollout.py
```

If you see missing-module errors, make sure your environment has the full
project dependencies installed.


## Python Debugger 
If you are using VSCode and want to run the python debugger in the Docker run the following command in a terminal that is attached to the Docker. Make sure to replace `experiments/test.py` with the file you want to debug.

```bash
pip install debugpy -t /tmp && python /tmp/debugpy --wait-for-client --listen 0.0.0.0:5678 experiments/test.py
```

Then go to the `Run and Debug` in VS Code (left side bar, the play button with the bug symbol). Make sure it is set to `Python Debugger: Remote Attach` in the top left. Then press the play button. 

### FAQ
If you have issues with e.g. `torch` imports try running the following, in a terminal that is attached to the Docker
```bash
pip install importlib-metadata==8.4.0
```

You can often fix rendering issues by setting permissions on the host side
```bash
xhost +local:root
```

### Known Issues
- The PID, LQR and MPC controllers are broken.
- The Cubesat environment is broken.

## License

Copyright (c) 2024, Jet Propulsion Laboratory, California Institute of Technology. All rights reserved. JPL NTR 53088.

This software is licensed under the Apache License, version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0.

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.
