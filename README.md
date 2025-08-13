# smallsat-sim
![image](docs/imgs//3DoF_Spacestation.png)
SmallSatSim is a MuJoCo-based simulation environment for microgravity robotics research. SmallSatSim is easily repurposable to other robotics research scenarios including, notably, maritime robotics. SmallSatSim provides:

- easy parallelization for reinforcement learning
- system dynamics modification for model perturbations and disturbances
- built-in environments for some common microgravity research scenarios
- interfacing with MuJoCo's high-fidelity dynamics and visualization utilities
- differentiable custom dynamics for some scenarios, such as the free-flyer dynamics using CasADi

SmallSatSim emerged from the SmallSat Steward project, a collaboration between researchers at Caltech's Jet Propulsion Laboratory (now University of Southern California) and the University of Michigan's Space Systems Laboratory. This project is open-sourced under an Apache 2.0 license.

## Development Practices

We will be using a "git flow" style and branching off of main for feature adds.

Please develop significant features on separate branches named `feature/FEATURE_NAME`, branching off of `main`. For small changes, committing directly to `main` is fine; for larger changes merqe requests are encouraged.

## Docker

### macOs

Run the setup script
```bash
cd .setup/macos
bash setup.sh
```
### Ubuntu

Run the setup script
```bash
cd .setup/ubuntu
bash setup.sh
```

### Running Docker
Build the docker
```bash
smallsat up
```

Run the docker
```bash
smallsat run
```

Additional terminals can be attached to the docker using 
```bash
smallsat attach
```

Make sure to shut down the docker when you are done developing
```bash
smallsat down
```

### Using the provided image
The `smallsat.tar`file contains the pre-built image. You can load this image by running

```bash
docker load -i smallsat.tar
``` 

Then run the docker as usual
```bash
smallsat run
```

### Using a Base Image
Download base image (.tar file) and add it to `code/` directory \
Load prebuilt image `docker load -i base_image.tar` \
Allow patching over local docker image by running the following in your terminal
```bash
docker context use default
docker buildx use default
```

Patch `Dockerfile` on top of the `base_image` base image by running 
```bash
docker buildx build --platform="linux/arm64" -t smallsat:latest --build-arg BASE_IMAGE=base_image .  --progress=plain
```

If you are not using an `arm64` base, make sure to change the `platform`. \
Save built image as .tar: `docker save -o smallsat.tar smallsat:latest`\
Run docker container: `docker run -it --rm -v .:/code smallsat:latest`

## Structure

### Models

### Enviroments
The environments are used to set up the MuJoCo simulations. The environments set up the symbolic model as well as the observations. The main functionalities the environment provides are:

`step()` : Simulates the environment for one timestep using the control inputs \
`set_obs()` : Sets all observations (position, orientation, velocity, angular velocity) \
`get_obs()` : Returns the current (noisy) observations 

Available base environments include `base_env` (6DoF) and `base_env_2d` (3DoF). The main difference is that the `base_env_2d` converts the standard 6DoF observations to 3DoF in the `set_obs()` function. 

In addition to the standard environments that use MuJoCo, a standalone environment (`standalone_env.py`) was added, that is indepenent from MuJoCo. This environment is used when running a controller independent from the simulator.

### Controllers
The main controller used is the MPCC. There is a 6DoF version (`controller.py`) and a 3DoF version (`controller_2d.py`). Note that the 3DoF version is still formulated as a 6DoF w.r.t the cost function (but not the dynamics). Therefore, the states (which are 3DoF) are extended with zeros and the quaternion corresponding to the 3DoF yaw angle is computed.

### Planners
The main planner is the `mission` planner. There is a 6DoF version (`mission.py`) and a 3DoF version (`mission_2d.py`). The main difference is that for the 3DoF planner, the waypoint coordinates corresponding to the `z-axis` and the angles `roll` and `pitch` are set to zero.

### Communication
The file `src/smallsat_sim/communication/udp.py` provides basic UDP communication capabilities. The main idea is to allow for e.g. the controller and simulator to run in seperate threads, i.e. asynchronous threads. The `UdpBuffer` class provides a UDP buffer. It continuously listens for incoming UDP messages and updates the shared data buffer. It can be activated by calling the `.start` method, which asynchronously updates the `.data` property.

### Example Experiments
The `experiments` folder contains various examples. 

`astrobee_CL.py` : Uses the 6DoF Astrobee model, MPCC controller and mission planner. \
`astrobee_CL_2d.py` : Uses the 3DoF Astrobee model, MPCC controller and mission planner. \
`udp_echo.py` : Provides a UPD receiver. This can be used for testing / development purposes when working with UDP. \
`udp_controller.py` : 6DoF MPCC controller that is run in a separate thread from the simulation. It listens for messages on a given port (in this case the messages are the 6DoF observations from the simulation environment) and sends the computed 6DoF control inputs on a separate port. \
`udp_controller_2d.py` : 3DoF MPCC controller that is run in a separate thread from the simulation. It listens for messages on a given port (in this case the messages are the 6DoF observations from the simulation environment) and sends the computed 3DoF control inputs on a separate port. \
`astrobee_CL_udp_external.py` : Runs the 6DoF simulation loop. Observations are sent to a given port, and 6DoF control inputs are received from the controller. \
`astrobee_CL_2d_udp_external.py` : Runs the 3DoF simulation loop. Observations are sent to a given port, and 3DoF control inputs are received from the controller. 

## How to run a standard experiment
In a terminal that is attached to the docker
```bash
python experiments/my_experiment.py
```

## UDP Controller
The experiments folder contains examples to run a controller and simulation environment in separate threads. The file `udp_controller_2d.py` listens for messages on a given port (in this case the messages are the observations from the simulation environment) and sends the computed control inputs on a separate port. The controller is run at a given frequency `Ts` and should compute control inputs at this frequency, regardless of whether new observations are available (hence the simulator thread and controller thread are asynchronous). 

The file `astrobee_CL_2d_udp_external.py` runs the simulation loop. Observations are sent to a given port, and control inputs are received from the controller. This thread will propagate the simulation every time a new control input is received (as opposed to at a given frequency).

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


## Python Debugger 
If you are using VSCode and want to run the python debugger in the docker run the following command in a terminal that is attached to the Docker. Make sure to replace `experiments/test.py` with the file you want to debug.

```bash
pip install debugpy -t /tmp && python /tmp/debugpy --wait-for-client --listen 0.0.0.0:5678 experiments/test.py
```

Then go to the `Run and Debug` in VS Code (left side bar, the play button with the bug symbol). Make sure it is set to `Python Debugger: Remote Attach` in the top left. Then press the play button. 

### FAQ
If you have issues with e.g. `torch` imports try running the following, in a terminal that is attached to the docker
```bash
pip install importlib-metadata==8.4.0
```

### Known Issues
- The PID, LQR and MPC controllers are broken.
- The Cubesat is broken.

## License

Copyright (c) 2024, Jet Propulsion Laboratory, California Institute of Technology. All rights reserved. JPL NTR 53088.

This software is licensed under the Apache License, version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0.

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.
