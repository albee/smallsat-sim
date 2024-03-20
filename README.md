# smallsat-sim

Simulation environment for SmallSat Steward. The simulation envrionment provides capabilities for:

- Dynamics modeling (with a differntiable framework?)
- Disturbance/degradation modeling 
- Collision modeling for realstic collision volumes
- Visualization

This is a ROS2 project (Foxy Fitzroy/Gazebo)!

## Notes

Suggestions for simulation frameworks:
- pybullet
- Gazebo
- MuJoCo

Suggestions for visualization:
- Gazebo built-in visualizer

## Installation
Create a virtual environment with python 3.10.11. This can be done with VSC's extension _Python Environment Manager_ or by creating one with pyenv as outlined in the following [article](https://robiokidenis.medium.com/how-to-install-multiple-python-on-your-mac-d20713740a2d).  

Activate the virutal environment by running `source venv/bin/activate`.  
Note that "venv" is the name of your virtual environment.  

Afterwards install all necessary packages by running the following command within the virutal environment:  
`pip install -r requirements.txt`.  

To check if the installation was successfull navigate to the folder `etc/` and run the test file:  
`python3 test.py`.  
It should show R2D2 dropping from the sky onto the ground.
Happy coding!


## Usage
TODO

## License
TODO
