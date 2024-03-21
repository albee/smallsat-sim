import mujoco
import mujoco.viewer
from mujoco import mjx
import numpy as np
import time

xml="""
<mujoco model="space_cube">
    <option gravity="0 0 0"/> <!-- Sets gravity to zero to simulate space -->
    <asset>
        <!-- Define any textures, materials, or meshes here -->
    </asset>

    <worldbody>
        <body name="cube" pos="0 0 0">
            <freejoint/>
            <geom type="box" size="0.1 0.1 0.1" mass="1"/> <!-- Defines the cube -->
            <site name="thruster_site" pos="0.1 0 0" size="0.01"/> <!-- Position for thruster action -->
        </body>
    </worldbody>

    <actuator>
        <general name="cube_thruster"
                 site="thruster_site"
                 gear="1 0 0"
                 forcerange="-0.5 0.5"
                 ctrlrange="-1 1"
        />
    </actuator>
</mujoco>

"""

# create model and data objects
model = mujoco.MjModel.from_xml_string(xml)
data = mujoco.MjData(model)


with mujoco.viewer.launch_passive(model, data) as viewer:

    # Close the viewer automatically after 30 wall-seconds.
    start_time = time.time()

    while viewer.is_running():
        real_time = time.time() - start_time

        sim_time = data.time

        if sim_time < real_time:
        # example control
            force = np.cos(sim_time)

            # apply force to the actuator
            actuator_index = model.actuator('cube_thruster').id
            data.ctrl[actuator_index] = force

            # Step simulation
            mujoco.mj_step(model,data)

            # Update renderer
            viewer.sync()