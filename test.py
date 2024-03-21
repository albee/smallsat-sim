import mujoco
import mujoco.viewer
from mujoco import mjx

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
                 forcerange="-100 100"
                 ctrlrange="-1 1"
        />
    </actuator>
</mujoco>

"""

# create model and data objects
model = mujoco.MjModel.from_xml_string(xml)
data = mujoco.MjData(model)

viewer = mujoco.viewer.launch(model,data)
