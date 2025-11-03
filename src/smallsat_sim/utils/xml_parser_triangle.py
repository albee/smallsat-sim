from smallsat_sim.utils.xml_parser import generate_mujoco_xml_gateway


def xmlify(array: list):
    """
    Converts a list of numbers into a string
    """
    return " ".join([str(x) for x in array])


def _compiler_attributes(env_config) -> str:
    """
    Filtering out the convexhull compiler option since it was removed in MuJoCo 3.2.7.
    """
    compiler_config = getattr(env_config, "compiler", {}) or {}
    filtered = {
        key: value for key, value in compiler_config.items() if key != "convexhull"
    }
    if not filtered:
        return ""
    return " " + " ".join(f'{key}="{value}"' for key, value in filtered.items())


def generate_mujoco_xml(env_config, model_config, use_flat_bed: bool = True):
    if use_flat_bed:
        return generate_mujoco_xml_flat_bed(
            env_config, model_config, env_config.dof == "6d"
        )
    else:
        return generate_mujoco_xml_gateway(
            env_config, model_config, env_config.dof == "6d"
        )


def generate_mujoco_xml_flat_bed(env_config, model_config, free_floating: bool = False):
    """
    This function generates the xml string which is fed to MuJoCo for setting up the sim environment
    """
    # Extract relevant quantities from configs
    geoms = model_config.Geoms
    bodies = env_config.Bodies
    thrusters = model_config.Thrusters
    props = model_config.pp
    joint_str = "<freejoint/>"
    if not free_floating:
        joint_str = """
            <joint name="slide_x" type="slide" axis="1 0 0" />
            <joint name="slide_y" type="slide" axis="0 1 0" />
            <joint name="yaw" type="hinge" axis="0 0 1" />
        """
        for body in bodies.bodies_list:
            body.euler[0] = 0
            body.euler[1] = 0

    xml_content = f"""<?xml version="1.0" encoding="utf-8"?>
    <mujoco model="{env_config.model}">
        <compiler texturedir="src/smallsat_sim/model" meshdir="src/smallsat_sim/model" eulerseq="XYZ"{_compiler_attributes(env_config)}/>
        <visual>
            <headlight ambient="{env_config.visual['headlight']['ambient']}" specular="{env_config.visual['headlight']['specular']}" diffuse="{env_config.visual['headlight']['diffuse']}"/>
        </visual>
        <option gravity="0 0 0"/>
        <default>
            <default class="visual">
                <geom group="2" type="mesh" contype="0" conaffinity="0"/>
            </default>
            <default class="collision">
                <geom group="3" type="mesh"/>
            </default>
        </default>
        <asset>
            <texture type="skybox" file="gateway/stars.png"/>
            <mesh file="smallsat_hardware_2d/meshes/triangle.obj"/>
            <material name="triangle_material" specular="0.5" shininess="0.25" rgba="0.2 0.2 0.8 1"/>
        </asset>
        <worldbody>
            <body name="triangle_robot" pos="0 0 0.1">
                {joint_str}
                <inertial pos="{xmlify(props.com_offset)}" mass="{props.mass}" diaginertia="{xmlify(props.diag_inertia)}"/>
                <geom mesh="triangle" material="triangle_material" class="visual"/>
                <geom type="box" size="0.16 0.16 0.16" class="collision"/>
            </body>
        </worldbody>
    </mujoco>
    """
    return xml_content


# File can be executed to test xml output
if __name__ == "__main__":
    # from smallsat_sim.envs.astrobee_2d.cfg.config import EnvConfig
    # from smallsat_sim.model.astrobee_2d.cfg.config import ModelConfig
    from smallsat_sim.envs.smallsat_hardware_2d.cfg.config import EnvConfig
    from smallsat_sim.model.smallsat_hardware_2d.cfg.config import ModelConfig

    model_config = ModelConfig()
    env = EnvConfig()
    xml = generate_mujoco_xml(env, model_config)
    with open("smallsat_hardware_triangle.xml", "w") as f:
        f.write(xml)
    print(xml)
