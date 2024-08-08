def xmlify(array: list):
    """
    Converts a list of numbers into a string
    """
    return " ".join([str(x) for x in array])


def generate_mujoco_xml(env_config, model_config):
    """
    This function generates the xml string which is fed to MuJoCo for setting up the sim environment
    """
    # Extract relevant quantities from configs
    geoms = model_config.Geoms
    bodies = env_config.Bodies
    thrusters = model_config.Thrusters
    props = model_config.pp

    # Beginning of xml file
    # Defines general options for the environment
    # Loads all assets such as .obj files and corresponding meshes/texture
    xml_content = f"""<?xml version="1.0" encoding="utf-8"?>
    <mujoco model="{env_config.model}">
        <compiler convexhull="{env_config.compiler['convexhull']}" texturedir="smallsat_sim/model" meshdir="smallsat_sim/model" eulerseq="XYZ"/>
        <visual>
            <headlight ambient="{env_config.visual['headlight']['ambient']}" specular="{env_config.visual['headlight']['specular']}" diffuse="{env_config.visual['headlight']['diffuse']}"/>
        </visual>
        <visual>
            <global offwidth="1920" offheight="1080"/>
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
            <texture type="2d" name="Canadarm_BaseColor" file="gateway/Canadarm_BaseColor.png"/>
            <material name="Canadarm" texture="Canadarm_BaseColor" specular="0.0" shininess="0.5"/>
            <texture type="2d" name="CrewAirlock_BaseColor" file="gateway/CrewAirlock_BaseColor.png"/>
            <material name="CrewAirlock" texture="CrewAirlock_BaseColor" specular="0.0" shininess="0.5"/>
            <texture type="2d" name="Espirit_Refueller_BaseColor" file="gateway/Espirit_Refueller_BaseColor.png"/>
            <material name="Espirit_Refueller" texture="Espirit_Refueller_BaseColor" specular="0.0" shininess="0.5"/>
            <texture type="2d" name="Habitation_And_Logistics_Outpost_BaseColor" file="gateway/Habitation_And_Logistics_Outpost_BaseColor.png"/>
            <material name="Habitation_And_Logistics_Outpost" texture="Habitation_And_Logistics_Outpost_BaseColor" specular="0.0" shininess="0.5"/>
            <texture type="2d" name="Human_Lander_System_BaseColor" file="gateway/Human_Lander_System_BaseColor.png"/>
            <material name="Human_Lander_System" texture="Human_Lander_System_BaseColor" specular="0.0" shininess="0.5"/>
            <texture type="2d" name="International_Habitation_Module_BaseColor" file="gateway/International_Habitation_Module_BaseColor.png"/>
            <material name="International_Habitation_Module" texture="International_Habitation_Module_BaseColor" specular="0.0" shininess="0.5"/>
            <texture type="2d" name="Logistics_Vehicle_BaseColor" file="gateway/Logistics_Vehicle_BaseColor.png"/>
            <material name="Logistics_Vehicle" texture="Logistics_Vehicle_BaseColor" specular="0.0" shininess="0.5"/>
            <texture type="2d" name="Orion_BaseColor" file="gateway/Orion_BaseColor.png"/>
            <material name="Orion" texture="Orion_BaseColor" specular="0.0" shininess="0.5"/>
            <texture type="2d" name="Power_And_Propulsion_Element_BaseColor" file="gateway/Power_And_Propulsion_Element_BaseColor.png"/>
            <material name="Power_And_Propulsion_Element" texture="Power_And_Propulsion_Element_BaseColor" specular="0.0" shininess="0.5"/>
            <mesh file="gateway/gateway_0.obj"/>
            <mesh file="gateway/gateway_1.obj"/>
            <mesh file="gateway/gateway_2.obj"/>
            <mesh file="gateway/gateway_3.obj"/>
            <mesh file="gateway/gateway_4.obj"/>
            <mesh file="gateway/gateway_5.obj"/>
            <mesh file="gateway/gateway_6.obj"/>
            <mesh file="gateway/gateway_7.obj"/>
            <mesh file="gateway/gateway_8.obj"/>
            <mesh file="gateway/gateway_simple_0.obj"/>
            <mesh file="gateway/gateway_simple_1.obj"/>
            <mesh file="gateway/gateway_simple_2.obj"/>
            <mesh file="gateway/gateway_simple_3.obj"/>
            <mesh file="gateway/gateway_simple_4.obj"/>
            <mesh file="gateway/gateway_simple_5.obj"/>
            <mesh file="gateway/gateway_simple_6.obj"/>
            <mesh file="gateway/gateway_simple_7.obj"/>
            <mesh file="gateway/gateway_simple_8.obj"/>
            <mesh file="gateway/gateway_simple_9.obj"/>
            <mesh file="gateway/gateway_simple_10.obj"/>
            <mesh file="gateway/gateway_simple_11.obj"/>
            <mesh file="gateway/gateway_simple_12.obj"/>
            <mesh file="gateway/gateway_simple_13.obj"/>
            <mesh file="gateway/gateway_simple_14.obj"/>
            <mesh file="gateway/gateway_simple_15.obj"/>
            <mesh file="gateway/gateway_simple_16.obj"/>
            <mesh file="gateway/gateway_simple_17.obj"/>
            <mesh file="gateway/gateway_simple_18.obj"/>
            <mesh file="gateway/gateway_simple_19.obj"/>
            <mesh file="gateway/gateway_simple_20.obj"/>
            <mesh file="gateway/gateway_simple_21.obj"/>
            <mesh file="gateway/gateway_simple_22.obj"/>
            <mesh file="gateway/gateway_simple_23.obj"/>
            <mesh file="gateway/gateway_simple_24.obj"/>
            <mesh file="gateway/gateway_simple_25.obj"/>
            <texture type="skybox" file="gateway/stars.png"/>
    """

    # Check whether astrobee or cubesat is used.
    # This code defines the materials and meshes.
    if model_config.name == "astrobee":
        xml_content += f"""        <!--Load astrobee model components-->
            <texture type="2d" name="black" file="astrobee/meshes/black.png"/>
            <material name="Material_001" texture="black" specular="0.5" shininess="0.25"/>
            <material name="Material_001.001" specular="0.5" shininess="0.25" rgba="0.100000 0.100000 0.100000 1.000000"/>
            <material name="Material_001.002" specular="0.5" shininess="0.25" rgba="0.100000 0.100000 0.100000 1.000000"/>
            <material name="Material_001_003" texture="black" specular="0.5" shininess="0.25"/>
            <material name="Material_002" specular="0.5" shininess="0.25" rgba="0.640000 0.005111 0.000000 1.000000"/>
            <material name="Material_002.001" specular="0.5" shininess="0.25" rgba="0.568443 0.568443 0.568443 1.000000"/>
            <material name="Material_002.002" specular="0.5" shininess="0.25" rgba="0.568443 0.568443 0.568443 1.000000"/>
            <material name="Material_003" specular="0.5" shininess="0.25" rgba="0.000000 0.000000 0.000000 1.000000"/>
            <texture type="2d" name="skin_bumble" file="astrobee/meshes/skin_bumble.png"/>
            <material name="Material_003.001" texture="skin_bumble" specular="0.5" shininess="0.25"/>
            <material name="Material_003.002" texture="skin_bumble" specular="0.5" shininess="0.25"/>
            <material name="Material_004" specular="0.5" shininess="0.25" rgba="0.001995 0.000000 0.640000 1.000000"/>
            <material name="Material_005" specular="0.5" shininess="0.25" rgba="0.000000 0.000000 0.000000 1.000000"/>
            <material name="Material_007" specular="0.5" shininess="0.25" rgba="0.015079 0.015079 0.015079 1.000000"/>
            <material name="Material_008" specular="0.5" shininess="0.25" rgba="0.018920 0.018920 0.018920 1.000000"/>
            <material name="Material_009" specular="0.5" shininess="0.25" rgba="0.011736 0.011736 0.011736 1.000000"/>
            <material name="Material_010" specular="0.5" shininess="0.25" rgba="0.050976 0.050976 0.050976 1.000000"/>
            <material name="Material_011" specular="0.5" shininess="0.25" rgba="0.640000 0.450660 0.080787 1.000000"/>
            <material name="Material_012" specular="0.5" shininess="0.25" rgba="0.005755 0.005755 0.005755 1.000000"/>
            <material name="Material_014" specular="0.5" shininess="0.25" rgba="0.241980 0.625158 0.800000 1.000000"/>
            <material name="Material_017" specular="0.5" shininess="0.25" rgba="0.000000 0.640000 0.003578 1.000000"/>
            <material name="Material_019" specular="0.5" shininess="0.25" rgba="0.752312 0.752312 0.752312 1.000000"/>
            <material name="Material_020" specular="0.5" shininess="0.25" rgba="0.050976 0.050976 0.050976 1.000000"/>
            <material name="Material_021" specular="0.5" shininess="0.25" rgba="0.138684 0.138684 0.138684 1.000000"/>
            <material name="Material_022" specular="0.5" shininess="0.25" rgba="0.110067 0.110067 0.110067 1.000000"/>
            <material name="Material_023" specular="0.5" shininess="0.25" rgba="0.640000 0.452536 0.017686 1.000000"/>
            <material name="Material_024" specular="0.5" shininess="0.25" rgba="0.000000 0.085228 0.000689 1.000000"/>
            <material name="Material_025" specular="0.5" shininess="0.25" rgba="0.202354 0.202354 0.202354 1.000000"/>
            <material name="Material_027" specular="0.5" shininess="0.25" rgba="0.145780 0.710281 0.800000 1.000000"/>
            <material name="Material_028" specular="0.5" shininess="0.25" rgba="0.171227 0.171227 0.171227 1.000000"/>
            <mesh file="astrobee/meshes/astrobee_0.obj"/>
            <mesh file="astrobee/meshes/astrobee_1.obj"/>
            <mesh file="astrobee/meshes/astrobee_2.obj"/>
            <mesh file="astrobee/meshes/astrobee_3.obj"/>
            <mesh file="astrobee/meshes/astrobee_4.obj"/>
            <mesh file="astrobee/meshes/astrobee_5.obj"/>
            <mesh file="astrobee/meshes/astrobee_6.obj"/>
            <mesh file="astrobee/meshes/astrobee_7.obj"/>
            <mesh file="astrobee/meshes/astrobee_8.obj"/>
            <mesh file="astrobee/meshes/astrobee_9.obj"/>
            <mesh file="astrobee/meshes/astrobee_10.obj"/>
            <mesh file="astrobee/meshes/astrobee_11.obj"/>
            <mesh file="astrobee/meshes/astrobee_12.obj"/>
            <mesh file="astrobee/meshes/astrobee_13.obj"/>
            <mesh file="astrobee/meshes/astrobee_14.obj"/>
            <mesh file="astrobee/meshes/astrobee_15.obj"/>
            <mesh file="astrobee/meshes/astrobee_16.obj"/>
            <mesh file="astrobee/meshes/astrobee_17.obj"/>
            <mesh file="astrobee/meshes/astrobee_18.obj"/>
            <mesh file="astrobee/meshes/astrobee_19.obj"/>
            <mesh file="astrobee/meshes/astrobee_20.obj"/>
            <mesh file="astrobee/meshes/astrobee_21.obj"/>
            <mesh file="astrobee/meshes/astrobee_22.obj"/>
            <mesh file="astrobee/meshes/astrobee_23.obj"/>
            <mesh file="astrobee/meshes/astrobee_24.obj"/>
            <mesh file="astrobee/meshes/astrobee_25.obj"/>
            <mesh file="astrobee/meshes/astrobee_26.obj"/>
            <mesh file="astrobee/meshes/astrobee_27.obj"/>
            <mesh file="astrobee/meshes/astrobee_28.obj"/>
            """

    else:
        # Add additional, smallsat-specific meshes
        # Add temp folder to avoid multiple definition of same mesh
        temp = []
        for geom in geoms.geom_list:
            if geom.type == "mesh":
                if geom.name not in temp:
                    temp.append(geom.name)
                    xml_content += f"        <mesh name='{geom.name}' file='{geom.mesh}' scale='{xmlify(geom.asset_scale)}'/>\n"

    # Import meshes of lunar gateway
    xml_content += """      </asset>
    <worldbody>
        <body name="gateway_full">
            <geom mesh="gateway_0" material="Canadarm" class="visual"/>
            <geom mesh="gateway_1" material="Habitation_And_Logistics_Outpost" class="visual"/>
            <geom mesh="gateway_2" material="Espirit_Refueller" class="visual"/>
            <geom mesh="gateway_3" material="Human_Lander_System" class="visual"/>
            <geom mesh="gateway_4" material="Logistics_Vehicle" class="visual"/>
            <geom mesh="gateway_5" material="CrewAirlock" class="visual"/>
            <geom mesh="gateway_6" material="Orion" class="visual"/>
            <geom mesh="gateway_7" material="Power_And_Propulsion_Element" class="visual"/>
            <geom mesh="gateway_8" material="International_Habitation_Module" class="visual"/>
            <geom mesh="gateway_simple_0" class="collision"/>
            <geom mesh="gateway_simple_1" class="collision"/>
            <geom mesh="gateway_simple_2" class="collision"/>
            <geom mesh="gateway_simple_3" class="collision"/>
            <geom mesh="gateway_simple_4" class="collision"/>
            <geom mesh="gateway_simple_5" class="collision"/>
            <geom mesh="gateway_simple_6" class="collision"/>
            <geom mesh="gateway_simple_7" class="collision"/>
            <geom mesh="gateway_simple_8" class="collision"/>
            <geom mesh="gateway_simple_9" class="collision"/>
            <geom mesh="gateway_simple_10" class="collision"/>
            <geom mesh="gateway_simple_11" class="collision"/>
            <geom mesh="gateway_simple_12" class="collision"/>
            <geom mesh="gateway_simple_13" class="collision"/>
            <geom mesh="gateway_simple_14" class="collision"/>
            <geom mesh="gateway_simple_15" class="collision"/>
            <geom mesh="gateway_simple_16" class="collision"/>
            <geom mesh="gateway_simple_17" class="collision"/>
            <geom mesh="gateway_simple_18" class="collision"/>
            <geom mesh="gateway_simple_19" class="collision"/>
            <geom mesh="gateway_simple_20" class="collision"/>
            <geom mesh="gateway_simple_21" class="collision"/>
            <geom mesh="gateway_simple_22" class="collision"/>
            <geom mesh="gateway_simple_23" class="collision"/>
            <geom mesh="gateway_simple_24" class="collision"/>
            <geom mesh="gateway_simple_25" class="collision"/>
        </body>
    </worldbody>

    <worldbody>
"""
    # Generates the free-floating bodiesin the simulation
    if model_config.name == "astrobee":
        for body in bodies.bodies_list:
            xml_content += f"""         <body name='{body.name}' pos='{xmlify(body.pos)}' euler='{xmlify(body.euler)}'>
            <freejoint/>
            <!Inertial properties are have been taken from Astrobee repo>
            <!(https://github.com/nasa/astrobee/blob/master/astrobee/config/worlds/iss.config)>
            <inertial pos="{xmlify(props.com_offset)}" mass="{props.mass}" diaginertia="{xmlify(props.diag_inertia)}"/>
            <geom mesh="astrobee_0" material="Material_003.002" class="visual"/>
            <geom mesh="astrobee_1" material="Material_001.002" class="visual"/>
            <geom mesh="astrobee_2" material="Material_002.002" class="visual"/>
            <geom mesh="astrobee_3" material="Material_003.001" class="visual"/>
            <geom mesh="astrobee_4" material="Material_001.001" class="visual"/>
            <geom mesh="astrobee_5" material="Material_002.001" class="visual"/>
            <geom mesh="astrobee_6" material="Material_001" class="visual"/>
            <geom mesh="astrobee_7" material="Material_001_003" class="visual"/>
            <geom mesh="astrobee_8" material="Material_028" class="visual"/>
            <geom mesh="astrobee_9" material="Material_027" class="visual"/>
            <geom mesh="astrobee_10" material="Material_025" class="visual"/>
            <geom mesh="astrobee_11" material="Material_024" class="visual"/>
            <geom mesh="astrobee_12" material="Material_023" class="visual"/>
            <geom mesh="astrobee_13" material="Material_022" class="visual"/>
            <geom mesh="astrobee_14" material="Material_021" class="visual"/>
            <geom mesh="astrobee_15" material="Material_020" class="visual"/>
            <geom mesh="astrobee_16" material="Material_017" class="visual"/>
            <geom mesh="astrobee_17" material="Material_014" class="visual"/>
            <geom mesh="astrobee_18" material="Material_012" class="visual"/>
            <geom mesh="astrobee_19" material="Material_011" class="visual"/>
            <geom mesh="astrobee_20" material="Material_010" class="visual"/>
            <geom mesh="astrobee_21" material="Material_009" class="visual"/>
            <geom mesh="astrobee_22" material="Material_008" class="visual"/>
            <geom mesh="astrobee_23" material="Material_007" class="visual"/>
            <geom mesh="astrobee_24" material="Material_005" class="visual"/>
            <geom mesh="astrobee_25" material="Material_004" class="visual"/>
            <geom mesh="astrobee_26" material="Material_003" class="visual"/>
            <geom mesh="astrobee_27" material="Material_002" class="visual"/>
            <geom mesh="astrobee_28" material="Material_019" class="visual"/>
            <geom type="box" size="0.16 0.16 0.16" class="collision"/>\n"""
            for thruster in thrusters.thruster_list:
                xml_content += f"            <site name='{body.name}_{thruster.site}' pos='{xmlify(thruster.pos)}' size='{thruster.size}'/>\n"

            xml_content += "        </body>\n"
    else:
        # Define all free floating bodies
        for body in bodies.bodies_list:
            xml_content += f"        <body name='{body.name}' pos='{xmlify(body.pos)}' euler='{xmlify(body.euler)}'>\n"
            xml_content += f"""            <freejoint/>
            <!Inertial properties are calculated from box with 3U dimensions and density 1000kg/m3>
            <inertial pos="{xmlify(props.com_offset)}" mass="{props.mass}" diaginertia="{xmlify(props.diag_inertia)}"/>
"""
            for geom in geoms.geom_list:
                if geom.type == "mesh":
                    xml_content += f"            <geom type='mesh' mesh='{geom.name}' pos='{xmlify(geom.pos)}' euler='{xmlify(geom.euler)}'/>\n"
                else:
                    xml_content += f"            <geom type='{geom.type}' size='{xmlify(geom.size)}' pos='{xmlify(geom.pos)}' euler='{xmlify(geom.euler)}'/>\n"
            for thruster in thrusters.thruster_list:
                xml_content += f"            <site name='{body.name}_{thruster.site}' pos='{xmlify(thruster.pos)}' size='{thruster.size}'/>\n"
                xml_content += "        </body>\n"

    xml_content += """    </worldbody>
"""
    xml_content += """    <!--Thruster sites are placed based on the description of Astrobee's thrusters-->
"""
    xml_content += """    <actuator>
"""
    for body in bodies.bodies_list:
        # Create thruster sites
        for thruster in thrusters.thruster_list:
            unique_site_name = f"{body.name}_{thruster.site}"
            xml_content += f"""        <general name="{body.name}_{thruster.name}"
                        site="{unique_site_name}"
                        gear="{xmlify(thruster.gear)}"
                        forcerange="{xmlify(thruster.forcerange)}"
                        ctrlrange="{xmlify(thruster.ctrlrange)}"
                        forcelimited="{thruster.forcelimited}"
                    />
"""
    xml_content += "    </actuator>\n"
    xml_content += "</mujoco>"
    return xml_content


# File can be executed to test xml output
if __name__ == "__main__":
    from smallsat_sim.envs.astrobee.cfg.config import EnvConfig
    from smallsat_sim.model.astrobee.cfg.config import ModelConfig

    model_config = ModelConfig()
    env = EnvConfig()
    xml = generate_mujoco_xml(env, model_config)
    with open("cubesat.xml", "w") as f:
        f.write(xml)
    print(xml)
