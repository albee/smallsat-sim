from smallsat_sim.envs.cubesat.cfg.config import Body
from smallsat_sim.lib.cubesat.cfg.config import Thruster, Geom


def generate_mujoco_xml(env_config, lib_config):
    geoms = lib_config.Geoms()
    bodies = env_config.Bodies(env_config.num_bodies)
    thrusters = lib_config.Thrusters()

    xml_content = f'''<?xml version="1.0" encoding="utf-8"?>
<mujoco model="{env_config.model}">
    <compiler convexhull="{env.compiler['convexhull']}" texturedir="smallsat_sim/lib" meshdir="smallsat_sim/lib"/>
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
'''

    temp = []
    for geom in geoms.geom_list:
        if geom.type == 'mesh':
            if geom.name not in temp:
                temp.append(geom.name)
            
                xml_content += f"        <mesh name='{geom.name}' file='{geom.mesh}' scale='{geom.asset_scale}'/>\n"
    xml_content += '''    </asset>
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
'''

    for body in bodies.bodies_list:
        xml_content += f"        <body name='{body.name}' pos='{body.pos}' quat='{body.quat}'>\n"
        for geom in geoms.geom_list:
            if geom.type == 'mesh':
                xml_content += f"            <geom type='mesh' mesh='{geom.name}' pos='{geom.pos}' euler='{geom.euler}'/>\n"
            else:
                xml_content += f"            <geom type='{geom.type}' size='{geom.size}' pos='{geom.pos}' euler='{geom.euler}'/>\n"
        
        for thruster_name, thruster_obj in vars(thrusters).items():
            if isinstance(thruster_obj, Thruster):
                xml_content += f"            <site name='{thruster_obj.site}' pos='{thruster_obj.pos}' size='{thruster_obj.size}'/>\n"
        xml_content += "        </body>\n"

    xml_content += '''    </worldbody>
'''
    xml_content += '''    <!--Thruster sites are placed based on the description of Astrobee's thrusters-->
'''
    xml_content += '''    <actuator>
'''
    for thruster_name, thruster_obj in vars(thrusters).items():
        if isinstance(thruster_obj, Thruster):
            xml_content += f'''        <general name="{thruster_obj.name}"
                    site="{thruster_obj.site}"
                    gear="{thruster_obj.gear}"
                    forcerange="{thruster_obj.forcerange}"
                    ctrlrange="{thruster_obj.ctrlrange}"
                    forcelimited="{thruster_obj.forcelimited}"
                />
'''
    xml_content += '    </actuator>\n'
    xml_content += '</mujoco>'
    return xml_content

if __name__ == "__main__":
    from smallsat_sim.envs.cubesat.cfg.config import EnvConfig
    from smallsat_sim.lib.cubesat.cfg.config import LibConfig
    import argparse

    args = argparse.Namespace()
    args.num_bodies = 1
    lib = LibConfig()
    env = EnvConfig(args)
    xml = generate_mujoco_xml(env,lib)
    with open("cubesat.xml", 'w') as f:
        f.write(xml)
    print(xml)
