class Thruster():
    """Thruster class for defining thruster properties
    Defines both actuator and site properties for the thruster.
    """
    def __init__(self, name, pos, gear, site=None, forcerange=[0,1], ctrlrange=[0,0], forcelimited='false', ctrllimited='false', size=0.005) -> None:
        self.name = name
        self.site = name if site is None else site
        self.gear = gear
        self.pos = pos
        self.forcerange = forcerange
        self.ctrlrange = ctrlrange
        self.ctrllimited = ctrllimited
        self.forcelimited = forcelimited
        self.size = size 
        

class Geom():
    """Geom class for defining geom properties
    Especially considers mesh geoms.
    """
    def __init__(self, name, type, pos=None, euler='0 0 0', meshtype=None, asset_scale='1 1 1', size='1 1 1') -> None:
        self.name = name
        self.type = type
        if pos is None:
            self.pos = '0 0 0'
            print(f"Warning: No position provided for geom {name}. Using default position [0,0,0].")
        else:
            self.pos = pos
        self.euler = euler
        self.size = size
        if self.type == 'mesh':
            try:
                assert meshtype is not None
            except AssertionError:
                print(f"Error: No meshtype provided for geom {name}.")
            self.mesh = f'cubesat/meshes/{name}.{meshtype}'
            self.asset_scale = asset_scale


class PhysicalProperties():
    """Physical properties class for defining physical properties of the vehicle.
    """
    def __init__(self, length, width, height, density) -> None:
        self.length = length
        self.width = width
        self.height = height
        self.density = density

class LibConfig:
    """Library configuration class. Contains static vehicle information.
    """
    # Physical properties of the vehicle (consisting of the above geoms)
    pp = PhysicalProperties(length=0.1, width=0.1, height=0.3, density=1000)

    class Thrusters:
        """Thrusters class for defining all thrusters in a particular MuJoCo body.
        """
        n_thrusters = 12  # Total number of thrusters
        # Dictionary of thruster site positions.
        # The site positions correspond to Astrobee 
        # (https://ntrs.nasa.gov/citations/20160007769)
        site_information = {
            "LX+": {"pos": [0.05, 0.025, 0.05], "gear": [1, 0, 0]},
            "LX-": {"pos": [-0.05, 0.025, -0.05], "gear": [-1, 0, 0]},
            "RX+": {"pos": [0.05, -0.025, -0.05], "gear": [1, 0, 0]},
            "RX-": {"pos": [-0.05, -0.025, 0.05], "gear": [-1, 0, 0]},
            "AY+": {"pos": [0.025, 0.05, 0.05], "gear": [0, 1, 0]},
            "AY-": {"pos": [0.025, -0.05, -0.05], "gear": [0, -1, 0]},
            "FY+": {"pos": [-0.025, 0.05, -0.05], "gear": [0, 1, 0]},
            "FY-": {"pos": [-0.025, -0.05, 0.05], "gear": [0, -1, 0]},
            "LZ+": {"pos": [-0.025, -0.025, 0.143], "gear": [0, 0, 1]},
            "LZ-": {"pos": [0.025, -0.025, -0.15], "gear": [0, 0, -1]},
            "RZ+": {"pos": [0.025, 0.025, 0.143], "gear": [0, 0, 1]},
            "RZ-": {"pos": [-0.025, 0.025, -0.15], "gear": [0, 0, -1]}
        }
        # Create list of thrusters based on dictionary above
        thruster_list = []
        i = 0
        for key, value in site_information.items():
            i += 1
            thruster = Thruster(name=f'thruster{i}', pos=value["pos"], gear=value["gear"])
            thruster_list.append(thruster)

    class Geoms:
        """Geoms class for defining all geoms in a particular MuJoCo body.
        """
        geom_list = [
            Geom(name='cubesat_top', type='mesh', pos=[-0.0495, -0.0495, 0.1395], euler=[0, 0, 0], meshtype='stl', asset_scale=[0.001, 0.001, 0.001]),
            Geom(name='cubesat_middle', type='mesh', pos=[-0.05, -0.05, -0.15], euler=[0, 0, 0], meshtype='stl', asset_scale=[0.001, 0.001, 0.001]),
            Geom(name='cubesat_middle', type='mesh', pos=[-0.05, -0.05, -0.0535], euler=[0, 0, 0], meshtype='stl', asset_scale=[0.001, 0.001, 0.001]),
            Geom(name='cubesat_middle', type='mesh', pos=[-0.05, -0.05, 0.043], euler=[0, 0, 0], meshtype='stl', asset_scale=[0.001, 0.001, 0.001]),
            Geom(name='cubesat_bottom', type='mesh', pos=[-0.05, -0.05, -0.15], euler=[0, 0, 0], meshtype='stl', asset_scale=[0.001, 0.001, 0.001]),
        ]
