class Thruster():
    """Thruster class for defining thruster properties
    Defines both actuator and site properties for the thruster.
    """
    def __init__(self, name, pos, gear, site=None, forcerange='0 1', ctrlrange='0 1', forcelimited='true', size=0.005) -> None:
        self.name = name
        self.site = name if site is None else site
        self.gear = gear
        self.pos = pos
        self.forcerange = forcerange
        self.ctrlrange = ctrlrange
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

class LibConfig:
    """Library configuration class. Contains static vehicle information.
    """
    class Thrusters:
        def __init__(self) -> None:
            # The site positions correspond to Astrobee 
            # (https://ntrs.nasa.gov/citations/20160007769)
            site_positions = {
                "LX+": '0.05 0.025 0.05',
                "LX-": '-0.05 0.025 -0.05',
                "RX+": '0.05 -0.025 -0.05',
                "RX-": '-0.05 -0.025 0.05',
                "AY+": '0.025 0.05 0.05',
                "AY-": '0.025 -0.05 -0.05',
                "FY+": '-0.025 0.05 -0.05',
                "FY-": '-0.025 -0.05 0.05',
                "LZ+": '-0.025 -0.025 0.143',
                "LZ-": '0.025 -0.025 -0.15',
                "RZ+": '0.025 0.025 0.143',
                "RZ-": '-0.025 0.025 -0.15',
            }
            self.thruster1 = Thruster(name='thruster1', gear='1 0 0', pos=site_positions["LX+"])
            self.thruster2 = Thruster(name='thruster2', gear='-1 0 0', pos=site_positions["LX-"])
            self.thruster3 = Thruster(name='thruster3', gear='1 0 0', pos=site_positions["RX+"])
            self.thruster4 = Thruster(name='thruster4', gear='-1 0 0', pos=site_positions["RX-"])
            self.thruster5 = Thruster(name='thruster5', gear='0 1 0', pos=site_positions["AY+"])
            self.thruster6 = Thruster(name='thruster6', gear='0 -1 0', pos=site_positions["AY-"])
            self.thruster7 = Thruster(name='thruster7', gear='0 1 0', pos=site_positions["FY+"])
            self.thruster8 = Thruster(name='thruster8', gear='0 -1 0', pos=site_positions["FY-"])
            self.thruster9 = Thruster(name='thruster9', gear='0 0 1', pos=site_positions["LZ+"])
            self.thruster10 = Thruster(name='thruster10', gear='0 0 -1', pos=site_positions["LZ-"])
            self.thruster11 = Thruster(name='thruster11', gear='0 0 1', pos=site_positions["RZ+"])
            self.thruster12 = Thruster(name='thruster12', gear='0 0 -1', pos=site_positions["RZ-"])


    class Geoms:
        def __init__(self) -> None:
            self.geom_list = [
            Geom(name='cubesat_top', type='mesh', pos='-0.0495 -0.0495 0.1395', euler='0 0 0', meshtype='stl', asset_scale='0.001 0.001 0.001'),
            Geom(name='cubesat_middle', type='mesh', pos='-0.05 -0.05 -0.15', euler='0 0 0', meshtype='stl', asset_scale='0.001 0.001 0.001'),
            Geom(name='cubesat_middle', type='mesh', pos='-0.05 -0.05 -0.0535', euler='0 0 0', meshtype='stl', asset_scale='0.001 0.001 0.001'),
            Geom(name='cubesat_middle', type='mesh', pos='-0.05 -0.05 0.043', euler='0 0 0', meshtype='stl', asset_scale='0.001 0.001 0.001'),
            Geom(name='cubesat_bottom', type='mesh', pos='-0.05 -0.05 -0.15', euler='0 0 0', meshtype='stl', asset_scale='0.001 0.001 0.001'),
            ]