class Thruster:
    """
    Thruster class for defining thruster properties
    Defines both actuator and site properties for the thruster.
    """

    def __init__(
        self,
        name: str,
        pos: list[float],
        gear: list[float],
        site=None,
        forcerange=[0, 1],
        ctrlrange=[0, 0],
        forcelimited="false",
        ctrllimited="false",
        size=0.005,
    ) -> None:
        self.name = name
        self.site = name if site is None else site
        self.gear = gear
        self.pos = pos
        self.forcerange = forcerange
        self.ctrlrange = ctrlrange
        self.ctrllimited = ctrllimited
        self.forcelimited = forcelimited
        self.size = size


class Geom:
    """
    Geom class for defining geom properties
    Especially considers mesh geoms.
    """

    def __init__(
        self,
        name: str,
        type: str,
        pos=None,
        euler="0 0 0",
        meshtype=None,
        asset_scale=[1, 1, 1],
        size=[1, 1, 1],
    ) -> None:
        self.name = name
        self.type = type
        if pos is None:
            self.pos = [0, 0, 0]
            print(
                f"Warning: No position provided for geom {name}. Using default position [0,0,0]."
            )
        else:
            self.pos = pos
        self.euler = euler
        self.size = size
        if self.type == "mesh":
            try:
                assert meshtype is not None
            except AssertionError:
                print(f"Error: No meshtype provided for geom {name}.")
            self.mesh = f"cubesat/meshes/{name}.{meshtype}"
            self.asset_scale = asset_scale


class PhysicalProperties:
    """
    Physical properties class for defining physical properties of the vehicle.
    """

    def __init__(self, length, width, height, density) -> None:
        self.length = length
        self.width = width
        self.height = height
        self.density = density


class BaseModelConfig(object):
    """
    Base class for all model configurations.
    TODO: Implement reusable base class methods/properties if useful.
    """
    pass
