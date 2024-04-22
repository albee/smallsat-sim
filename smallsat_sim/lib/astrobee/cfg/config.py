from smallsat_sim.lib.base_lib_config import (
    Thruster,
    Geom,
    PhysicalProperties,
    BaseLibConfig,
)


class LibConfig(BaseLibConfig):
    """
    Config class for the astrobee model.
    """

    # Name of model
    name = "astrobee"
    
    # Define physical properties of the vehicle
    pp = PhysicalProperties(length=0.1, width=0.1, height=0.3, density=1000)

    class Thrusters:
        """
        Thrusters class defines the configuration of all thrusters in the astrobee model
        """

        n_thrusters = 12 # Total number of thrusters

        site_information = {
            "LX+": {"pos": [0.1524, 0.1019, -0.0396], "gear": [-1, 0, 0]},
            "LX-": {"pos": [-0.1524, 0.1019, 0.0396], "gear": [1, 0, 0]},
            "RX+": {"pos": [0.1524, -0.1019, 0.0396], "gear": [-1, 0, 0]},
            "RX-": {"pos": [-0.1524, -0.1019, -0.0396], "gear": [1, 0, 0]},
            "AY+": {"pos": [-0.0719, 0.1524, -0.0719], "gear": [0, -1, 0]},
            "AY-": {"pos": [-0.0719, -0.1524, 0.0719], "gear": [0, 1, 0]},
            "FY+": {"pos": [0.0719, 0.1524, 0.0719], "gear": [0, -1, 0]},
            "FY-": {"pos": [0.0719, -0.1524, -0.0719], "gear": [0, 1, 0]},
            "LZ+": {"pos": [-0.0676, 0.1019, 0.1524], "gear": [0, 0, -1]},
            "LZ-": {"pos": [0.0676, 0.1019, -0.1524], "gear": [0, 0, 1]},
            "RZ+": {"pos": [0.0676, -0.1019, 0.1524], "gear": [0, 0, -1]},
            "RZ-": {"pos": [-0.0676, -0.1019, -0.1524], "gear": [0, 0, 1]},
        }
        # Create list of thrusters based on dictionary above
        thruster_list = []
        i = 0
        for key, value in site_information.items():
            i += 1
            thruster = Thruster(
                name=f"thruster{i}", pos=value["pos"], gear=value["gear"], forcerange=[0,10], ctrlrange=[0,10]
            )
            thruster_list.append(thruster)

    class Geoms:
        """
        Geoms class for defining all geoms in a particular MuJoCo body.
        """

        geom_list = []
        