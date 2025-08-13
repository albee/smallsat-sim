from smallsat_sim.model.base_model_config import (
    Thruster,
    Geom,
    PhysicalProperties,
    BaseModelConfig,
)


class ModelConfig(BaseModelConfig):
    """
    Config class for the astrobee model.
    """

    # Name of model
    name = "astrobee"

    # Define physical properties of the vehicle
    pp = PhysicalProperties(
        length=0.32,
        width=0.32,
        height=0.32,
        mass=9.583788668,
        diag_inertia=[0.153427995, 0.14271405, 0.162302759],
        com_offset=[0,0,0], #TODO: Check the CoM offset behaves as expected.
    )

    class Thrusters:
        """
        Thrusters class defines the configuration of all thrusters in the astrobee model
        """

        n_thrusters = 12  # Total number of thrusters

        # gear: direction of applied force on body

        site_information = {
            "LX+": {
                "pos": [0.1524, 0.1019, -0.0396],
                "gear": [-1, 0, 0],
                "forcerange": [0, 0.6],
            },
            "LX-": {
                "pos": [-0.1524, 0.1019, 0.0396],
                "gear": [1, 0, 0],
                "forcerange": [0, 0.6],
            },
            "RX+": {
                "pos": [0.1524, -0.1019, 0.0396],
                "gear": [-1, 0, 0],
                "forcerange": [0, 0.6],
            },
            "RX-": {
                "pos": [-0.1524, -0.1019, -0.0396],
                "gear": [1, 0, 0],
                "forcerange": [0, 0.6],
            },
            "AY+": {
                "pos": [-0.0719, 0.1524, -0.0719],
                "gear": [0, -1, 0],
                "forcerange": [0, 0.3],
            },
            "AY-": {
                "pos": [-0.0719, -0.1524, 0.0719],
                "gear": [0, 1, 0],
                "forcerange": [0, 0.3],
            },
            "FY+": {
                "pos": [0.0719, 0.1524, 0.0719],
                "gear": [0, -1, 0],
                "forcerange": [0, 0.3],
            },
            "FY-": {
                "pos": [0.0719, -0.1524, -0.0719],
                "gear": [0, 1, 0],
                "forcerange": [0, 0.3],
            },
            "LZ+": {
                "pos": [-0.0676, 0.1019, 0.1524],
                "gear": [0, 0, -1],
                "forcerange": [0, 0.3],
            },
            "LZ-": {
                "pos": [0.0676, 0.1019, -0.1524],
                "gear": [0, 0, 1],
                "forcerange": [0, 0.3],
            },
            "RZ+": {
                "pos": [0.0676, -0.1019, 0.1524],
                "gear": [0, 0, -1],
                "forcerange": [0, 0.3],
            },
            "RZ-": {
                "pos": [-0.0676, -0.1019, -0.1524],
                "gear": [0, 0, 1],
                "forcerange": [0, 0.3],
            },
        }
        # Create list of thrusters based on dictionary above
        thruster_list = []
        i = 0
        for key, value in site_information.items():
            i += 1
            thruster = Thruster(
                name=f"thruster{i}",
                pos=value["pos"],
                gear=value["gear"],
                forcerange=value["forcerange"],
                ctrlrange=value["forcerange"],
            )
            thruster_list.append(thruster)

    class Geoms:
        """
        Geoms class for defining all geoms in a particular MuJoCo body.
        """

        geom_list = []
