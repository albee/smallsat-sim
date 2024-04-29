from smallsat_sim.model.base_model_config import (
    Thruster,
    Geom,
    PhysicalProperties,
    BaseModelConfig,
)


class ModelConfig(BaseModelConfig):
    """
    Model configuration class. Contains static vehicle information.
    """

    # Name of model
    name = "cubesat"

    # Physical properties of the vehicle (consisting of the above geoms)
    pp = PhysicalProperties(
        length=0.1,
        width=0.1,
        height=0.3,
        mass=3.0,
        diag_inertia=[0.025, 0.025, 0.005],
        com_offset=[0, 0, 0],
    )

    class Thrusters:
        """
        Thrusters class for defining all thrusters in a particular MuJoCo body.
        """

        n_thrusters = 12  # Total number of thrusters
        # Dictionary of thruster site positions.
        # The site positions correspond to Astrobee
        # (https://ntrs.nasa.gov/citations/20160007769)
        site_information = {
            "LX+": {
                "pos": [0.05, 0.025, 0.05],
                "gear": [1, 0, 0],
                "forcerange": [0, 0.4],
            },
            "LX-": {
                "pos": [-0.05, 0.025, -0.05],
                "gear": [-1, 0, 0],
                "forcerange": [0, 0.4],
            },
            "RX+": {
                "pos": [0.05, -0.025, -0.05],
                "gear": [1, 0, 0],
                "forcerange": [0, 0.4],
            },
            "RX-": {
                "pos": [-0.05, -0.025, 0.05],
                "gear": [-1, 0, 0],
                "forcerange": [0, 0.4],
            },
            "AY+": {
                "pos": [0.025, 0.05, 0.05],
                "gear": [0, 1, 0],
                "forcerange": [0, 0.4],
            },
            "AY-": {
                "pos": [0.025, -0.05, -0.05],
                "gear": [0, -1, 0],
                "forcerange": [0, 0.4],
            },
            "FY+": {
                "pos": [-0.025, 0.05, -0.05],
                "gear": [0, 1, 0],
                "forcerange": [0, 0.4],
            },
            "FY-": {
                "pos": [-0.025, -0.05, 0.05],
                "gear": [0, -1, 0],
                "forcerange": [0, 0.4],
            },
            "LZ+": {
                "pos": [-0.025, -0.025, 0.143],
                "gear": [0, 0, 1],
                "forcerange": [0, 0.4],
            },
            "LZ-": {
                "pos": [0.025, -0.025, -0.15],
                "gear": [0, 0, -1],
                "forcerange": [0, 0.4],
            },
            "RZ+": {
                "pos": [0.025, 0.025, 0.143],
                "gear": [0, 0, 1],
                "forcerange": [0, 0.4],
            },
            "RZ-": {
                "pos": [-0.025, 0.025, -0.15],
                "gear": [0, 0, -1],
                "forcerange": [0, 0.4],
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

        geom_list = [
            Geom(
                name="cubesat_top",
                type="mesh",
                pos=[-0.0495, -0.0495, 0.1395],
                euler=[0, 0, 0],
                meshtype="stl",
                asset_scale=[0.001, 0.001, 0.001],
            ),
            Geom(
                name="cubesat_middle",
                type="mesh",
                pos=[-0.05, -0.05, -0.15],
                euler=[0, 0, 0],
                meshtype="stl",
                asset_scale=[0.001, 0.001, 0.001],
            ),
            Geom(
                name="cubesat_middle",
                type="mesh",
                pos=[-0.05, -0.05, -0.0535],
                euler=[0, 0, 0],
                meshtype="stl",
                asset_scale=[0.001, 0.001, 0.001],
            ),
            Geom(
                name="cubesat_middle",
                type="mesh",
                pos=[-0.05, -0.05, 0.043],
                euler=[0, 0, 0],
                meshtype="stl",
                asset_scale=[0.001, 0.001, 0.001],
            ),
            Geom(
                name="cubesat_bottom",
                type="mesh",
                pos=[-0.05, -0.05, -0.15],
                euler=[0, 0, 0],
                meshtype="stl",
                asset_scale=[0.001, 0.001, 0.001],
            ),
        ]
