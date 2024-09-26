from smallsat_sim.model.base_model_config import (
    Thruster,
    Geom,
    PhysicalProperties,
    BaseModelConfig,
)


class ModelConfig(BaseModelConfig):
    """
    Config class for the hardware model.
    """

    # Name of model
    name = "smallsat_hardware_3d"

    # Define physical properties of the vehicle
    pp = PhysicalProperties(
        length=0.5,
        width=0.5,
        height=0.8,
        mass=18.5,
        diag_inertia=[0.51, 0.51, 0.51],
        com_offset=[0,0,0], #TODO: Check the CoM offset behaves as expected.
    )

    class Thrusters:
        """
        Thrusters class defines the configuration of all thrusters in the hardware model
        """

        n_thrusters = 12  # Total number of thrusters

        site_information = {
            "Th1": { 
                "pos": [-0.102, 0.143, -0.35],
                "gear": [1, 0, 0],
                "forcerange": [0, 0.1],
            },   
            "Th2": { 
                "pos": [-0.21, -0.065, -0.35],
                "gear": [1, 0, 0],
                "forcerange": [0, 0.1],
            },  
            "Th3": { 
                "pos": [-0.06, -0.157, -0.35],
                "gear": [0, 1, 0],
                "forcerange": [0, 0.1],
            },
            "Th4": { 
                "pos": [0.06, -0.157, -0.35],
                "gear": [0, 1, 0],
                "forcerange": [0, 0.1],
            },
            "Th5": { 
                "pos": [0.21, -0.065, -0.35],
                "gear": [-1, 0, 0],
                "forcerange": [0, 0.1],
            },
            "Th6": { 
                "pos": [0.102, 0.143, -0.35],
                "gear": [-1, 0, 0],
                "forcerange": [0, 0.1],
            },
            "Th7": { 
                "pos": [0.06, 0.209, -0.35],
                "gear": [0, -1, 0],
                "forcerange": [0, 0.1],
            },
            "Th8": { 
                "pos": [-0.06, 0.209, -0.35],
                "gear": [0, -1, 0],
                "forcerange": [0, 0.1],
            },
            "Th9": { 
                "pos": [0.0, 0.0, 0],
                "gear": [0, 0, 0],
                "forcerange": [0, 0],
            },
            "Th10": { 
                "pos": [0, 0.0, 0],
                "gear": [0, 0, 0],
                "forcerange": [0, 0],
            },
            "Th11": { 
                "pos": [0, 0, 0],
                "gear": [0, 0, 0],
                "forcerange": [0, 0],
            },
            "Th12": { 
                "pos": [0, 0, 0],
                "gear": [0, 0, 0],
                "forcerange": [0, 0],
            }
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
