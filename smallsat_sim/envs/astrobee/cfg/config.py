class Body:
    def __init__(self, name, pos=[0, 0, 0], quat=[1, 0, 0, 0]) -> None:
        self.name = name
        self.pos = pos
        self.quat = quat


class EnvConfig:
    """Environment configuration class. Contains dynamic vehicle information."""

    model = "cubesat"
    compiler = {"convexhull": "true"}
    visual = {
        "headlight": {
            "ambient": "0.5 0.5 0.5",
            "specular": "0.5 0.5 0.5",
            "diffuse": "0.5 0.5 0.5",
        }
    }

    class Bodies:
        num_bodies = 1
        bodies_list = []
        for i in range(num_bodies):
            bodies_list.append(Body(name=f"body{i}", pos=[i, 0, 10]))

    # Holds all information for the controller in use
    class control:

        # PD controller parameters
        class PD:
            # Decimate the controller frequency such that it doesn't
            # run equally fast to the simulation discretization
            control_decimation = 10

            class gains:
                Kp_x = 20.0
                Kd_x = 20.0
                Kp_q = 1.0
                Kd_q = 2.0
