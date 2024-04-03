


class Disturbance(object):
    """
    Base class for all disturbances applied to the model.
    Disturbances are defined as an external influence,
    which apply a force or torque on the model.
    """
    def __init__(self) -> None:
        pass

    def apply(self):
        pass