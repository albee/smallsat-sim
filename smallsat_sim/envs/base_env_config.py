class BaseEnvConfig:
    """
    Base class of all environment configurations
    Holds general parameters applicable for all envs
    """
    
    # Viewer options
    class viewer:
        # Standard viewer rendering @50Hz
        viewer_decimation = 10

    class sim:
        dt = 1/500 # simulation runs @500Hz
