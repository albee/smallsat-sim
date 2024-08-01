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
        max_sim_time = 60 # max. simulation time

    class renderer:
        start_recording = 0.0 # Start time of the recorded window
        end_recording = 60.0 # Finish time of the recorded window
