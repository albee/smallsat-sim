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
        max_sim_time = 3600 # max. simulation time
        class noise:
            add_obs_noise = True # add noise to ob of env
            sigma_r = 1e-2
            sigma_q = 1e-3
            sigma_v = 1e-3
            sigma_w = 1e-3

    class renderer:
        start_recording = 0.0 # Start time of the recorded window
        end_recording = 2.0 # Finish time of the recorded window
