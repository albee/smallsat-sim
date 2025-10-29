import os

try:
    import torch
except ImportError:
    torch = None
else:
    if torch.cuda.is_available():
        torch.cuda.set_per_process_memory_fraction(0.05, device=0)


# Absolute path to root directoy to smallsat-sim folder
SMALLSAT_SIM_ROOT_DIR = os.path.dirname(os.path.realpath(__file__))

# Absolute path to root directory of smallsat-steward repository
SMALLSAT_STEWARD_ROOT_DIR = os.path.join(SMALLSAT_SIM_ROOT_DIR, "..")

# Absolute path to controllers directory in smallsat-sim repository
SMALLSAT_SIM_CONTROLLERS_DIR = os.path.join(SMALLSAT_SIM_ROOT_DIR, "controllers")

# Absolute path to envs directory in smallsat-sim repository
SMALLSAT_SIM_ENVS_DIR = os.path.join(SMALLSAT_SIM_ROOT_DIR, "envs")

# Absolute path to model directory in smallsat-sim repository
SMALLSAT_SIM_MODEL_DIR = os.path.join(SMALLSAT_SIM_ROOT_DIR, "model")
