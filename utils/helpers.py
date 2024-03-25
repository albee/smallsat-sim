# This file includes various helper functions
# Can be included at the beginning of a file the following way:
# from utils.helpers import "function name"

# Parsing
import argparse

def get_args() -> argparse.Namespace:
    """
    This parser includes all non-environment and non-controller specific settings
    """
    # Create the parser
    parser = argparse.ArgumentParser(description='Parse command line inputs')

    # Add arguments
    parser.add_argument('--test', type=int, help='test variable', default=1)
    parser.add_argument('--num_envs', type=int, help='number of envs run in parallel', default=1)

    # Parse the arguments
    args = parser.parse_args()

    return args