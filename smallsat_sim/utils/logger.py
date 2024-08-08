import csv
import os
import pandas as pd
from collections import defaultdict
from typing import Dict, List

from smallsat_sim import SMALLSAT_STEWARD_ROOT_DIR


class SimulationLogger:
    """
    Logger class to record desired quantities when running simulations

    Example usage:
    logger = SimulationLogger(log_dir="/path/to/logs")
    logger.log(timestamp, tracking_error=0.05, mpc_cost=1.2)
    logger.save_log(run_id=0)

    Later in another file for data processing:
    df = logger.load_log(run_id=0)
    print(df)
    """

    def __init__(self, log_name: str = "simulation_log"):
        self.log_dir = os.path.join(SMALLSAT_STEWARD_ROOT_DIR, "logs")
        os.makedirs(self.log_dir, exist_ok=True)
        self.log_name = log_name
        self.logs = defaultdict(list)

    def log(self, timestamp: float, **kwargs):
        """Log multiple quantities with a provided timestamp."""
        for key, value in kwargs.items():
            self.logs[key].append((timestamp, value))

    def save_log(self, run_id: int):
        """Save the log data to a CSV file."""
        log_file = os.path.join(self.log_dir, f"{self.log_name}_run_{run_id}.csv")
        df = pd.DataFrame(
            {key: [v[1] for v in values] for key, values in self.logs.items()}
        )
        df.insert(0, "Timestamp", [v[0] for v in next(iter(self.logs.values()))])
        df.to_csv(log_file, index=False)
        self.reset_logs()

    def reset_logs(self):
        """Reset the logs after saving."""
        self.logs = defaultdict(list)

    def load_log(self, run_id: int) -> pd.DataFrame:
        """Load a saved log file into a pandas DataFrame."""
        log_file = os.path.join(self.log_dir, f"{self.log_name}_run_{run_id}.csv")
        return pd.read_csv(log_file)
