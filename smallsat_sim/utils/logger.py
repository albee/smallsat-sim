import os
import pandas as pd
from collections import defaultdict
from typing import Dict, List

from smallsat_sim import SMALLSAT_STEWARD_ROOT_DIR


class SimulationLogger:
    """
    Logger class to record desired quantities when running simulations.

    Example usage:
    logger = SimulationLogger()
    logger.log(run_id, timestamp, tracking_error=0.05, mpc_cost=1.2)
    logger.save_log()

    Later in another file for data processing:
    df = logger.load_log()
    print(df)
    """

    def __init__(self, log_name: str = "simulation_log") -> None:
        self.log_dir = os.path.join(SMALLSAT_STEWARD_ROOT_DIR, "logs")
        os.makedirs(self.log_dir, exist_ok=True)
        self.log_name = log_name
        self.logs = defaultdict(list)

    def log(self, run_id: int, timestamp: float, **kwargs) -> None:
        """
        Log multiple quantities with a provided timestamp.
        """
        for key, value in kwargs.items():
            self.logs[key].append((run_id, timestamp, value))

    def save_log(self) -> None:
        """
        Save the log data to a CSV file, appending if the file already exists.
        """
        log_file = os.path.join(self.log_dir, f"{self.log_name}.csv")
        new_data = pd.DataFrame(
            {
                key: [(v[0], v[1], v[2]) for v in values]
                for key, values in self.logs.items()
            }
        )
        new_data.insert(0, "RunID", [v[0] for v in next(iter(self.logs.values()))])
        new_data.insert(1, "Timestamp", [v[1] for v in next(iter(self.logs.values()))])

        if os.path.exists(log_file):
            existing_data = pd.read_csv(log_file)
            combined_data = pd.concat([existing_data, new_data], ignore_index=True)
            combined_data.to_csv(log_file, index=False)
        else:
            new_data.to_csv(log_file, index=False)

        self.reset_logs()

    def reset_logs(self) -> None:
        """
        Reset the logs after saving.
        """
        self.logs = defaultdict(list)

    def load_log(self) -> pd.DataFrame:
        """
        Load a saved log file into a pandas DataFrame.
        """
        log_file = os.path.join(self.log_dir, f"{self.log_name}.csv")
        return pd.read_csv(log_file)
