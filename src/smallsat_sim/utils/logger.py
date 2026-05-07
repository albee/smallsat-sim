import os
import pandas as pd
from collections import defaultdict
from typing import Optional

from smallsat_sim import SMALLSAT_STEWARD_ROOT_DIR


class Logger(object):
    """
    Logger class to record desired quantities when running simulations.

    Example usage:
    logger = Logger()
    logger.log(run_id, timestamp, tracking_error=0.05, mpc_cost=1.2)
    logger.save_log()

    Later in another file for data processing:
    df = logger.load_log()
    print(df)
    """

    def __init__(self, log_name: Optional[str] = None) -> None:
        self.base_log_dir = os.path.join(SMALLSAT_STEWARD_ROOT_DIR, "logs")

        # Logname only passed when recording data, not for loading
        if log_name:
            self.log_dir = os.path.join(self.base_log_dir, log_name)
            os.makedirs(self.log_dir, exist_ok=True)
            self.log_name = log_name
            self.logs = defaultdict(dict)

    def log(self, run_id: int, timestamp: float, **kwargs) -> None:
        """
        Log multiple quantities with their associated run_id and timestamp.
        If the same run_id and timestamp already exist, it will add the new values
        to the same entry.
        """
        # Include stage/step in the key so different phases (training/eval/etc.)
        # do not overwrite each other when they share the same timestamp.
        stage = kwargs.get("stage")
        step = kwargs.get("step")
        key = (run_id, stage, step, timestamp)

        if key not in self.logs:
            self.logs[key] = {"RunID": run_id, "Timestamp": timestamp}

        # Update the log entry for this run_id and timestamp with the new values
        self.logs[key].update(kwargs)

    def save_log(self) -> None:
        """
        Save the log data to a pickle file, saving in a folder with the run_id as the filename.
        """
        # Extract the run_id for naming the file
        run_id = list(self.logs.values())[0]["RunID"] if self.logs else 0
        log_file = os.path.join(self.log_dir, f"{run_id:04d}.pkl")

        # Convert the dictionary of logs into a DataFrame
        df = pd.DataFrame(list(self.logs.values()))

        # Save the DataFrame using pickle
        df.to_pickle(log_file)

        # Reset the logs after saving
        self.reset_logs()

    def reset_logs(self) -> None:
        """
        Reset the logs after saving to free up memory.
        """
        self.logs = defaultdict(dict)

    def _get_most_recent_log_dir(self) -> Optional[str]:
        """
        Return the path to the most recent log directory in the base logs directory,
        using the fact that folder names are datetime strings.
        """
        # Find all subfolders
        subfolders = [
            f
            for f in os.listdir(self.base_log_dir)
            if os.path.isdir(os.path.join(self.base_log_dir, f))
        ]
        if not subfolders:
            return None
        # Sort subfolders based on their names (datetime strings)
        most_recent_folder = max(subfolders)
        return os.path.join(self.base_log_dir, most_recent_folder)

    def _get_all_pickle_files(self, folder: Optional[str] = None) -> list:
        """
        Return a list of all pickle files in the specified folder, sorted by name.
        If no folder is specified, use the most recent folder.
        """
        if isinstance(folder, str):
            folder = os.path.join(self.base_log_dir, folder)
        elif folder is None:
            folder = self._get_most_recent_log_dir()

        if folder is None:
            return []

        print(f"Loading log files from directory: {folder}")

        pkl_files = [f for f in os.listdir(folder) if f.endswith(".pkl")]
        pkl_files.sort()
        return [os.path.join(folder, f) for f in pkl_files]

    def load_log(
        self,
        folder: Optional[str] = None,
        run_id: Optional[int] = None,
    ) -> pd.DataFrame:
        """
        Load the saved log data from all pickle files in the most recent log directory, concatenating them into a single DataFrame.
        If run_id is specified, filter the DataFrame by that run_id.
        """
        pkl_files = self._get_all_pickle_files(folder=folder)
        pkl_files = self._get_all_pickle_files(folder=folder)

        if not pkl_files:
            raise FileNotFoundError(
                "No log files found in the most recent log directory."
            )

        # Load and concatenate all DataFrames
        df_list = [pd.read_pickle(file) for file in pkl_files]
        df = pd.concat(df_list, ignore_index=True)

        if run_id is not None:
            df = df[df["RunID"] == run_id]

        return df

    def load_all_logs(self, run_id: int | None = None) -> pd.DataFrame:
        """
        Load all saved log data from every log directory in the base log directory, concatenating them
        into a single DataFrame. If run_id is specified, filter the DataFrame by that run_id.
        """
        df_list = []
        # Iterate over each subfolder in the base log directory
        subfolders = [
            os.path.join(self.base_log_dir, folder)
            for folder in os.listdir(self.base_log_dir)
            if os.path.isdir(os.path.join(self.base_log_dir, folder))
        ]
        if not subfolders:
            raise FileNotFoundError(
                "No log directories found in the base log directory."
            )

        for folder in subfolders:
            pkl_files = [
                os.path.join(folder, file)
                for file in os.listdir(folder)
                if file.endswith(".pkl")
            ]
            for file in pkl_files:
                df_list.append(pd.read_pickle(file))

        if not df_list:
            raise FileNotFoundError("No log files found in any log directories.")

        df = pd.concat(df_list, ignore_index=True)

        if run_id is not None:
            df = df[df["RunID"] == run_id]

        return df
