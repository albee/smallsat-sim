from datetime import datetime
import csv
import numpy as np
import h5py

class Logger(object):
    def __init__(self, filename_prefix="mujoco_log"):
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.filename = f"{filename_prefix}_{self.timestamp}.h5"
        self.logged_data = {}
        self.timestamps = []

    def log(self, sim_timestamp, **kwargs):
        """
        Logs the given data at the specified simulation timestamp.
        """
        self.timestamps.append(sim_timestamp)
        
        for key, value in kwargs.items():
            if key not in self.logged_data:
                self.logged_data[key] = []
            self.logged_data[key].append(value)

        # Save to HDF5 if sim_timestamp is larger than 10
        if sim_timestamp > 10:
            self.save_to_hdf5()
            print("Saved data to file.")

    def save_to_hdf5(self):
        """
        Saves the logged data to an HDF5 file.
        """
        with h5py.File(self.filename, 'w') as f:
            f.create_dataset('timestamps', data=np.array(self.timestamps))
            for key, data in self.logged_data.items():
                f.create_dataset(key, data=np.array(data))

    def load_data_from_hdf5(self, hdf5_filename):
        """
        Loads and extracts all logged data from an HDF5 file.
        """
        with h5py.File(hdf5_filename, 'r') as f:
            timestamps = np.array(f['timestamps'])
            logged_data = {key: np.array(f[key]) for key in f.keys() if key != 'timestamps'}
        return timestamps, logged_data