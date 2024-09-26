import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

SHOW_PLOT = False
FOLDER = "hardware/2025-02-27"

files = glob.glob(f"{FOLDER}/*.csv")
for file in files:
    df = pd.read_csv(file)
    if len(df) == 0:
        print(f"Empty file: {file}, Skipping...")
        continue

    # Extract timestamps, solve_time, and ctrl_loop_time
    timestamp, solve_time, control_callback_time = df["Timestamp"], df["solve_time"], df["control_callback_time"]

    # Convert 'timestamp' column from string to datetime
    df["Timestamp"] = pd.to_datetime(df["Timestamp"])  # FIX: Convert before subtraction
    # Convert timestamps to seconds relative to the first timestamp
    seconds = (df["Timestamp"] - df["Timestamp"].iloc[0]).dt.total_seconds()
    # Compute average of the 'value' column
    average_solve_time = df["solve_time"].mean()
    average_solve_control_callback_time= df["control_callback_time"].mean()
    # Calculate moving average with a window of 5
    df['Moving_Average_solve_time'] = df['solve_time'].rolling(window=5).mean()
    df['Moving_Average_control_callback_time'] = df['control_callback_time'].rolling(window=5).mean()

    fig, axs = plt.subplots(2)
    axs[0].plot(df['Moving_Average_solve_time'],  label='Moving avg')
    # axs[0].plot(seconds, solve_time)
    axs[0].plot(np.full_like(seconds, average_solve_time), label=f'Avg: {average_solve_time:.3f}')
    axs[0].set_title('Acados Solver Solve Times')
    axs[0].grid()
    axs[0].legend(loc="upper right")
    axs[1].plot(df['Moving_Average_control_callback_time'], label='Moving avg')
    # axs[1].plot(seconds, control_callback_time)
    axs[1].plot(np.full_like(seconds, average_solve_control_callback_time), label=f'Avg: {average_solve_control_callback_time:.3f}')
    axs[1].set_title('Control Callback Times')
    axs[1].grid()
    axs[1].legend(loc="upper right")

    # Labels and title with adjustable font size and weight
    for ax in axs.flat:
        ax.set(xlabel='Time [s]', ylabel='Solve Time [s]')
    for ax in axs.flat:
        ax.label_outer()

    plt.legend()

    if SHOW_PLOT:
        plt.show()

    # save plot next to the csv
    output_file = os.path.splitext(file)[0] + ".png"
    output_file = os.path.join(
        os.path.dirname(file), "plots/stats", os.path.basename(output_file)
    )
    if not os.path.exists(os.path.dirname(output_file)):
        os.makedirs(os.path.dirname(output_file))
    print("Saving plot to", output_file)
    plt.savefig(output_file)

    # close figure to avoid memory leak
    plt.close()
