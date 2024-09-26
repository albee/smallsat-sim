import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import re

# Define label size and font weight for easy customization
labelsize = 22
fontweight = "normal"

SHOW_PLOT = False
WITH_YAW = True
SHOW_REFERENCE = True

xlim = [-1.0, 3.0]
ylim = [-3.0, 1.0]
title = ""
FOLDER = "hardware/2025-02-27"

DEFAULT_REFERENCE = np.array(
    [
        [0.0, 0.0],
        [0, 0],
    ]
)

dist = 0.7
astrobee_height = 0.4
x_shift = 1.5
y_shift = -2.0
r = 0.3

x = 0.6
y_start = -0.5
step = 0.2


x = 0.5
y = -1.0

reference = {
    "data_log_2025-02-15_01-59.*": np.array(
        [
            [-0.4, -1.5],
            [-0.4, 1],
        ]
    ),
    "data_log_2025-02-15_01-39.*": np.array(
        [
            [-0.0, -1.3],
            [-0.1, 0.2],
            [0.3, 0.2],
            [0.3, -0.3],
        ]
    ),
    # "00:33:14 PST-0800.*": np.array(
    "00:33:14 PST-0800.*": np.array(
        [
            [0, -dist, astrobee_height],
            [dist/2, -dist, astrobee_height],
            [dist, -dist, astrobee_height],
            [dist, -dist/2, astrobee_height],
            [dist, 0, astrobee_height],
            [dist, dist/2, astrobee_height],
            [dist, dist, astrobee_height],
            [dist/2, dist, astrobee_height],
            [0, dist, astrobee_height],
            [-dist/2, dist, astrobee_height],
            [-dist, dist, astrobee_height],
            [-dist, dist/2, astrobee_height],
            [-dist, 0, astrobee_height],
            [-dist, -dist/2, astrobee_height],
            [-dist, -dist, astrobee_height],
            [-dist/2, -dist, astrobee_height]
        ]
    ),
    "00:33:14 PST-0800.*": np.array(
        [
            [0.4, -0.6, astrobee_height],
            [0.4, -0.8, astrobee_height],
            [0.4, -1.0, astrobee_height],
            [0.4, -1.2, astrobee_height],
            [0.4, -1.4, astrobee_height],
            [0.4, -1.6, astrobee_height],
            [0.4, -1.8, astrobee_height],
            [0.4, -2.0, astrobee_height],
            [0.4, -2.1, astrobee_height]
        ]
    ),
    "circle.*": np.array(
        [
            [0+x_shift, r+y_shift, astrobee_height],
            [0.75*r+x_shift, 0.75*r+y_shift, astrobee_height],
            [r+x_shift, 0+y_shift, astrobee_height],
            [0.75*r+x_shift, -0.75*r+y_shift, astrobee_height],
            [0+x_shift, -r+y_shift, astrobee_height],
            [-0.75*r+x_shift, -0.75*r+y_shift, astrobee_height],
            [-r+x_shift, 0+y_shift, astrobee_height],
            [-0.75*r+x_shift, 0.75*r+y_shift, astrobee_height],
        ]
    ),
    "line.*": np.array( [
            [x, y_start, astrobee_height],
            [x, y_start-step, astrobee_height],
            [x, y_start-2*step, astrobee_height],
            [x, y_start-3*step, astrobee_height],
            [x, y_start-4*step, astrobee_height],
            [x, y_start-5*step, astrobee_height],
            [x, y_start-6*step, astrobee_height],
            [x, y_start-7*step, astrobee_height],
            [x, y_start-8*step, astrobee_height]
        ]
        ),

        ".*": np.array([
            [x, y-0.01, astrobee_height],
            [x, y-0.02, astrobee_height],
            [x, y-0.01, astrobee_height],
            [x, y, astrobee_height]
        ]
        )
}


files = glob.glob(f"{FOLDER}/*.csv")
for file in files:
    df = pd.read_csv(file)
    if len(df) == 0:
        print(f"Empty file: {file}, Skipping...")
        continue
    # Extract position, velocity, and yaw
    x, y = df["obs_pos_x"], df["obs_pos_y"]
    yaw = df["obs_euler_yaw"]
    vel = np.sqrt(df["obs_vel_x"] ** 2 + df["obs_vel_y"] ** 2 + df["obs_vel_z"] ** 2)
    # ref_pos = df["reference_pos"]   
    # ref_att = df["reference_att"]   
    # ref_x, ref_y = df["ref_x"], df["ref_y"]

    # Define arrow components (unit vectors for direction)
    arrow_length = 0.1  # Adjust length for visi
    u = arrow_length * np.cos(yaw)  # X component of arrow
    v = arrow_length * np.sin(yaw)  # Y component of arrow

    # Create scatter plot with velocity-based colors
    plt.figure(figsize=(10, 8))
    # plt.plot(
    #             # ref[:, 0],
    #             # ref[:, 1],
    #             ref_x,
    #             ref_y,
    #             "--",
    #             label="Reference",
    #             color="red",
    #             alpha=0.8,
    #         )

    if SHOW_REFERENCE:
        for pattern, ref in reference.items():
            if re.match(pattern, os.path.basename(file)):
                plt.plot(
                    ref[:, 0],
                    ref[:, 1],
                    "--",
                    label="Reference",
                    color="red",
                    alpha=0.8,
                )
                break
            else:
                plt.plot(
                    DEFAULT_REFERENCE[:, 0],
                    DEFAULT_REFERENCE[:, 1],
                    label="Reference",
                    color="red",
                )

    sc = plt.scatter(x, y, c=vel, cmap="viridis", alpha=0.7, label="Trajectory")

    if WITH_YAW:
        # Add arrows for yaw direction
        u, v, x, y = u.to_numpy(), v.to_numpy(), x.to_numpy(), y.to_numpy()
        valid = ~np.isnan(u) & ~np.isnan(v) & ~np.isnan(x) & ~np.isnan(y)
        x, y, u, v = x[valid], y[valid], u[valid], v[valid]
        plt.quiver(
            x,
            y,
            u,
            v,
            angles="uv",
            scale_units="xy",
            scale=0.75,
            color="blue",
            width=0.005,
            alpha=0.1,
        )
        # plt.quiver(
        #     x, 
        #     y, 
        #     u, 
        #     v,
        #     angles=yaw,
        #     color="blue",
        # )

    # Set axis limits
    plt.xlim(xlim)
    plt.ylim(ylim)

    # Labels and title with adjustable font size and weight
    plt.xlabel("X Position [m]", fontsize=labelsize, fontweight=fontweight)
    plt.ylabel("Y Position [m]", fontsize=labelsize, fontweight=fontweight)
    plt.title(title, fontsize=labelsize + 2, fontweight=fontweight)

    # Add colorbar
    cbar = plt.colorbar(sc, label="Velocity (m/s)")
    cbar.ax.tick_params(labelsize=labelsize - 2)
    cbar.set_label("Velocity (m/s)", fontsize=labelsize, fontweight=fontweight)

    # Increase tick label size
    plt.xticks(fontsize=labelsize - 2, fontweight=fontweight)
    plt.yticks(fontsize=labelsize - 2, fontweight=fontweight)

    plt.legend()
    plt.grid()

    if SHOW_PLOT:
        plt.show()

    # save plot next to the csv
    output_file = os.path.splitext(file)[0] + ".png"
    output_file = os.path.join(
        os.path.dirname(file), "plots/trajectory", os.path.basename(output_file)
    )
    if not os.path.exists(os.path.dirname(output_file)):
        os.makedirs(os.path.dirname(output_file))
    print("Saving plot to", output_file)
    plt.savefig(output_file)

    # close figure to avoid memory leak
    plt.close()
