import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D, art3d
import pandas as pd
import numpy as np
import trimesh  # For loading .obj files
import os
import glob
from smallsat_sim.utils.logger import Logger  # Assuming this is your custom module
from smallsat_sim import SMALLSAT_SIM_MODEL_DIR

# Updated list of positions with the new points added
positions = [
    [-3.3, -9, 0],         # Point 1
    [-3.3, -18, 0],        # Point 2
    [-1, -20, 5],          # Point 3
    [16, 0, 23],           # Point 4
    [16, 0, 0],            # Point 5
    [16, 0, -23],          # Point 6
    [16, 0, -26],          # Point 7
    [7, 0, -26],           # Point 8
    [7, 0, -4.5],          # Point 9
    [3, 0, -4.5],          # Point 10
    [3, 0, -8],            # Point 11
    [3.6, 16, -8],         # Point 12
    [3.6, 16, 0],          # Point 13
    [-2.5, 16, 0],         # Point 14
    [-1.5, 10.5, -2],      # Point 15
    [-2.0, 7.75, -1],      # Point 16
    [-5.0, 5.0, 0],        # Point 17
    [-7.5, 5, 0],          # Point 18
    [-18, 10, 0],          # Point 19
    [-18, 0, 0],           # Point 20
    [-18, 0, -5],          # Point 21
    [-8, 0, -5],           # Point 22
    [-3.3, 0, -3.5],       # Point 23
    [-3.3, -9, -3.5],      # Point 24
]

def plot_trajectory_with_pos_and_tube(
    start_point,
    end_point,
    df,
    subsample_step=1,
    meshes=None,
    tube_radius=1.0
):
    import scipy.interpolate as si

    # Validate input
    if start_point < 1 or end_point > len(positions):
        raise ValueError("Start point and end point must be within the range of available points.")
    if start_point > end_point:
        raise ValueError("Start point must be less than or equal to end point.")

    # Adjust indices to be zero-based
    start_idx = start_point - 1
    end_idx = end_point

    # Slice the positions list
    positions_subset = positions[start_idx:end_idx]

    # Convert positions to numpy array for easier calculations
    positions_array = np.array(positions_subset)

    # Increase the number of points along the trajectory for smoother tube
    t = np.linspace(0, 1, len(positions_array))
    t_fine = np.linspace(0, 1, 200)
    positions_fine = np.vstack((
        np.interp(t_fine, t, positions_array[:, 0]),
        np.interp(t_fine, t, positions_array[:, 1]),
        np.interp(t_fine, t, positions_array[:, 2])
    )).T

    # Calculate tangent vectors
    tangents = np.gradient(positions_fine, axis=0)
    tangent_norms = np.linalg.norm(tangents, axis=1)
    tangents_norm = tangents / tangent_norms[:, np.newaxis]

    # Initialize normals using the first tangent
    normals_norm = np.zeros_like(tangents_norm)
    binormals_norm = np.zeros_like(tangents_norm)

    # Choose an arbitrary normal vector perpendicular to the first tangent
    t0 = tangents_norm[0]
    if np.allclose(t0, [0, 0, 1]):
        n0 = np.array([1, 0, 0])
    else:
        n0 = np.cross(t0, [0, 0, 1])
        n0 /= np.linalg.norm(n0)

    normals_norm[0] = n0
    binormals_norm[0] = np.cross(t0, n0)

    # Parallel Transport Frames
    for i in range(1, len(positions_fine)):
        # Calculate rotation axis and angle
        v_prev = tangents_norm[i - 1]
        v_curr = tangents_norm[i]
        cos_theta = np.dot(v_prev, v_curr)
        # Correct numerical errors
        cos_theta = np.clip(cos_theta, -1.0, 1.0)
        angle = np.arccos(cos_theta)
        if angle != 0:
            axis = np.cross(v_prev, v_curr)
            axis_norm = np.linalg.norm(axis)
            if axis_norm != 0:
                axis /= axis_norm
                # Rodrigues' rotation formula
                K = np.array([
                    [0, -axis[2], axis[1]],
                    [axis[2], 0, -axis[0]],
                    [-axis[1], axis[0], 0]
                ])
                R = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * np.dot(K, K)
                normals_norm[i] = R.dot(normals_norm[i - 1])
            else:
                normals_norm[i] = normals_norm[i - 1]
        else:
            normals_norm[i] = normals_norm[i - 1]
        # Recompute binormal
        binormals_norm[i] = np.cross(tangents_norm[i], normals_norm[i])

    # Define the angle around the circle
    theta = np.linspace(0, 2 * np.pi, 20)

    # Generate tube coordinates
    tube_x = []
    tube_y = []
    tube_z = []

    for i in range(len(positions_fine)):
        # Circle in the normal-binormal plane
        circle_x = tube_radius * (normals_norm[i, 0] * np.cos(theta) + binormals_norm[i, 0] * np.sin(theta))
        circle_y = tube_radius * (normals_norm[i, 1] * np.cos(theta) + binormals_norm[i, 1] * np.sin(theta))
        circle_z = tube_radius * (normals_norm[i, 2] * np.cos(theta) + binormals_norm[i, 2] * np.sin(theta))

        # Add circle to the trajectory point
        tube_x.append(positions_fine[i, 0] + circle_x)
        tube_y.append(positions_fine[i, 1] + circle_y)
        tube_z.append(positions_fine[i, 2] + circle_z)

    # Convert lists to numpy arrays
    tube_x = np.array(tube_x)
    tube_y = np.array(tube_y)
    tube_z = np.array(tube_z)

    # Extract X, Y, Z coordinates from df['pos']
    pos_array = df['pos'].to_list()

    # Subsample the positions to avoid too many points
    pos_array_subsampled = pos_array[::subsample_step]

    pos_x = [pos[0] for pos in pos_array_subsampled]
    pos_y = [pos[1] for pos in pos_array_subsampled]
    pos_z = [pos[2] for pos in pos_array_subsampled]

    # Create a new figure for 3D plotting with larger size
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Set the viewing angles
    ax.view_init(elev=45, azim=76, roll=0)

    # Plot the tube (keep-in zone)
    ax.plot_surface(
        tube_x, tube_y, tube_z,
        color='yellow', alpha=0.3, linewidth=0, shade=True, rstride=1, cstride=1
    )

    # Plot the trajectory from positions
    ax.plot(
        positions_array[:, 0],
        positions_array[:, 1],
        positions_array[:, 2],
        marker='o',
        color='blue',
        label='Reference Trajectory',
        markersize=5,
        linewidth=2
    )

    # Plot the pos timeseries from df
    ax.plot(
        pos_x,
        pos_y,
        pos_z,
        marker='^',
        color='red',
        label='Pos Timeseries',
        markersize=5,
        linewidth=2
    )

    # Plot the meshes if provided
    if meshes is not None:
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection

        for mesh in meshes:
            # Extract face colors from material properties if available
            face_colors = None
            if hasattr(mesh.visual, 'face_colors'):
                # Trimesh stores colors in RGBA format (0-255), normalize to 0-1
                face_colors = mesh.visual.face_colors[:, :3] / 255.0  # Ignore alpha channel

            # Create a collection for the mesh
            mesh_collection = Poly3DCollection(mesh.triangles, alpha=1.0)
            if face_colors is not None:
                mesh_collection.set_facecolor(face_colors)
            else:
                mesh_collection.set_facecolor('gray')  # Default color

            # Add the collection to the plot
            ax.add_collection3d(mesh_collection)

    # Annotate each point with its index (relative to the full positions list)
    for idx, (x, y, z) in enumerate(positions_array, start=start_idx):
        ax.text(
            x, y, z, f'{idx+1}',
            fontsize=10,  # Increased font size
            color='black'
        )

    # Set labels and title with increased font sizes
    ax.set_xlabel('X-axis', fontsize=14)
    ax.set_ylabel('Y-axis', fontsize=14)
    ax.set_zlabel('Z-axis', fontsize=14)
    ax.set_title('3D Trajectory Plot', fontsize=16)

    # Add a legend in the top right
    ax.legend(loc='upper right', fontsize=12)

    # Set aspect ratio (optional)
    all_x = np.concatenate([positions_array[:, 0], pos_x, tube_x.flatten()])
    all_y = np.concatenate([positions_array[:, 1], pos_y, tube_y.flatten()])
    all_z = np.concatenate([positions_array[:, 2], pos_z, tube_z.flatten()])
    ax.set_box_aspect([np.ptp(all_x), np.ptp(all_y), np.ptp(all_z)])

    # Save the plot as a PDF
    plt.savefig('3D_Trajectory_Plot.pdf', bbox_inches='tight')

    # Display the plot
    plt.show()

# Example usage:
start_point = 12   # Starting at Point 12
end_point = 18    # Ending at Point 18

# Load your DataFrame using your Logger
logger = Logger()

# Load the pandas DataFrame
df = logger.load_log()

# Subsample step to avoid points being too close
subsample_step = 30  # Adjust this value as needed

# Directory containing your .obj files
obj_folder = os.path.join(SMALLSAT_SIM_MODEL_DIR, "gateway")  # Replace with the path to your folder

# Get a list of all .obj files in the folder
all_obj_files = glob.glob(os.path.join(obj_folder, '*.obj'))

# Exclude files that contain 'gateway_simple' in their filenames
obj_files = [f for f in all_obj_files if 'gateway_simple' not in os.path.basename(f)]

# Load all matching .obj files into a list of meshes
meshes = []
for obj_file in obj_files:
    mesh = trimesh.load(obj_file, process=False)  # Set process=False to preserve materials
    # Optionally apply transformations to each mesh (e.g., scaling, translation)
    # mesh.apply_scale(0.1)
    # mesh.apply_translation([x_offset, y_offset, z_offset])
    meshes.append(mesh)

# Call the plotting function with the list of meshes
plot_trajectory_with_pos_and_tube(
    start_point,
    end_point,
    df,
    subsample_step=subsample_step,
    meshes=meshes,
    tube_radius=1.0
)
