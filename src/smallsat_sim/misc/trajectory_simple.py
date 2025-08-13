import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib as mpl

# Update font sizes as specified
mpl.rcParams['font.size'] = 13  # Slightly increase the base font size
mpl.rcParams['axes.labelsize'] = 14  # Slightly increase axis label size
mpl.rcParams['axes.titlesize'] = 15  # Slightly increase plot title size
mpl.rcParams['legend.fontsize'] = 13  # Slightly increase legend font size
mpl.rcParams['xtick.labelsize'] = 13  # Slightly increase x-tick label size
mpl.rcParams['ytick.labelsize'] = 13  # Slightly increase y-tick label size

# List of positions
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
            [-1.5, 12, -2],        # Point 15
            [-2.0, 8, 0], 
            [-2.0, 6.0, 0],         # Point 16 
            [-4.0, 5.5, 0],        # Point 17
            [-10, 5, 0],          # Point 18
            [-18, 10, 0],          # Point 19
            [-18, 0, 0],           # Point 20
            [-18, 0, -5],          # Point 21
            [-8, 0, -5],           # Point 22
            [-3.3, 0, -3.5],       # Point 23
            [-3.3, -9, -3.5],      # Point 24
        ]

# Close the loop by adding the first point at the end
positions.append(positions[0])  # Connect last point to first

# Extract X, Y, Z coordinates
x_coords = [point[0] for point in positions]
y_coords = [point[1] for point in positions]
z_coords = [point[2] for point in positions]

# Create a 3D plot with increased figure size
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Plot the trajectory
ax.plot(x_coords, y_coords, z_coords, color='red', linewidth=2, label='Mission Trajectory for the Inspection Task')

# Mark each point with a sphere and annotate
for i, (x, y, z) in enumerate(zip(x_coords[:-1], y_coords[:-1], z_coords[:-1]), start=1):
    ax.scatter(x, y, z, color='blue', s=50)
    ax.text(x, y, z, f' {i}', size=14)  # Increased annotation size

# Set labels and title
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

# Set equal aspect ratio for all axes
ax.set_box_aspect([
    2*(max(x_coords) - min(x_coords)),
    2*(max(y_coords) - min(y_coords)),
    max(z_coords) - min(z_coords)
])

# Add grid lines
ax.grid(True)

# Adjust viewing angle for better visualization
ax.view_init(elev=45, azim=12)

# Optimize layout
plt.tight_layout()

plt.savefig('ReferenceTrajectory.pdf', format='pdf')

# Show the plot
plt.show()
