import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Updated list of positions with the new point added between points 14 and 15
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

# Extract X, Y, Z coordinates
x_coords = [pos[0] for pos in positions]
y_coords = [pos[1] for pos in positions]
z_coords = [pos[2] for pos in positions]

# Create a new figure for 3D plotting
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Plot the trajectory
ax.plot(x_coords, y_coords, z_coords, marker='o')

# Annotate each point with its index
for i, (x, y, z) in enumerate(positions):
    ax.text(x, y, z, f'{i+1}', fontsize=9)

# Set labels and title
ax.set_xlabel('X-axis')
ax.set_ylabel('Y-axis')
ax.set_zlabel('Z-axis')
ax.set_title('3D Trajectory Plot')

# Display the plot
plt.show()
