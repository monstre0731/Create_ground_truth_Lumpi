import numpy as np

# Given camera coordinate point (x, y, z)
x, y, z = 10, 20, 30  # Example values, replace with actual coordinates

# Given extrinsic matrix
extrinsic_matrix = np.array([
    [-0.9692334269503937, 0.12503988582591893, -0.2120179026276723, -6.947983298624156],
    [0.2415922009707795, 0.3183896488995327, -0.9166576459637081, 46.775224052008134],
    [-0.04711446181471777, -0.9396771032787131, -0.3388025517342455, 17.755323216933267],
    [0.0, 0.0, 0.0, 1.0]
])

# Transform point from camera coordinates to world coordinates
camera_point = np.array([x, y, z, 1])  # Homogeneous coordinates
world_point = np.dot(extrinsic_matrix, camera_point)

# Extract world coordinates from the result
X_w, Y_w, Z_w = world_point[:-1]  # Remove the last element (homogeneous coordinate)

# Print the world coordinates
print(f"World Coordinates (X_w, Y_w, Z_w): ({X_w}, {Y_w}, {Z_w})")
