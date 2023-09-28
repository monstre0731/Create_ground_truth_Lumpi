import numpy as np

# Define the input list A
A = [[0, 1, 1, 1], [3, 1, 1, 3], [6, 3, 3]]

# Extract the x-values (first column) and y-values (second column)
x_values = [item[0] for item in A]
y_values = [item[1:] for item in A]

# Convert the lists to NumPy arrays
x_values = np.array(x_values)
y_values = np.array(y_values)

# Define the new x-values you want to interpolate to
new_x_values = np.arange(7)

# Initialize an empty array to store the interpolated results
interpolated_results = []

# Interpolate for each row of y-values
for row in y_values:
    interpolated_row = np.interp(new_x_values, x_values, row)
    interpolated_results.append(interpolated_row)

# Stack the interpolated rows to form the final array
interpolated_array = np.array(interpolated_results)

print(interpolated_array)
