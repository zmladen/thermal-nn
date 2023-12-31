import numpy as np

# Create a square matrix (4x4 for example)
n = 4
output_size = 3

adj_matrix = np.zeros((n, n), dtype=int)

print("adj_matrix:")
print(adj_matrix)

adj_idx_matrix = np.ones_like(adj_matrix)

# Print the original square matrix
print("adj_idx_matrix:")
print(adj_idx_matrix)

# Get the indices of the upper triangular part
triu_idx = np.triu_indices(n, 1)

print("Indices of the upper triangular part:")
print(triu_idx)

adj_idx_array = adj_idx_matrix[triu_idx].ravel()

print("adj_idx_array")
print(adj_idx_array)

# Calculate the cumulative sum of elements in the upper triangular part
cumulative_sum = np.cumsum(adj_idx_array) - 1

print("Cumulative sum:")
print(cumulative_sum)

# Assign the cumulative sum to the upper triangular elements.

# Like this each connection has unique index!
adj_matrix[triu_idx] = cumulative_sum

# Print the modified square matrix
print("\nModified Adj Matrix:")
print(adj_matrix)

adj_matrix += adj_matrix.T

# Print the modified square matrix
print("\nModified Square Matrix after Transpose:")
print(adj_matrix)


adj_matrix = adj_matrix[:output_size, :]

# Print crop
print("\nModified Square Matrix after crop:")
print(adj_matrix)