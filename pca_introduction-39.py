# Result of projecting on first component, via array dot
# np.outer does the equivalent of a matrix multiply of a column vector
# with a row vector, to give a matrix.
projected_onto_1 = np.outer(U[:, 0], C[0, :])
# The same as doing the original calculation
np.allclose(projected_onto_1, line_projection(u_best, X))
# True

# Result of projecting on second component, via np.outer
projected_onto_2 = np.outer(U[:, 1], C[1, :])
# The same as doing the original calculation
np.allclose(projected_onto_2, line_projection(u_best_orth, X))
# True
