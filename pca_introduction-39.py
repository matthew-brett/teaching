# Result of projecting on first component, via array dot
u_1 = U[:, 0].reshape((2, 1))  # First component as column vector
c_1 = C[0, :].reshape((1, 50))  # Scalar projections as row vector
projected_onto_1 = u_1.dot(c_1)
# The same as doing the original calculation
np.allclose(projected_onto_1, line_projection(u_best, X))
# True

# Result of projecting on second component, via array dot
u_2 = U[:, 1].reshape((2, 1))  # Second component as column vector
c_2 = C[1, :].reshape((1, 50))  # Scalar projections as row vector
projected_onto_2 = u_2.dot(c_2)
# The same as doing the original calculation
np.allclose(projected_onto_2, line_projection(u_best_orth, X))
# True
