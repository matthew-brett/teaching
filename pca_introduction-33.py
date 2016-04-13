# Result of projecting on second component, via array dot
u = u_best_orth.reshape(1, 2)  # second component as row vector
c = c_values[1].reshape(1, 50)  # c for second component as row vector
projected_2 = u.T.dot(c)
# The same as doing the original calculation
np.allclose(projected_2, line_projection(u_best_orth, X))
# True
