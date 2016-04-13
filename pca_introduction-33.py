# Calculating projection coefficients with array dot
c_values = components.dot(X)
# Result of projecting on first component, via array dot
u = u_best.reshape(1, 2)  # first component as row vector
c = c_values[0].reshape(1, 50)  # c for first component as row vector
projected_1 = u.T.dot(c)
# The same as doing the original calculation
np.allclose(projected_1, line_projection(u_best, X))
# True
