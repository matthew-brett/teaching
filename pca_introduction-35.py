projected_onto_orth = line_projection(u_best_orth, remaining)
np.allclose(projected_onto_orth, remaining)
# True
