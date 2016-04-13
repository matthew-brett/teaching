# project onto second component direct from data
projected_onto_orth_again = line_projection(u_best_orth, X)
# Gives same answer as projecting remainder from first component
np.allclose(projected_onto_orth_again - projected_onto_orth, 0)
# True
