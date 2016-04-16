# The data projected onto the first component
proj_onto_first = line_projection(u_best, X)
# The data projected onto the second component
proj_onto_second = line_projection(u_best_orth, X)
# Sum of squares in the projection onto the first
ss_in_first = np.sum(proj_onto_first ** 2)
# Sum of squares in the projection onto the second
ss_in_second = np.sum(proj_onto_second ** 2)
# They add up to the total sum of squares
print((ss_in_first, ss_in_second, ss_in_first + ss_in_second))
# (143.97317154347922, 11.696118314873956, 155.66928985835318)
