# Sum of squares in the projection onto the first
ss_in_first = np.sum(projected_onto_1 ** 2)
# Sum of squares in the projection onto the second
ss_in_second = np.sum(projected_onto_2 ** 2)
# They add up to the total sum of squares
print((ss_in_first, ss_in_second, ss_in_first + ss_in_second))
# (143.97317154347922, 11.696118314873956, 155.66928985835318)
