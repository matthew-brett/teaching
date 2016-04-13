remaining_ss = []
for u in u_vectors.T: # iterate over columns
    remaining = line_remaining(u, X)
    remaining_ss.append(np.sum(remaining ** 2))
plt.plot(angles, remaining_ss)
# [...]
plt.xlabel('Angle of unit vector')
# <...>
plt.ylabel('Remaining sum of squares')
# <...>
