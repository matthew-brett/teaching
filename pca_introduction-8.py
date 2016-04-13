# Plot points and lines connecting points to origin
plt.scatter(X[0], X[1])
# <...>
for point in X.T:  # iterate over columns
    plt.plot(0, 0)
    plt.plot([0, point[0]], [0, point[1]], 'r:')
# [...]
plt.axis('equal')
# (...)
