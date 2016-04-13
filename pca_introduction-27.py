projected = line_projection(u_best, X)
plt.scatter(X[0], X[1], label='actual')
# <...>
plt.scatter(projected[0], projected[1], color='r', label='projected')
# <...>
for i in range(X.shape[1]):
    # Plot line between projected and actual point
    proj_pt = projected[:, i]
    actual_pt = X[:, i]
    plt.plot([proj_pt[0], actual_pt[0]], [proj_pt[1], actual_pt[1]], 'k')
# [...]
plt.axis('equal')
# (...)
plt.legend(loc='upper left')
# <...>
plt.title("Actual and projected points for $\hat{u_{best}}$")
# <...>
