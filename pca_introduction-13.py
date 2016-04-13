u_guessed_row = u_guessed.reshape(1, 2)  # A row vector
c_values = u_guessed_row.dot(X)  # c values for scaling u
projected = u_guessed_row.T.dot(c_values)
# scale u by values to get projection
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
plt.title("Actual and projected points for guessed $\hat{u}$")
# <...>
