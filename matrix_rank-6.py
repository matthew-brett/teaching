X_not_full_rank = np.zeros((10, 4))
X_not_full_rank[:, :3] = X
X_not_full_rank[:, 3] = np.dot(X, [-1, 0.5, 0.5])
plt.imshow(X_not_full_rank)
# <...>
