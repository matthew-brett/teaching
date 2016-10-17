trend = np.linspace(0, 1, 10)
X = np.ones((10, 3))
X[:, 0] = trend
X[:, 1] = trend ** 2
plt.imshow(X)
# <...>
