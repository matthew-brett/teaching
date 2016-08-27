# Calculate unscaled variance covariance again
unscaled_cov = X.dot(X.T)
# When divided by N-1, same as result of 'np.cov'
N = X.shape[1]
np.allclose(unscaled_cov / (N - 1), np.cov(X))
# True
