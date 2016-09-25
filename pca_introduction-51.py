C = U.T.dot(X)
np.allclose(np.diag(S).dot(VT), C)
# True
