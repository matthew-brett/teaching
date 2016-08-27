C_both = np.array([1, 0, 0])[:, None]  # column vector
np.sqrt(C_both.T.dot(npl.pinv(X_both.T.dot(X_both)).dot(C_both)))
# array([[ 2.0861]])
