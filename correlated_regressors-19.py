C_both_1 = np.array([0, 1, 0])[:, None]  # column vector
np.sqrt(C_both_1.T.dot(npl.pinv(X_both.T.dot(X_both)).dot(C_both_1)))
# array([[ 2.0865]])
