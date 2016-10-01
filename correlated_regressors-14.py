C_one = np.array([1, 0])[:, None]  # column vector
np.sqrt(C_one.T.dot(npl.pinv(X_one.T.dot(X_one)).dot(C_one)))
# array([[ 1.485]])
