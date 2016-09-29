# Fit X_one to signals + noise
B_ones = npl.pinv(X_one).dot(Ys)
