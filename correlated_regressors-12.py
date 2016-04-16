# Fit X_both to signals + noise
B_boths = npl.pinv(X_both).dot(Ys)
