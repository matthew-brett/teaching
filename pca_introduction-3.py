# Subtract mean across samples (mean of each variable)
x_mean = X.mean(axis=1)
X[0] = X[0] - x_mean[0]
X[1] = X[1] - x_mean[1]
