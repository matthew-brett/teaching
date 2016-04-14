fitted = X.dot(B)
errors = psychopathy - fitted
print(np.sum(errors ** 2))
# 252.92560645
