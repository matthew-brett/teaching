fitted = X.dot([10, 0.9])
errors = psychopathy - fitted
print(np.sum(errors ** 2))
# 255.75076072
