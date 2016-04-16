def bonferroni_thresh(alpha_fwe, n):
    return alpha_fwe / n
# ...
print(bonferroni_thresh(0.05, 10))
# 0.005
