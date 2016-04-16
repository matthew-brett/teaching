n_tests = np.arange(1, 11)  # n = 1 through 10
# The exact threshold for independent p values
print(sidak_thresh(0.05, n_tests))
# [ 0.05    0.0253  0.017   0.0127  0.0102  0.0085  0.0073  0.0064  0.0057
# 0.0051]
