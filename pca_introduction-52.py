S_from_C = np.sqrt(np.sum(C ** 2, axis=1))
np.allclose(S_from_C, S)
# True
