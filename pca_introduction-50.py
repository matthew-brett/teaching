scaled_U, scaled_S, scaled_VT = npl.svd(np.cov(X))
np.allclose(scaled_U, U), np.allclose(scaled_VT, VT_vcov)
# (True, True)
