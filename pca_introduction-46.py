# Finding principal components using SVD on X X^T
unscaled_cov = X.dot(X.T)
U_vcov, S_vcov, VT_vcov = npl.svd(unscaled_cov)
U_vcov
# array([[-0.878298, -0.478114],
# [-0.478114,  0.878298]])
