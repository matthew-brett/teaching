# The parameter for the orthogonalized hrf2 regressor is the same as the
# parameter for the non-orthogonalize hrf2 regressor in the
# non-orthogonalized model
plt.plot(B_boths[1], B_boths_o[1], '.')
# [...]
np.allclose(B_boths[1], B_boths_o[1])
# True
