# The parameter for the hrf1 regressor in the orth model
# is the same as the parameter for the hrf1 regressor in the
# single regressor model
plt.plot(B_ones[0], B_boths_o[0], '.')
# [...]
np.allclose(B_ones[0], B_boths_o[0])
# True
