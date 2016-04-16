# The parameter for the hrf1 regressor in the non-orth model
# is correlated with the parameter for the hrf1 regressor
# in the orth model.
plt.plot(B_boths[0], B_boths_o[0], '.')
# [...]
np.corrcoef(B_boths[0], B_boths_o[0])
# array([[ 1.    ,  0.7128],
# [ 0.7128,  1.    ]])
