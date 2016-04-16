# Relationship of estimated parameters for hrf1 and orthogonalized hrf2
# (they should be independent)
plt.plot(B_boths_o[0], B_boths_o[1], '+')
# [...]
np.corrcoef(B_boths_o[0], B_boths_o[1])
# array([[ 1.    , -0.0053],
# [-0.0053,  1.    ]])
