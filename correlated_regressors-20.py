# Relationship of estimated parameter of hrf1 and hrf2
plt.plot(B_boths[0], B_boths[1], '.')
# [...]
np.corrcoef(B_boths[0], B_boths[1])
# array([[ 1.    , -0.7052],
# [-0.7052,  1.    ]])
