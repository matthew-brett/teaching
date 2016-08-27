# Predicted variance of hrf1 parameter is the same as for the
# model with hrf1 on its own
np.sqrt(C_both.T.dot(npl.pinv(X_both_o.T.dot(X_both_o)).dot(C_both)))
# array([[ 1.485]])
